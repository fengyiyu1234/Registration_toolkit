"""Full-resolution, row/column-aligned masks for 2D section annotation."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage


def mask_paths(output_dir, name):
    root = Path(output_dir)
    return {
        "tissue": root / f"{name}_tissue.tif",
        "edited_tissue": root / f"{name}_tissue_edited.tif",
        "damage": root / f"{name}_damage.tif",
        "regions": root / f"{name}_regions.tif",
        "sidecar": root / f"{name}_regions.regions.json",
    }


def validate_layers(image_shape, tissue, damage, regions):
    shape = tuple(image_shape)
    if len(shape) != 2:
        raise ValueError(f"registration image must be 2D, got {shape}")
    for name, layer in (("tissue", tissue), ("damage", damage), ("regions", regions)):
        if np.asarray(layer).ndim != 2 or tuple(np.asarray(layer).shape) != shape:
            raise ValueError(f"{name} must have full image shape {shape}, got {np.asarray(layer).shape}")
    if not np.asarray(tissue).any():
        raise ValueError("tissue mask is empty")
    if np.any(np.asarray(regions) < 0):
        raise ValueError("region labels must be nonnegative")


def compose_masks(image_shape, tissue, damage, regions):
    validate_layers(image_shape, tissue, damage, regions)
    damaged = np.asarray(damage) > 0
    final = (np.asarray(tissue) > 0) & ~damaged
    if not final.any():
        raise ValueError("no tissue remains after excluding damage")
    return final.astype(np.uint8), damaged.astype(np.uint8)


def mask_report(tissue, damage, regions):
    tissue = np.asarray(tissue) > 0
    damage = np.asarray(damage) > 0
    final = tissue & ~damage
    labels, count = ndimage.label(final)
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "tissue_pixels": int(final.sum()),
        "tissue_fraction": float(final.mean()),
        "damage_pixels": int(damage.sum()),
        "damage_inside_tissue_pixels": int((damage & tissue).sum()),
        "region_outside_tissue_pixels": int(((np.asarray(regions) > 0) & ~final).sum()),
        "components": int(count),
        "largest_components": sorted((int(n) for n in sizes), reverse=True)[:10],
    }


def _read_2d(path, shape):
    data = tifffile.imread(path)
    if data.ndim != 2 or tuple(data.shape) != tuple(shape):
        raise ValueError(f"{path}: expected one 2D page of shape {shape}, got {data.shape}")
    return data


def save_session(output_dir, name, image_path, pixel_size_um, channel, z_projection,
                 tissue, damage, regions, assignments, ontology_path,
                 confirm_outside=False, init_parameters=None):
    shape = tuple(np.asarray(tissue).shape)
    final, damaged = compose_masks(shape, tissue, damage, regions)
    report = mask_report(tissue, damage, regions)
    if report["region_outside_tissue_pixels"] and not confirm_outside:
        raise ValueError(f"{report['region_outside_tissue_pixels']} region pixels lie outside final tissue")
    regions = np.asarray(regions)
    if not np.issubdtype(regions.dtype, np.integer):
        raise ValueError("region labels must be integers")
    if regions.max(initial=0) > np.iinfo(np.uint32).max:
        raise ValueError("region label exceeds uint32 TIFF capacity")
    labels = set(int(x) for x in np.unique(regions) if x)
    missing = labels - {int(k) for k, v in assignments.items() if v.get("region_ids")}
    if missing:
        raise ValueError(f"painted labels without ontology assignments: {sorted(missing)}")
    source = Path(image_path).resolve()
    source_stat = source.stat()
    paths = mask_paths(output_dir, name)
    paths["sidecar"].unlink(missing_ok=True)
    paths["tissue"].parent.mkdir(parents=True, exist_ok=True)
    # Write sidecar last. Its presence indicates that all three TIFFs were saved.
    tifffile.imwrite(paths["tissue"], final, photometric="minisblack")
    tifffile.imwrite(paths["edited_tissue"], (np.asarray(tissue) > 0).astype(np.uint8), photometric="minisblack")
    tifffile.imwrite(paths["damage"], damaged, photometric="minisblack")
    tifffile.imwrite(paths["regions"], regions.astype(np.uint32), photometric="minisblack")
    info = {
        "format": "registration_ants.section2d.mask.v1",
        "image": str(source),
        "image_size_bytes": source_stat.st_size,
        "image_mtime_ns": source_stat.st_mtime_ns,
        "shape_yx": list(shape),
        "pixel_size_um": float(pixel_size_um),
        "channel": channel,
        "z_projection": z_projection,
        "ontology_path": str(Path(ontology_path).resolve()),
        "ontology_sha256": hashlib.sha256(Path(ontology_path).read_bytes()).hexdigest(),
        "assignments": {str(int(k)): {"region_ids": [int(i) for i in v["region_ids"]],
                                    "names": list(v["names"])} for k, v in assignments.items() if v},
        "init_parameters": init_parameters or {},
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": report,
    }
    paths["sidecar"].write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths, report


def load_session(output_dir, name, image_path, shape, pixel_size_um, channel,
                 z_projection, ontology_path):
    paths = mask_paths(output_dir, name)
    if not paths["sidecar"].exists():
        if any(paths[key].exists() for key in ("tissue", "edited_tissue", "damage", "regions")):
            raise ValueError(f"mask TIFF exists but session sidecar is missing: {paths['sidecar']}")
        return None
    info = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
    checks = {
        "format": "registration_ants.section2d.mask.v1",
        "image": str(Path(image_path).resolve()),
        "shape_yx": list(shape),
        "pixel_size_um": float(pixel_size_um),
        "channel": channel,
        "z_projection": z_projection,
        "ontology_path": str(Path(ontology_path).resolve()),
        "ontology_sha256": hashlib.sha256(Path(ontology_path).read_bytes()).hexdigest(),
    }
    for key, expected in checks.items():
        if info.get(key) != expected:
            raise ValueError(f"session {key} mismatch: saved {info.get(key)!r}, current {expected!r}")
    source = Path(image_path)
    if info.get("image_size_bytes") != source.stat().st_size or info.get("image_mtime_ns") != source.stat().st_mtime_ns:
        raise ValueError("source image changed since masks were saved")
    tissue = _read_2d(paths["edited_tissue"], shape)
    damage = _read_2d(paths["damage"], shape)
    regions = _read_2d(paths["regions"], shape)
    validate_layers(shape, tissue, damage, regions)
    final, _ = compose_masks(shape, tissue, damage, regions)
    if not np.array_equal(final, _read_2d(paths["tissue"], shape)):
        raise ValueError("exported tissue mask no longer matches editable layers")
    assignments = {int(k): v for k, v in info["assignments"].items()}
    missing = set(int(x) for x in np.unique(regions) if x) - set(assignments)
    if missing:
        raise ValueError(f"painted labels missing assignments in sidecar: {sorted(missing)}")
    empty = [label for label in set(int(x) for x in np.unique(regions) if x)
             if not assignments[label].get("region_ids")]
    if empty:
        raise ValueError(f"painted labels have empty ontology assignments: {sorted(empty)}")
    return tissue.astype(np.uint8), damage.astype(np.uint8), regions.astype(np.uint32), assignments


def auto_mask_raw(raw, pixel_size_um, target_um, threshold=None):
    """The registration auto mask, mapped back by pixel centers to full Y,X."""
    from registration_ants import section2d

    img = section2d.downsample_section(raw, pixel_size_um, target_um)
    mask_xy = section2d.auto_tissue_mask(img, threshold)
    xs = np.rint((np.arange(raw.shape[1]) * pixel_size_um - img.origin[0]) / img.spacing[0]).astype(int)
    ys = np.rint((np.arange(raw.shape[0]) * pixel_size_um - img.origin[1]) / img.spacing[1]).astype(int)
    in_x = (xs >= 0) & (xs < mask_xy.shape[0])
    in_y = (ys >= 0) & (ys < mask_xy.shape[1])
    result = mask_xy[np.ix_(np.clip(xs, 0, mask_xy.shape[0] - 1),
                             np.clip(ys, 0, mask_xy.shape[1] - 1))].T
    result[~in_y, :] = False
    result[:, ~in_x] = False
    return result.astype(np.uint8)

