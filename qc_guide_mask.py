"""QC a hand-painted `mask.guide_regions.regions_mask` BEFORE spending hours on
a registration with it -- both the interpolation (what the sparse keyframes
actually became in 3D) and the pairing (how the painted volume compares to the
atlas structure it will be matched against).

Why this exists, i.e. the two failure modes it is built to catch, both of which
are otherwise completely silent -- the run finishes, the metric goes down, and
the labels just come out wrong:

1. STRAY KEYFRAMES. A single accidental brush click on a plane registers that
   plane as a keyframe with an area of a handful of voxels. mask_utils.
   interpolate_sparse_mask blends signed-distance fields between *consecutive*
   keyframes, so a 4-voxel keyframe wedged between two 500000-voxel ones does
   not "barely matter" -- it collapses the region across both intervals, and
   where the two SDFs disagree by more than their own scale the intermediate
   planes come out EMPTY (the annihilation mode: {(1-t)*d0 + t*d1 < 0} is the
   empty set). Measured on this repo's own s12t_guide6 mask: label 4 had
   keyframes of 4, 1, 1 and 1 voxels and lost 32 of its 70 planes outright,
   ending at 4.7% of the atlas structure's volume.

2. SYSTEMATIC VOLUME MISMATCH. The guide term is MeanSquares between the atlas
   structure and the painted region, so it encodes "these two volumes are the
   same thing". A region drawn at half the atlas structure's size does not give
   a weaker pull toward the right answer, it gives a full-strength pull toward
   a wrong one -- and because SyN is diffeomorphic, that wrong pull drags every
   neighbouring region with it. Random per-plane sloppiness averages out;
   drawing consistently small does not. Below ~50% or above ~200% a guide is
   usually worse than no guide at all.

This is the QC pass for what `paint_mask.py`'s `kind: guide` exports -- run it
between painting and registering.

Usage (antsreg env; --napari needs a display):

    cp configs/qc_guide_mask.example.yaml configs/qc_guide_mask.yaml   # once
    python qc_guide_mask.py                        # uses configs/qc_guide_mask.yaml
    python qc_guide_mask.py configs/other.yaml     # or an explicit one
    python qc_guide_mask.py --napari               # + open the mask in napari
    python qc_guide_mask.py --no-atlas             # interpolation only, no atlas

The config only points at the ../Registration_ants pipeline config for the
sample being audited; every path this tool needs (painted mask, its voxel size,
the raw tiff, the atlas and its orientation/slicing) is read from THAT file
rather than copied here, so the mask is always checked against exactly what the
registration will consume. A copy would drift, and a QC tool that audits a
stale pairing is worse than none.

Writes one PNG per label into out_dir (default: alongside the mask, in a
"<mask stem>_qc" directory): a strip of planes through the region with the
hand-painted keyframes framed in green and the interpolated planes in red, so
the interpolation can be eyeballed against the actual image rather than
trusted. Empty-but-inside-the-range planes are framed in magenta.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from registration_ants import atlas_utils, config as config_mod

import _local_config  # sibling module

TOOL = "qc_guide_mask"

# A keyframe smaller than this fraction of the label's largest keyframe is
# almost certainly a stray click rather than a real tapering end of the
# structure: the taper at a real end is gradual, and the planes on either side
# of a stray one are full size. Deliberately loose -- the point is to surface
# candidates for a human to look at in the PNGs, not to auto-delete anything.
STRAY_KEYFRAME_FRACTION = 0.02
# `rel` (ratio against the median ratio) is what flags a region drawn
# inconsistently with the others -- that is the one that will pull the
# deformation somewhere its neighbours disagree with, and the band below is
# applied to it rather than to the raw ratio.
#
# But do NOT read a uniform shared factor as harmless. It is only absorbed while
# the Affine is driven by image intensity and ignores these masks entirely. Once
# the masks drive the Affine (which on this data is the only way it optimizes at
# all -- intensity MI between sample and DevCCF template is ~1/10 of what the
# optimizer needs, so it stalls after ~10 iterations and returns its input), the
# shared factor stops being absorbed and starts being obeyed: measured on s12t,
# a shape-driven Affine over the 5 usable regions converged to stretch
# [0.66, 0.84, 1.01] -- cube root 0.825 against the masks' own 0.788 -- and left
# 27.3% of the brain unlabelled, versus 14.7% for the intensity Affine. Each
# region has to be drawn to the atlas structure's true anatomical extent, not
# merely consistently with the other regions.
RELATIVE_RATIO_OK = (0.7, 1.4)


def load_painted_mask(path):
    """(z, y, x) int array, as SimpleITK reads it -- axis 0 is the plane index
    a person scrolls through in the painting tool, matching the indices in the
    <mask>.regions.json sidecar."""
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.int32)


def resolve_atlas_ids(label, guide_cfg, sidecar_ids):
    """Same precedence ../Registration_ants pipeline._build_guide_regions_from_labels uses:
    config atlas_ids > config atlas_names > the mask's own .regions.json.
    Returns (kind, value, source_string); kind is "ids" or "names"."""
    ids_cfg = guide_cfg.get("atlas_ids") or {}
    names_cfg = guide_cfg.get("atlas_names") or {}
    if label in ids_cfg:
        return "ids", ids_cfg[label], f"atlas_ids[{label}]"
    if label in names_cfg:
        return "names", names_cfg[label], f"atlas_names[{label}]"
    if label in sidecar_ids:
        return "ids", sidecar_ids[label], f"regions.json.region_ids[{label}]"
    return None, None, "<unpaired>"


def audit_label(dense, label, keyframes):
    """Per-label interpolation report from the dense mask alone.

    keyframes: the plane indices the sidecar says were hand-painted. Everything
    else about a label's 3D extent is interpolation, and that is what is worth
    checking -- the keyframes themselves are by definition what the human drew.
    """
    m = dense == label
    present = np.where(m.any(axis=(1, 2)))[0]
    areas = {int(z): int(m[z].sum()) for z in keyframes}
    biggest = max(areas.values()) if areas else 0
    strays = sorted(z for z, a in areas.items()
                    if biggest and a < STRAY_KEYFRAME_FRACTION * biggest)
    # Planes inside the painted span that came out empty. interpolate_sparse_mask
    # never fills outside [min, max], so only this range is meaningful.
    if keyframes:
        span = range(min(keyframes), max(keyframes) + 1)
        empty_inside = [z for z in span if not m[z].any()]
    else:
        empty_inside = []
    return {
        "voxels": int(m.sum()),
        "z_range": (int(present.min()), int(present.max())) if present.size else None,
        "n_planes": int(present.size),
        "keyframe_areas": areas,
        "strays": strays,
        "empty_inside": empty_inside,
    }


def atlas_volume_mm3(label, guide_cfg, sidecar_ids, annotation_arr, structures, res_um):
    """Volume of the atlas side of this pairing, built exactly the way the
    pipeline builds it (descendants included, atlas_exclude_ids subtracted)."""
    kind, value, source = resolve_atlas_ids(label, guide_cfg, sidecar_ids)
    if kind is None:
        return None, source, {}
    if kind == "ids":
        arr, matched = atlas_utils.build_region_inclusion_mask_by_ids(annotation_arr, structures, value)
    else:
        arr, matched = atlas_utils.build_region_inclusion_mask(annotation_arr, structures, value)
    exclude_cfg = guide_cfg.get("atlas_exclude_ids") or {}
    if label in exclude_cfg:
        drop = np.isin(annotation_arr, list(atlas_utils.descendant_ids_of(structures, exclude_cfg[label])))
        arr = arr & ~drop
    return int(arr.sum()) * (res_um ** 3) / 1e9, source, matched


def _raw_plane(raw_path, z):
    """One plane of the raw stack, read lazily -- the stack is several GB and
    the montage only ever needs a handful of planes."""
    import tifffile
    with tifffile.TiffFile(str(raw_path)) as tf:
        return tf.pages[z].asarray()


def write_montage(dense, label, name, keyframes, raw_path, out_path, n_planes=12, downsample=4):
    """A strip of planes spanning the label's extent, region outline over the
    raw image, frame colour saying where each plane came from."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = dense == label
    present = np.where(m.any(axis=(1, 2)))[0]
    if not present.size:
        return None
    lo, hi = int(present.min()), int(present.max())
    span = range(lo, hi + 1)
    picks = sorted(set(np.linspace(lo, hi, n_planes).round().astype(int).tolist()))
    # Always show the problems, even if the even spacing happened to miss them.
    kf = set(int(z) for z in keyframes)
    empty_inside = [z for z in span if not m[z].any()]
    biggest = max((int(m[z].sum()) for z in kf), default=0)
    forced = [z for z in kf if biggest and int(m[z].sum()) < STRAY_KEYFRAME_FRACTION * biggest]
    picks = sorted(set(picks) | set(forced) | set(empty_inside[:4]))

    cols = min(6, len(picks))
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, z in enumerate(picks):
        ax = axes[i // cols][i % cols]
        img = _raw_plane(raw_path, z)[::downsample, ::downsample]
        plane = m[z][::downsample, ::downsample]
        finite = img[img > 0]
        vmin, vmax = (np.percentile(finite, [1, 99.5]) if finite.size else (0, 1))
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        if plane.any():
            ax.contour(plane.astype(float), levels=[0.5], colors="cyan", linewidths=1.2)
        if z in kf:
            colour, tag = "lime", f"painted  {int(m[z].sum())} vox"
        elif not plane.any():
            colour, tag = "magenta", "EMPTY (interp collapsed)"
        else:
            colour, tag = "red", f"interp  {int(m[z].sum())} vox"
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(colour)
            s.set_linewidth(3)
        ax.set_title(f"z={z}  {tag}", color=colour, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("on")
    fig.suptitle(f"label {label}: {name}   (green=hand-painted, red=interpolated, magenta=empty)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def open_napari(dense, raw_path, voxel_size_um, names):
    """Interpolated mask over the raw stack, with the display scaled by the
    real voxel size -- without `scale` the 2.6/2.6/32um stack is squashed 12x
    in z and every interpolated shape looks wrong for reasons that are purely
    a display artefact."""
    import napari
    import tifffile
    vx, vy, vz = voxel_size_um
    scale = (vz, vy, vx)  # (z, y, x), matching the arrays
    raw = tifffile.memmap(str(raw_path))
    # Contrast from one mid-stack plane rather than letting napari scan the
    # whole (multi-GB, memory-mapped) stack to autoscale.
    mid = np.asarray(raw[raw.shape[0] // 2])
    lo, hi = np.percentile(mid[mid > 0], [1, 99.5]) if (mid > 0).any() else (0, 1)
    viewer = napari.Viewer()
    viewer.add_image(raw, name="raw", scale=scale, colormap="gray",
                     contrast_limits=(float(lo), float(hi)))
    viewer.add_labels(dense, name="guide regions (interpolated)", scale=scale)
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = "  ".join(f"{k}={v}" for k, v in sorted(names.items()))
    napari.run()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    _local_config.add_config_arg(ap, TOOL)
    ap.add_argument("--registration-config",
                    help="audit this ../Registration_ants pipeline config directly, "
                         "ignoring configs/%s.yaml" % TOOL)
    ap.add_argument("--out", help="directory for the montage PNGs (overrides out_dir)")
    ap.add_argument("--napari", action="store_true", help="also open the mask in napari")
    ap.add_argument("--no-atlas", action="store_true",
                    help="skip the atlas-side volume comparison (no antspyx / atlas files needed)")
    ap.add_argument("--no-montage", action="store_true", help="numbers only, no PNGs")
    ap.add_argument("--planes", type=int, default=None, help="planes per montage (default 12)")
    args = ap.parse_args()

    # --registration-config skips this tool's own config entirely; otherwise the
    # toolkit convention applies (configs/<tool>.yaml, ~ and ${VAR} expanded),
    # and its one required key is the pipeline config to audit.
    if args.registration_config:
        reg_config_path = args.registration_config
        local = {}
    else:
        local = _local_config.load_config(TOOL, args.config, required=("registration_config",))
        reg_config_path = local["registration_config"]
    if not Path(reg_config_path).exists():
        sys.exit(f"registration_config does not exist: {reg_config_path}")
    print(f"[audit] {reg_config_path}")

    n_planes = args.planes if args.planes is not None else int(local.get("planes", 12))
    out_override = args.out or local.get("out_dir")

    cfg = config_mod.load_config(reg_config_path)
    guide_cfg = (cfg.get("mask") or {}).get("guide_regions")
    if not isinstance(guide_cfg, dict):
        sys.exit("This config has no dict-form mask.guide_regions (nothing to QC).")

    mask_path = Path(guide_cfg["regions_mask"])
    voxel_size_um = tuple(guide_cfg["voxel_size_um"])
    vox_mm3 = (voxel_size_um[0] * voxel_size_um[1] * voxel_size_um[2]) / 1e9
    dense = load_painted_mask(mask_path)
    sidecar_ids = atlas_utils.load_regions_sidecar_ids(mask_path)

    import json
    sidecar_path = atlas_utils.regions_sidecar_path(mask_path)
    sidecar = json.loads(Path(sidecar_path).read_text()) if Path(sidecar_path).exists() else {}
    annotated = {int(k): v for k, v in (sidecar.get("annotated_slices") or {}).items()}
    names = {int(k): (v[0] if isinstance(v, list) else v)
             for k, v in (sidecar.get("regions") or {}).items()}

    ignored = {int(v) for v in (guide_cfg.get("ignore_labels") or [])}
    painted = sorted(int(v) for v in np.unique(dense) if v != 0)

    print(f"mask     : {mask_path}")
    print(f"grid     : {dense.shape} (z, y, x)   voxel {voxel_size_um} um")
    print(f"labels   : {painted}   ignored by config: {sorted(ignored) or 'none'}")
    print()

    atlas_vols = {}
    if not args.no_atlas:
        atlas_cfg = cfg["atlas"]
        _, annotation = atlas_utils.prepare_custom_atlas(
            atlas_cfg["template_path"], atlas_cfg["annotation_path"], atlas_cfg["resolution_um"],
            orientation=atlas_cfg.get("orientation"), slicing=atlas_cfg.get("slicing"),
            background_margin_voxels=atlas_cfg.get("background_margin_voxels"))
        structures = atlas_utils.load_ccf_ontology_json(atlas_cfg["ontology_path"])
        annotation_arr = annotation.numpy()
        for label in painted:
            if label in ignored:
                continue
            vol, source, _ = atlas_volume_mm3(label, guide_cfg, sidecar_ids, annotation_arr,
                                              structures, atlas_cfg["resolution_um"])
            atlas_vols[label] = (vol, source)

    reports = {label: audit_label(dense, label, annotated.get(label, [])) for label in painted}
    ratios = {}
    for label in painted:
        vol, _ = atlas_vols.get(label, (None, None))
        if vol and label not in ignored:
            ratios[label] = reports[label]["voxels"] * vox_mm3 / vol
    shared = float(np.median(list(ratios.values()))) if ratios else None
    if shared is not None:
        print(f"shared painted/atlas volume factor (median over regions): {shared:.0%}"
              f"   -- the global size difference the Affine is there to absorb;"
              f" 'rel' below is each region against it")
        print()

    hdr = (f"{'lab':>3}  {'region':<26} {'painted':>9} {'atlas':>9} {'ratio':>6} {'rel':>5}  "
           f"{'planes':>6} {'keyfr':>5}  notes        (volumes in mm3)")
    print(hdr)
    print("-" * len(hdr))
    problems = []
    for label in painted:
        rep = reports[label]
        sample_mm3 = rep["voxels"] * vox_mm3
        name = names.get(label, "?")
        vol, _source = atlas_vols.get(label, (None, None))
        ratio = ratios.get(label)
        rel = (ratio / shared) if (ratio is not None and shared) else None
        notes = []
        if label in ignored:
            notes.append("IGNORED by config")
        else:
            if rel is not None and not (RELATIVE_RATIO_OK[0] <= rel <= RELATIVE_RATIO_OK[1]):
                notes.append(f"OFF THE SHARED FACTOR ({rel:.0%})")
                problems.append((label, f"painted at {ratio:.0%} of the atlas structure while the other "
                                        f"regions share {shared:.0%} -- {rel:.0%} of what they agree on"))
            if rep["strays"]:
                notes.append(f"STRAY keyframes z={rep['strays']}")
                problems.append((label, f"stray keyframe(s) at z={rep['strays']} "
                                        f"(areas {[rep['keyframe_areas'][z] for z in rep['strays']]})"))
            if rep["empty_inside"]:
                notes.append(f"{len(rep['empty_inside'])} EMPTY planes inside span")
                problems.append((label, f"{len(rep['empty_inside'])} plane(s) inside the painted span "
                                        f"came out empty: {rep['empty_inside']}"))
        print(f"{label:>3}  {name[:26]:<26} {sample_mm3:>9.2f} "
              f"{(f'{vol:.2f}' if vol is not None else '-'):>9} "
              f"{(f'{ratio:.0%}' if ratio is not None else '-'):>6} "
              f"{(f'{rel:.0%}' if rel is not None else '-'):>5}  "
              f"{rep['n_planes']:>6} {len(rep['keyframe_areas']):>5}  {'; '.join(notes)}")

    print()
    if problems:
        print("PROBLEMS")
        for label, msg in problems:
            print(f"  label {label}: {msg}")
        print()
        print("Fixes, in the order that pays off:")
        print("  * stray keyframe -> erase that plane in paint_mask.py and re-export;")
        print("    a keyframe of a few voxels collapses BOTH intervals it touches.")
        print("  * off the shared factor -> redraw that region to the extent the atlas")
        print("    structure actually has, or drop it into mask.guide_regions.ignore_labels.")
        print("    A guide at half the size the other regions agree on does not pull weakly")
        print("    toward the right answer, it pulls at full strength toward a wrong one, and")
        print("    SyN's diffeomorphism drags the neighbouring regions along with it.")
    else:
        print("No problems found.")

    if not args.no_montage:
        out_dir = (Path(out_override) if out_override
                   else mask_path.parent / (mask_path.name.split(".")[0] + "_qc"))
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = cfg["sample"]["raw_tiff"]
        print()
        for label in painted:
            p = write_montage(dense, label, names.get(label, "?"), annotated.get(label, []),
                              raw_path, out_dir / f"label{label}.png", n_planes=n_planes)
            if p:
                print(f"  wrote {p}")

    if args.napari:
        open_napari(dense, cfg["sample"]["raw_tiff"], voxel_size_um, names)


if __name__ == "__main__":
    main()
