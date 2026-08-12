"""Interactive tool: paint or hand-correct a mask on a 3D volume. Two
kinds, one shared napari GUI, two different export semantics:

  mask: directly, densely paint an inclusion/exclusion mask -- label 1 =
    include (valid tissue), label 0 (eraser) = exclude. Covers both "mark a
    crack/damage region to exclude" and "touch up an existing mask (e.g.
    the auto-generated brain/tissue mask from
    registration_ants.brain_mask.generate_brain_mask, written when
    mask.auto_brain_mask is set) -- add missed tissue back or erase
    false-positive blobs" with the same paint layer, since both are just
    edits to the same binary mask. No interpolation: every plane you don't
    touch is kept exactly as the starting value, including planes that are
    legitimately all-background (e.g. past the brain's anterior/posterior
    tip) -- a fully-erased plane is a real, meaningful edit, not "not yet
    annotated". Set EXISTING_MASK_PATH to start from an existing mask
    (e.g. the auto-generated one) to touch up; leave it unset to start from
    a blank canvas where everything is included by default (label 0 then
    just marks the parts you want excluded, e.g. a crack). Output goes to
    `mask.sample_damage_mask_path` in a pipeline config.

  guide: paint a rough outline of a structure that's genuinely present in
    both images but needs help being aligned correctly (e.g. a
    bulged/deformed patch of cortex that keeps ending up mapped to
    background). This is NOT an inclusion/exclusion mask -- it marks tissue
    to *actively align*, consumed completely differently, as a paired
    sample+atlas outline feeding ants.registration()'s multivariate_extras
    (see register.register_to_atlas's `guide_regions` parameter and
    ../Registration_ants/scripts/project_outline.py), not as a `mask`/`moving_mask` argument.
    Sparse keyframes only (you paint a handful of representative planes and
    the rest is interpolated between them) -- unlike `mask`'s dense
    editing, since a guide outline is a bounded 3D blob, not a mask that
    needs a meaningful value on every single plane.

    Normally you only paint the SAMPLE side (role="sample"). The atlas side
    no longer has to be drawn by hand: the atlas ships a complete
    annotation volume (P04_DevCCF_Annotations_20um.nii.gz) from which
    Registration_ants builds the matching atlas-side outline by looking the
    region up *by name*. That is what makes the label->region-name mapping
    below load-bearing rather than cosmetic -- it is the only thing tying
    "the blob I painted" to "which atlas structure to pair it with".
    role="atlas" is still supported for the cases where the automatic
    atlas-side region needs to be hand-corrected (set
    EXISTING_MASK_PATH to ../Registration_ants/scripts/project_outline.py's
    output, or to the auto-generated region, as a starting guess instead of
    a blank canvas).

    Several regions at once: the paint layer is a napari Labels layer, so
    label 1/2/3/... are different brush values, one per brain region (see
    `region_labels` in configs/paint_mask.example.yaml). They are exported
    as ONE multi-label volume plus a `.regions.json` sidecar naming each
    label, and each label is interpolated on its own -- see
    interpolate_labels_separately for why they must not be interpolated
    together.

TWO FACTS THIS TOOL DELIBERATELY DOES NOT PAPER OVER

  1. The raw registration.tif carries no voxel size in its header.
     SimpleITK reads spacing=(1.0, 1.0, 1.0) for it even though the real
     voxel size is e.g. [2.6, 2.6, 32.0] um (x, y, z), and the export's
     CopyInformation() copies that same (1,1,1) onto the output -- on
     purpose, so the outline stays on exactly the input's grid. The
     consequence is that NOTHING downstream can learn the voxel size by
     reading either file's header; it has to be passed explicitly. The
     `.regions.json` sidecar says so in writing (voxel_size_um_note), and
     `display_scale_zyx` in the config only affects how napari draws the
     volume on screen, never the exported values (those are voxel indices).

  2. Axis order: both kinds read images via SimpleITK
     (`sitk.GetArrayFromImage`), giving the natural (z,y,x) array order
     with axis 0 = the actual imaging/atlas planes -- deliberately NOT
     `ants.image_read().numpy()`, which gives the reverse axis order for
     the same file (verified against real pipeline output), so axis 0 would
     scroll through a left-right cross-section instead of the actual
     z-planes, and you'd paint on the wrong slices without any error to
     warn you.

Usage (needs a display; runs in the antsreg conda env, which has
napari+PyQt5+SimpleITK alongside antspyx and the pip-installed-editable
registration_ants package this file imports from): edit
configs/paint_mask.yaml (gitignored -- copy it from
configs/paint_mask.example.yaml the first time), then just run the file --
no command-line arguments.

    conda activate antsreg
    python paint_mask.py
    python paint_mask.py configs/paint_mask.guide_s12t.yaml  # 或指定另一份配置

The export logic is separately runnable with no display and no config, on
purely synthetic data (same style as align_masks.py --selftest):

    python paint_mask.py --selftest
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

import _local_config  # sibling module

# napari/PyQt5 are imported lazily by _import_gui() rather than here, and
# mask_utils by _interpolate_sparse_mask(), so that --selftest (pure numpy +
# scipy, no window) runs in the gt_sam env too, which has no
# ../Registration_ants editable install. Both are hard requirements for the
# actual painting GUI, which only ever runs in antsreg.
napari = QLabel = QPushButton = QVBoxLayout = QWidget = None


def _import_gui():
    """Bind the napari/Qt names used by the viewer code. Called once at the
    top of the GUI entry points; import errors surface there rather than at
    module import, which is what keeps --selftest env-independent."""
    global napari, QLabel, QPushButton, QVBoxLayout, QWidget
    import napari as _napari
    from PyQt5.QtWidgets import (QLabel as _QLabel, QPushButton as _QPushButton,
                                 QVBoxLayout as _QVBoxLayout, QWidget as _QWidget)
    napari, QLabel, QPushButton = _napari, _QLabel, _QPushButton
    QVBoxLayout, QWidget = _QVBoxLayout, _QWidget


def _interpolate_sparse_mask():
    """registration_ants.mask_utils.interpolate_sparse_mask -- resolved
    through the editable install of ../Registration_ants (pip install -e
    there puts it on sys.path for the antsreg env), no path hacking needed.
    mask_utils is pure numpy/scipy, so this import does NOT drag in
    antspyx."""
    from registration_ants import mask_utils
    return mask_utils.interpolate_sparse_mask

# 这份配置以前放在仓库根目录，现在统一到 configs/ 下（和其它工具一致）。
# 旧位置仍然能读，只是会打印一条迁移提示。
_LEGACY_CONFIG_PATHS = (Path(__file__).resolve().parent / "paint_mask_local.yaml",)


def _load_local_config(cli_path=None):
    """Paths live in a gitignored configs/paint_mask.yaml instead of constants
    here, so editing them for a new sample never shows up as a git diff."""
    cfg = _local_config.load_config(
        "paint_mask", cli_path=cli_path,
        required=("kind", "image_path", "output_path"),
        legacy_paths=_LEGACY_CONFIG_PATHS)
    return SimpleNamespace(
        kind=cfg["kind"],
        image_path=cfg["image_path"],
        output_path=cfg["output_path"],
        existing_mask_path=cfg.get("existing_mask_path") or None,
        role=cfg.get("role", "sample"),
        region_labels=_normalize_region_labels(cfg.get("region_labels") or {}),
        display_scale_zyx=_normalize_display_scale(cfg.get("display_scale_zyx")),
    )


def _normalize_region_labels(raw):
    """{brush label -> brain region name}, keys forced to int.

    YAML gives `1: cortex` as an int key but `"1": cortex` as a string one,
    and both spellings look identical in the file, so everything downstream
    would silently miss half the mapping if this didn't normalize. Region
    names are passed straight through: whether a name actually resolves in
    the atlas ontology is checked on the Registration_ants side (substring
    match), not here.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"region_labels 应该是 {{label: 脑区名}} 映射，实际是 {type(raw).__name__}")

    normalized = {}
    for key, name in raw.items():
        try:
            label = int(key)
        except (TypeError, ValueError):
            raise ValueError(
                f"region_labels 的 key 必须是画笔 label（整数），得到 {key!r}") from None
        if label < 1:
            raise ValueError(f"region_labels 的 label 必须 >= 1（0 是背景/橡皮），得到 {label}")
        if label in normalized:
            raise ValueError(f"region_labels 里 label {label} 出现了两次（int 和 str key 各一次？）")
        name = str(name).strip()
        if not name:
            raise ValueError(f"region_labels 里 label {label} 的脑区名是空的")
        normalized[label] = name
    return normalized


def _normalize_display_scale(raw):
    """Optional (z, y, x) display scale for the napari layers -- None if unset.

    (z,y,x) to match the array axis order SimpleITK hands back, which is the
    REVERSE of the (x,y,z) order voxel_size_um uses elsewhere in the
    pipeline configs. Display only: the exported mask is still in voxel
    indices and is byte-for-byte unaffected by this.
    """
    if raw is None or raw == "":
        return None
    try:
        scale = [float(v) for v in raw]
    except (TypeError, ValueError):
        raise ValueError(f"display_scale_zyx 应该是三个数 [z, y, x]，实际是 {raw!r}") from None
    if len(scale) != 3:
        raise ValueError(f"display_scale_zyx 需要正好 3 个数 [z, y, x]，得到 {len(scale)} 个")
    if any(v <= 0 for v in scale):
        raise ValueError(f"display_scale_zyx 必须全是正数，得到 {scale}")
    return scale


def _read_sitk_array(path):
    image = sitk.ReadImage(str(path))
    return image, sitk.GetArrayFromImage(image)


def _load_mask_array(path, expected_shape):
    """Read + binarize an existing mask/guess file. Returns None (with a
    warning) if its shape doesn't match -- caller decides the fallback."""
    arr = (sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0).astype(np.uint8)
    if arr.shape != expected_shape:
        print(f"WARNING: existing-mask shape {arr.shape} != image shape {expected_shape}, not pre-filling.")
        return None
    return arr


def _launch_viewer(arr, prefill, title, image_layer_name, mask_layer_name,
                   opacity=None, scale=None):
    """scale: optional (z, y, x) physical size per voxel, applied to BOTH
    layers so they stay registered to each other. Without it a raw
    2.6/2.6/32 um stack is drawn as if it were isotropic, i.e. squashed 12x
    along z, which makes the orthogonal views unusable. Purely a display
    transform -- layer .data, and therefore the export, is untouched."""
    viewer = napari.Viewer(title=title)
    scale_kwargs = {"scale": scale} if scale is not None else {}
    viewer.add_image(arr, name=image_layer_name, colormap="gray", **scale_kwargs)
    labels_kwargs = {"opacity": opacity} if opacity is not None else {}
    paint_layer = viewer.add_labels(prefill.copy(), name=mask_layer_name,
                                    **labels_kwargs, **scale_kwargs)
    return viewer, paint_layer


def _make_export_dock(viewer, status_label, on_export, button_text, panel_name):
    status_label.setWordWrap(True)
    export_btn = QPushButton(button_text)
    export_btn.clicked.connect(on_export)

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(status_label)
    layout.addWidget(export_btn)
    viewer.window.add_dock_widget(dock, area="right", name=panel_name)


# =====================================================================================
# guide export: sparse multi-label keyframes -> dense multi-label volume
# =====================================================================================
MAX_LABEL = 255           # the exported volume is uint8, as is the paint layer

VOXEL_SIZE_UM_NOTE = (
    "Voxel size is NOT in this file's header, by design. The source image (a raw "
    "registration.tif) carries no spacing, so SimpleITK reads spacing=(1,1,1) for it, "
    "and this mask copies the source's geometry verbatim (CopyInformation) so that both "
    "sit on exactly the same grid. The real voxel size (e.g. [2.6, 2.6, 32.0] um in x,y,z) "
    "therefore cannot be recovered from either header and must be passed explicitly by "
    "whatever consumes this mask. All indices in this sidecar are voxel indices along "
    "array axis 0 (the imaging planes, i.e. z), matching sitk.GetArrayFromImage's (z,y,x) order."
)


def sparse_keyframes_by_label(paint_data):
    """{label: {z: 2D bool plane}} for every nonzero brush label present.

    One entry per label per plane the user actually painted that label on;
    planes they never touched are absent, which is what makes the outline
    "sparse keyframes + interpolation" rather than a dense mask (see
    interpolate_labels_separately).
    """
    keyframes = {}
    for z, plane in enumerate(paint_data):
        for label in np.unique(plane):
            label = int(label)
            if label == 0:
                continue
            keyframes.setdefault(label, {})[z] = (plane == label)
    return keyframes


def interpolate_labels_separately(keyframes_by_label, full_shape, interpolate=None):
    """Sparse per-label keyframes -> one dense uint8 multi-label volume.

    Each label is interpolated on its OWN keyframes and only then written
    into the shared output. Handing a multi-label array to
    interpolate_sparse_mask as one binarized blob instead would be wrong,
    not merely lossy. That function interpolates between *consecutive*
    keyframe planes via a signed distance field, and once the labels are
    merged, "consecutive" means consecutive across all regions: region A's
    plane gets blended into region B's plane whenever the two regions'
    keyframes interleave along z, which they normally do (you pick each
    region's own representative planes). The signed-distance blend of two
    cross-sections that don't overlap is empty, so what actually comes out
    is that both regions vanish on every plane between such a pair, and the
    planes that survive are the ones that happened to be bracketed by two
    keyframes of the same region -- silently, with no error. Per-label
    interpolation also confines each region to its own [first plane, last
    plane] span rather than the union's span.
    selftest_per_label_beats_merged_interpolation() measures exactly this.

    Returns a SimpleNamespace:
      volume          uint8 (z,y,x) array, 0 = background
      slices_by_label {label: [z, ...]} planes actually painted, ascending
      voxels_by_label {label: n} voxel count in the final volume (i.e.
                      AFTER overwrites, so these always sum to the nonzero
                      count of `volume`)
      overlap_pairs   {(earlier_label, later_label): n voxels} where two
                      labels' interpolated volumes collided
      n_contested     distinct voxels claimed by more than one label

    Labels are written in ascending order, so on a collision the HIGHER
    label id wins. That is silent in the volume itself, hence overlap_pairs
    / n_contested and the warnings guide_export_warnings() builds from them.
    n_contested is exact; overlap_pairs attributes each contested voxel to
    the pair that collided over it *in write order*, so a voxel claimed by
    three labels is reported as (1,2) and (2,3) rather than also (1,3) --
    enough to point at the regions to go look at, which is the job.
    """
    if interpolate is None:
        interpolate = _interpolate_sparse_mask()

    labels = sorted(keyframes_by_label)
    too_big = [lab for lab in labels if lab > MAX_LABEL or lab < 1]
    if too_big:
        raise ValueError(f"label 必须在 1..{MAX_LABEL} 之间（导出是 uint8），得到 {too_big}")

    volume = np.zeros(full_shape, dtype=np.uint8)
    contested = None
    overlap_pairs = {}

    for label in labels:
        painted = interpolate(keyframes_by_label[label], full_shape)
        # Compare against what is already claimed BEFORE overwriting, so the
        # collision is attributable to a specific pair of labels. Only one
        # interpolated volume is alive at a time here: at the real 2273x3974x157
        # this loop is already several GB per array.
        clash = np.logical_and(painted, volume != 0)
        if clash.any():
            prior, counts = np.unique(volume[clash], return_counts=True)
            for other, count in zip(prior, counts):
                key = (int(other), label)
                overlap_pairs[key] = overlap_pairs.get(key, 0) + int(count)
            contested = clash if contested is None else np.logical_or(contested, clash)
        volume[painted] = label

    return SimpleNamespace(
        volume=volume,
        slices_by_label={lab: sorted(keyframes_by_label[lab]) for lab in labels},
        voxels_by_label={lab: int(np.count_nonzero(volume == lab)) for lab in labels},
        overlap_pairs=overlap_pairs,
        n_contested=int(contested.sum()) if contested is not None else 0,
    )


def _label_name(label, region_labels):
    return region_labels.get(label, "unnamed")


def guide_export_warnings(result, region_labels):
    """Everything worth shouting about in an export, as a list of strings.

    All of these are warnings, never refusals: painting a few regions today
    and the rest tomorrow is a normal way to use this tool, and refusing to
    write the file would just lose the work already done.
    """
    warnings = []
    painted = set(result.slices_by_label)
    named = set(region_labels)

    for label in sorted(painted):
        planes = result.slices_by_label[label]
        if len(planes) < 2:
            warnings.append(
                f"label {label} ({_label_name(label, region_labels)}) was painted on only "
                f"{len(planes)} plane ({planes}) -- there is nothing to interpolate between, "
                f"so it exports as that single flat slice, not a volume. Paint at least 2 planes.")

    if result.n_contested:
        breakdown = "; ".join(
            f"{a} ({_label_name(a, region_labels)}) vs {b} ({_label_name(b, region_labels)}): "
            f"{n} voxels"
            for (a, b), n in sorted(result.overlap_pairs.items()))
        warnings.append(
            f"{result.n_contested} voxels are claimed by more than one label after "
            f"interpolation; the higher label id silently wins there. Overlaps: {breakdown}")

    for label in sorted(named - painted):
        warnings.append(
            f"region_labels lists label {label} ({region_labels[label]}) but nothing was "
            f"painted with it -- that region has no outline in this export.")

    # A single unnamed label with no region_labels at all is the original
    # one-region-per-file usage, not a mistake -- don't nag about it.
    legacy_single_region = not region_labels and painted == {1}
    if not legacy_single_region:
        for label in sorted(painted - named):
            warnings.append(
                f"label {label} was painted but has no region_labels entry -- nothing "
                f"downstream can tell which atlas region to pair it with, so this outline "
                f"cannot be used.")
    return warnings


def _output_stem(output_path):
    """Path with the image suffix removed, for hanging sidecars off.
    .nii.gz is special-cased the same way edit_sample_labels.py's
    _annotation_sidecar_path and annotate_gt_sam.py's sidecar_path_for do
    it, so the names line up with the sidecar convention already in use."""
    path = Path(output_path)
    name = path.name
    name = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else Path(name).stem
    return path.with_name(name)


def write_guide_sidecars(output_path, image_path, result, region_labels, role, total_z,
                         spacing_xyz=None):
    """Write the two sidecars next to the exported outline, and return their paths.

    <stem>.regions.json is the one that matters for this tool: it is the
    only record of which brush label is which brain region, and
    Registration_ants needs exactly that to pull the matching region out of
    the atlas annotation volume by name.

    <stem>.annotated_slices.json is the repo's pre-existing per-mask
    sidecar (written by edit_sample_labels.py and annotate_gt_sam.py, read
    by registration_eval.py's load_region_annotation_hint) -- same
    {"hand_drawn_slices": [...]} shape, holding the union over all labels
    of the planes actually painted. It is written so this output drops into
    the evaluation path unchanged; the per-label breakdown that format has
    no room for lives in .regions.json.
    """
    stem = _output_stem(output_path)
    regions_path = stem.with_name(stem.name + ".regions.json")
    slices_path = stem.with_name(stem.name + ".annotated_slices.json")

    all_planes = sorted({z for planes in result.slices_by_label.values() for z in planes})
    regions_path.write_text(json.dumps({
        "regions": {str(lab): region_labels[lab] for lab in sorted(region_labels)},
        "annotated_slices": {str(lab): planes
                             for lab, planes in sorted(result.slices_by_label.items())},
        "image_path": str(image_path),
        "mask_path": str(output_path),
        "role": role,
        "total_z": int(total_z),
        "header_spacing_xyz": list(spacing_xyz) if spacing_xyz is not None else None,
        "voxel_size_um_note": VOXEL_SIZE_UM_NOTE,
    }, indent=2))
    slices_path.write_text(json.dumps({
        "hand_drawn_slices": all_planes,
        "total_z": int(total_z),
        "regions": {str(lab): region_labels[lab] for lab in sorted(region_labels)},
    }, indent=2))
    return regions_path, slices_path


def _run_mask(args):
    _import_gui()
    sample_sitk, arr = _read_sitk_array(args.sample_path)

    prefill = np.ones(arr.shape, dtype=np.uint8)
    if args.existing_mask:
        loaded = _load_mask_array(args.existing_mask, arr.shape)
        if loaded is not None:
            prefill = loaded

    viewer, mask_layer = _launch_viewer(
        arr, prefill, "Paint/edit mask", "sample", "mask (edit here)", opacity=0.4)

    status_label = QLabel(
        "Paint label 1 to include tissue, label 0 (eraser) to exclude\n"
        "(crack/damage/background), on whichever slices need it. Then click Export.")

    def export():
        edited = (mask_layer.data > 0).astype(np.uint8)
        n_changed = int(np.sum(edited != prefill))

        out_sitk = sitk.GetImageFromArray(edited)
        out_sitk.CopyInformation(sample_sitk)
        sitk.WriteImage(out_sitk, args.output_path)

        msg = (f"Wrote {args.output_path}\n"
               f"Coverage: {100 * edited.mean():.1f}%\n"
               f"Voxels changed from starting mask: {n_changed}")
        status_label.setText(msg)
        print(msg)

    _make_export_dock(viewer, status_label, export, "Export Mask", "Mask Export")


def _region_legend(region_labels):
    """The label -> region-name mapping, shown in the side panel so the
    brush number you're about to paint with is never a guess."""
    if not region_labels:
        return ("No region_labels in the config: paint one region with label 1.\n"
                "For several regions, add region_labels to the config first --\n"
                "an unnamed label cannot be paired with an atlas region.\n")
    lines = "\n".join(f"  label {lab} = {region_labels[lab]}"
                      for lab in sorted(region_labels))
    return f"Brush label -> brain region:\n{lines}\n"


def _run_guide(args):
    _import_gui()
    base_sitk, arr = _read_sitk_array(args.image_path)

    prefill = np.zeros(arr.shape, dtype=np.uint8)
    if args.existing_mask:
        loaded = _load_mask_array(args.existing_mask, arr.shape)
        if loaded is not None:
            prefill = loaded

    viewer, paint_layer = _launch_viewer(
        arr, prefill, f"Paint guide outline ({args.role})", args.role,
        "guide outline (paint here)", scale=args.display_scale_zyx)

    guess_note = "Pre-filled with the existing mask -- adjust/redraw as needed.\n" if args.existing_mask else ""
    status_label = QLabel(
        _region_legend(args.region_labels) +
        "Paint a rough outline on a few planes per region (start, end, and\n"
        "any plane where the shape changes a lot; at least 2 planes each),\n"
        "then click Export.\n" + guess_note)

    def export():
        keyframes = sparse_keyframes_by_label(paint_layer.data)
        if not keyframes:
            status_label.setText("No planes painted yet -- nothing to export.")
            return

        n_planes = sum(len(planes) for planes in keyframes.values())
        status_label.setText(
            f"Exporting... ({len(keyframes)} labels, {n_planes} painted planes)")
        result = interpolate_labels_separately(keyframes, arr.shape)

        out_sitk = sitk.GetImageFromArray(result.volume)
        out_sitk.CopyInformation(base_sitk)      # keeps the source's (1,1,1) -- see module docstring
        sitk.WriteImage(out_sitk, args.output_path)
        regions_path, slices_path = write_guide_sidecars(
            args.output_path, args.image_path, result, args.region_labels, args.role,
            arr.shape[0], spacing_xyz=base_sitk.GetSpacing())

        lines = [f"Wrote {args.output_path}", f"Wrote {regions_path}", f"Wrote {slices_path}"]
        for label in sorted(result.slices_by_label):
            lines.append(
                f"  label {label} ({_label_name(label, args.region_labels)}): "
                f"{len(result.slices_by_label[label])} painted planes "
                f"{result.slices_by_label[label]} -> {result.voxels_by_label[label]} voxels")
        lines += [f"WARNING: {w}" for w in guide_export_warnings(result, args.region_labels)]

        msg = "\n".join(lines)
        status_label.setText(msg)
        print(msg)

    _make_export_dock(viewer, status_label, export, "Export Outline", "Guide Outline Export")


# =====================================================================================
# selftests -- synthetic arrays only, no GUI, no config, no image on disk
# =====================================================================================
def _reference_interpolate_sparse_mask(keyframe_planes, full_shape):
    """Deliberate standalone minimal copy of
    registration_ants.mask_utils.interpolate_sparse_mask, used by the
    selftests ONLY when registration_ants isn't importable -- it lives in
    ../Registration_ants, which is pip-installed-editable in the antsreg
    env but absent from gt_sam, and these tests are meant to run in both
    (same reasoning as align_masks.py's copy of the Dice/surface code).
    selftest_interpolator_matches_registration_ants() asserts the two agree
    voxel-for-voxel whenever the real one IS available, so drift can't hide.
    """
    from scipy import ndimage

    def sdf(plane):
        return (ndimage.distance_transform_edt(~plane)
                - ndimage.distance_transform_edt(plane))

    dense = np.zeros(full_shape, dtype=bool)
    indices = sorted(keyframe_planes)
    for idx in indices:
        dense[idx] = keyframe_planes[idx]
    for i0, i1 in zip(indices[:-1], indices[1:]):
        if i1 - i0 <= 1:
            continue
        sdf0, sdf1 = sdf(keyframe_planes[i0]), sdf(keyframe_planes[i1])
        for idx in range(i0 + 1, i1):
            t = (idx - i0) / (i1 - i0)
            dense[idx] = (1 - t) * sdf0 + t * sdf1 <= 0
    return dense


def _selftest_interpolator():
    """The real interpolator when the antsreg install is there, the local
    copy otherwise (see _reference_interpolate_sparse_mask)."""
    try:
        return _interpolate_sparse_mask()
    except ImportError:
        print("   (registration_ants not importable here -- using the local reference "
              "interpolator; run this in antsreg to test against the real one)")
        return _reference_interpolate_sparse_mask


SHAPE = (16, 40, 40)


def _canvas(shape=SHAPE):
    return np.zeros(shape, dtype=np.uint8)


def _box(canvas, planes, label, y0, y1, x0, x1):
    for z in planes:
        canvas[z, y0:y1, x0:x1] = label
    return canvas


def _extent(volume, label):
    """(z, y, x) min/max bounds of one label, for asserting a region stayed
    where it was painted."""
    idx = np.argwhere(volume == label)
    assert idx.size, f"label {label} is missing from the export entirely"
    return idx.min(axis=0), idx.max(axis=0)


def selftest_three_labels_stay_separate(interp):
    print("1. three regions, 3 keyframe planes each -> three intact, non-bleeding labels")
    canvas = _canvas()
    _box(canvas, [0, 4, 8], 1, 2, 10, 2, 10)        # 8x8 = 64 px/plane, spans z 0..8
    _box(canvas, [2, 6, 10], 2, 20, 30, 4, 12)      # 10x8 = 80 px/plane, spans z 2..10
    _box(canvas, [5, 9, 13], 3, 30, 38, 25, 35)     # 8x10 = 80 px/plane, spans z 5..13

    result = interpolate_labels_separately(sparse_keyframes_by_label(canvas), SHAPE,
                                           interpolate=interp)
    vol = result.volume

    assert vol.dtype == np.uint8, vol.dtype
    # No label may invent a value the config never mentioned (a merged
    # interpolation, or an off-by-one in the write-back, shows up here).
    assert set(np.unique(vol).tolist()) == {0, 1, 2, 3}, np.unique(vol)

    # Constant cross-sections interpolate to themselves, so the counts are
    # exact rather than "in a plausible range": area * (last - first + 1).
    assert result.voxels_by_label == {1: 64 * 9, 2: 80 * 9, 3: 80 * 9}, result.voxels_by_label
    assert result.slices_by_label == {1: [0, 4, 8], 2: [2, 6, 10], 3: [5, 9, 13]}, \
        result.slices_by_label

    # Each label filled exactly its own z span and its own footprint --
    # nothing leaked into a neighbour's box or past its own keyframes.
    for label, (zlo, zhi), (ylo, yhi), (xlo, xhi) in [
            (1, (0, 8), (2, 9), (2, 9)),
            (2, (2, 10), (20, 29), (4, 11)),
            (3, (5, 13), (30, 37), (25, 34))]:
        lo, hi = _extent(vol, label)
        assert tuple(lo) == (zlo, ylo, xlo) and tuple(hi) == (zhi, yhi, xhi), \
            f"label {label} extent {lo}..{hi} != {(zlo, ylo, xlo)}..{(zhi, yhi, xhi)}"

    assert result.n_contested == 0 and not result.overlap_pairs, result.overlap_pairs
    assert guide_export_warnings(result, {1: "a", 2: "b", 3: "c"}) == []
    print("   ok")


def selftest_per_label_beats_merged_interpolation(interp):
    print("2. per-label vs one merged interpolation, on two adjacent regions")
    # Region 1 on the left (planes 0, 8), region 2 on the right (planes 2, 6):
    # the two labels' keyframes interleave along z, which is the normal case
    # when you pick each region's own representative planes.
    canvas = _canvas()
    _box(canvas, [0, 8], 1, 5, 15, 2, 12)
    _box(canvas, [2, 6], 2, 5, 15, 28, 38)

    result = interpolate_labels_separately(sparse_keyframes_by_label(canvas), SHAPE,
                                           interpolate=interp)
    vol = result.volume

    # Per-label: each region fills its own z span with its own footprint,
    # and neither one appears in the strip between them.
    lo1, hi1 = _extent(vol, 1)
    lo2, hi2 = _extent(vol, 2)
    assert (lo1[0], hi1[0]) == (0, 8) and (lo1[2], hi1[2]) == (2, 11), (lo1, hi1)
    assert (lo2[0], hi2[0]) == (2, 6) and (lo2[2], hi2[2]) == (28, 37), (lo2, hi2)
    assert not np.any(vol[:, :, 12:28]), "per-label interpolation leaked into the gap"
    assert result.voxels_by_label == {1: 100 * 9, 2: 100 * 5}, result.voxels_by_label

    # The merged version: exactly what `keyframes = {z: data[z] > 0}` did
    # before -- one binary blob, so consecutive keyframes belonging to
    # DIFFERENT regions get blended into each other. Their cross-sections
    # don't overlap, so the blend comes out empty and BOTH regions disappear
    # from every plane bracketed by a mismatched keyframe pair.
    merged_keyframes = {z: (canvas[z] > 0) for z in range(SHAPE[0]) if np.any(canvas[z])}
    merged = interp(merged_keyframes, SHAPE)

    for z in (1, 7):                # bracketed by region 1 and region 2 keyframes
        assert not np.any(merged[z]), f"merged: plane {z} was expected to be annihilated"
        assert np.count_nonzero(vol[z]) == 100, f"per-label: region 1 missing on plane {z}"
    # Plane 4 sits between region 2's keyframes (2 and 6), so the merged run
    # keeps region 2 there and drops region 1 -- the whole plane ends up
    # attributed to the region that happened to bracket it.
    assert np.count_nonzero(merged[4, :, 2:12]) == 0, "merged: region 1 survived on plane 4"
    assert np.count_nonzero(merged[4, :, 28:38]) == 100
    assert np.count_nonzero(vol[4, :, 2:12]) == 100 and np.count_nonzero(vol[4, :, 28:38]) == 100

    lost = np.logical_and(vol > 0, ~merged)
    assert int(lost.sum()) == 700, int(lost.sum())          # all of region 1's planes 1..7
    assert not np.any(np.logical_and(merged, vol == 0)), "merged produced outline nobody painted"
    print(f"   ok (merging the labels loses {int(lost.sum())} voxels of region 1; per-label: 0)")


def selftest_single_plane_label_warns(interp):
    print("3. a label painted on only one plane -> warning, but still exported")
    canvas = _canvas()
    _box(canvas, [0, 6], 1, 2, 10, 2, 10)
    _box(canvas, [4], 3, 20, 28, 20, 28)            # one plane only: nothing to interpolate

    result = interpolate_labels_separately(sparse_keyframes_by_label(canvas), SHAPE,
                                           interpolate=interp)
    assert result.slices_by_label[3] == [4]
    assert result.voxels_by_label[3] == 64, result.voxels_by_label
    assert result.voxels_by_label[1] == 64 * 7, result.voxels_by_label

    warnings = guide_export_warnings(result, {1: "cortex", 3: "corpus callosum"})
    flat = [w for w in warnings if "label 3" in w and "1 plane" in w]
    assert flat, warnings
    assert "corpus callosum" in flat[0], flat[0]
    assert not any("label 1" in w for w in warnings), warnings
    print("   ok")


def selftest_overlap_is_counted_and_reported(interp):
    print("4. two overlapping regions -> exact contested voxel count, named pair")
    # The overlap can only ever come from the INTERPOLATION: a single paint
    # layer can't hold two labels on one pixel, so no two keyframe planes
    # can disagree. Here label 1's keyframes bracket label 2's, and their
    # interpolated bodies pass through each other on planes 2..6.
    canvas = _canvas()
    _box(canvas, [0, 8], 1, 5, 15, 5, 15)           # 100 px/plane, spans z 0..8
    _box(canvas, [2, 6], 2, 10, 20, 10, 20)         # 100 px/plane, spans z 2..6
    #                                                 intersection y 10:15 * x 10:15 = 25 px

    result = interpolate_labels_separately(sparse_keyframes_by_label(canvas), SHAPE,
                                           interpolate=interp)
    expected = 25 * 5                                # 25 px on each of planes 2..6
    assert result.overlap_pairs == {(1, 2): expected}, result.overlap_pairs
    assert result.n_contested == expected, result.n_contested

    # The later (higher) label wins the contested voxels, silently -- which is
    # exactly why it gets reported.
    assert np.all(result.volume[2:7, 10:15, 10:15] == 2)
    assert result.voxels_by_label[1] == 100 * 9 - expected, result.voxels_by_label
    assert result.voxels_by_label[2] == 100 * 5, result.voxels_by_label

    warnings = guide_export_warnings(result, {1: "cortex", 2: "cerebellar hemisphere"})
    overlap = [w for w in warnings if "claimed by more than one label" in w]
    assert overlap, warnings
    assert str(expected) in overlap[0] and "cerebellar hemisphere" in overlap[0], overlap[0]
    print(f"   ok ({expected} contested voxels, reported as 1 vs 2)")


def selftest_unnamed_and_unpainted_labels_warn(interp):
    print("5. region_labels/canvas mismatches in both directions -> warnings")
    canvas = _canvas()
    _box(canvas, [0, 6], 1, 2, 10, 2, 10)
    _box(canvas, [1, 7], 4, 20, 28, 20, 28)         # painted but not in region_labels
    result = interpolate_labels_separately(sparse_keyframes_by_label(canvas), SHAPE,
                                           interpolate=interp)

    warnings = guide_export_warnings(result, {1: "cortex", 2: "cerebellar hemisphere"})
    assert any("label 2" in w and "nothing was" in w for w in warnings), warnings
    assert any("label 4" in w and "region_labels entry" in w for w in warnings), warnings
    print("   ok")


def selftest_single_label_matches_old_behaviour(interp):
    print("6. one label, no region_labels -> bit-identical to the pre-multi-label export")
    rng = np.random.default_rng(0)
    canvas = _canvas()
    # Irregular blobs, not boxes: the signed-distance interpolation has real
    # work to do between keyframes, so an accidental change in how keyframes
    # are collected would show up as a diff.
    for z, (cy, cx, r) in zip([1, 5, 6, 12], [(15, 15, 7), (22, 18, 10), (20, 20, 4), (12, 25, 8)]):
        yy, xx = np.ogrid[:SHAPE[1], :SHAPE[2]]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        blob |= rng.random(blob.shape) > 0.995      # a bit of speckle
        canvas[z][blob] = 1

    # Verbatim the old code path: one binary keyframe dict, one interpolation.
    old_keyframes = {z: (canvas[z] > 0) for z in range(SHAPE[0]) if np.any(canvas[z])}
    old = interp(old_keyframes, SHAPE).astype(np.uint8)

    new = interpolate_labels_separately(sparse_keyframes_by_label(canvas), SHAPE,
                                        interpolate=interp).volume
    assert new.dtype == old.dtype, (new.dtype, old.dtype)
    assert np.array_equal(new, old), f"{int(np.sum(new != old))} voxels differ from old behaviour"

    result = interpolate_labels_separately(sparse_keyframes_by_label(canvas), SHAPE,
                                           interpolate=interp)
    assert guide_export_warnings(result, {}) == [], guide_export_warnings(result, {})
    print(f"   ok ({int(old.sum())} voxels, identical)")


def selftest_sidecars(interp, tmp_dir):
    print("7. sidecars: .regions.json + the .annotated_slices.json convention")
    canvas = _canvas()
    _box(canvas, [0, 4, 8], 1, 2, 10, 2, 10)
    _box(canvas, [2, 6], 2, 20, 30, 4, 12)
    result = interpolate_labels_separately(sparse_keyframes_by_label(canvas), SHAPE,
                                           interpolate=interp)

    region_labels = {1: "cortex", 2: "cerebellar hemisphere"}
    out_path = tmp_dir / "s12t_guide_sample.nii.gz"
    regions_path, slices_path = write_guide_sidecars(
        out_path, "/data/s12t/registration.tif", result, region_labels, "sample",
        SHAPE[0], spacing_xyz=(1.0, 1.0, 1.0))

    assert regions_path.name == "s12t_guide_sample.regions.json", regions_path
    assert slices_path.name == "s12t_guide_sample.annotated_slices.json", slices_path

    regions = json.loads(regions_path.read_text())
    assert regions["regions"] == {"1": "cortex", "2": "cerebellar hemisphere"}, regions["regions"]
    assert regions["annotated_slices"] == {"1": [0, 4, 8], "2": [2, 6]}, regions["annotated_slices"]
    assert regions["image_path"] == "/data/s12t/registration.tif"
    assert "voxel size" in regions["voxel_size_um_note"].lower()

    # Same key registration_eval.load_region_annotation_hint() reads. Asserted
    # by name rather than by importing it: registration_eval pulls in
    # registration_ants.transforms -> antspyx, which gt_sam does not have.
    hint = json.loads(slices_path.read_text())
    assert hint["hand_drawn_slices"] == [0, 2, 4, 6, 8], hint

    # .nii (not .gz) and .tif outputs must hang their sidecars off the same stem.
    for name, stem in [("guide.nii", "guide"), ("guide.tif", "guide"), ("guide", "guide")]:
        assert _output_stem(tmp_dir / name).name == stem, name
    print("   ok")


def selftest_config_normalizers():
    print("8. config: region_labels int/str keys, display_scale_zyx validation")
    assert _normalize_region_labels({1: "cortex", "2": "cerebellar hemisphere"}) == \
        {1: "cortex", 2: "cerebellar hemisphere"}
    assert _normalize_region_labels(None) == {}
    assert _normalize_region_labels({}) == {}

    def rejects(fn, value, needle):
        try:
            fn(value)
        except ValueError as exc:
            assert needle in str(exc), f"{value!r}: wrong reason {exc}"
        else:
            raise AssertionError(f"{value!r} should have been rejected")

    rejects(_normalize_region_labels, {"cortex": 1}, "整数")
    rejects(_normalize_region_labels, {0: "background"}, ">= 1")
    rejects(_normalize_region_labels, {1: "cortex", "1": "cortex"}, "两次")
    rejects(_normalize_region_labels, ["cortex"], "映射")

    assert _normalize_display_scale([32.0, 2.6, 2.6]) == [32.0, 2.6, 2.6]
    assert _normalize_display_scale(None) is None
    rejects(_normalize_display_scale, [2.6, 2.6], "3 个")
    rejects(_normalize_display_scale, [32.0, 0.0, 2.6], "正数")
    rejects(_normalize_display_scale, "32,2.6,2.6", "三个数")

    # uint8 export: a brush value the output can't hold must fail loudly.
    try:
        interpolate_labels_separately({300: {0: np.ones((4, 4), bool)}}, (2, 4, 4),
                                      interpolate=_reference_interpolate_sparse_mask)
    except ValueError as exc:
        assert "uint8" in str(exc), exc
    else:
        raise AssertionError("label 300 should have been rejected")
    print("   ok")


def selftest_interpolator_matches_registration_ants():
    print("9. local reference interpolator == registration_ants.mask_utils (when available)")
    try:
        real = _interpolate_sparse_mask()
    except ImportError:
        print("   skipped (no registration_ants in this env)")
        return
    rng = np.random.default_rng(7)
    keyframes = {z: (rng.random((20, 20)) > 0.7) for z in (0, 3, 9)}
    a = real(keyframes, (12, 20, 20))
    b = _reference_interpolate_sparse_mask(keyframes, (12, 20, 20))
    assert np.array_equal(a, b), f"{int(np.sum(a != b))} voxels differ -- the copy has drifted"
    print("   ok")


def run_selftests():
    import tempfile

    print("=== paint_mask.py selftests (synthetic data only, no GUI) ===")
    interp = _selftest_interpolator()
    selftest_three_labels_stay_separate(interp)
    selftest_per_label_beats_merged_interpolation(interp)
    selftest_single_plane_label_warns(interp)
    selftest_overlap_is_counted_and_reported(interp)
    selftest_unnamed_and_unpainted_labels_warn(interp)
    selftest_single_label_matches_old_behaviour(interp)
    with tempfile.TemporaryDirectory() as tmp:
        selftest_sidecars(interp, Path(tmp))
    selftest_config_normalizers()
    selftest_interpolator_matches_registration_ants()
    print("=== all selftests passed ===")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Paint a binary mask or a paired guide outline")
    _local_config.add_config_arg(parser, "paint_mask")
    parser.add_argument("--selftest", action="store_true",
                        help="run the built-in synthetic tests (no GUI, no config) and exit")
    args_cli = parser.parse_args()

    if args_cli.selftest:
        return run_selftests()

    cfg = _load_local_config(args_cli.config)
    if cfg.kind == "mask":
        args = SimpleNamespace(sample_path=cfg.image_path, output_path=cfg.output_path,
                                existing_mask=cfg.existing_mask_path)
        _run_mask(args)
    elif cfg.kind == "guide":
        args = SimpleNamespace(image_path=cfg.image_path, output_path=cfg.output_path,
                                existing_mask=cfg.existing_mask_path, role=cfg.role,
                                region_labels=cfg.region_labels,
                                display_scale_zyx=cfg.display_scale_zyx)
        _run_guide(args)
    else:
        raise ValueError(f"Unknown kind: {cfg.kind!r} (expected 'mask' or 'guide')")

    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
