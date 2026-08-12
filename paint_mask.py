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
    region up. That is what makes the label->region mapping below
    load-bearing rather than cosmetic -- it is the only thing tying "the blob
    I painted" to "which atlas structure to pair it with".
    role="atlas" is still supported for the cases where the automatic
    atlas-side region needs to be hand-corrected (set
    EXISTING_MASK_PATH to ../Registration_ants/scripts/project_outline.py's
    output, or to the auto-generated region, as a starting guess instead of
    a blank canvas).

    Picking the region: configure an atlas (`atlas_preset`, reusing
    Registration_ants' own preset file) and the viewer grows an ontology
    tree. Selecting any node highlights that structure AND all its
    descendants in an atlas layer -- the only way a high-level node shows
    anything, since the annotation's own labels sit at ontology depths 2-12
    and a depth-3 node owns no voxels under its own id. Assign the selected
    region to a brush label right there; the export records the ontology IDS
    (see write_guide_sidecars for why ids rather than names). The atlas is a
    reference only, never registered to the sample -- what is exported is
    ids plus voxel indices on the sample's own grid, so nothing about how
    the atlas is displayed, oriented, or downsampled can reach the output.

    Several regions at once: the paint layer is a napari Labels layer, so
    label 1/2/3/... are different brush values, one per brain region (see
    `region_labels` in configs/paint_mask.example.yaml). One label can carry
    SEVERAL atlas regions -- DevCCF has no single "cortex" structure, only 36
    separate `layer N of <area>` ones. They are exported as ONE multi-label
    volume plus a `.regions.json` sidecar naming each label, and each label
    is interpolated on its own -- see interpolate_labels_separately for why
    they must not be interpolated together.

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
import yaml

import _local_config  # sibling module

# napari/PyQt5 are imported lazily by _import_gui() rather than here, and
# mask_utils by _interpolate_sparse_mask(), so that --selftest (pure numpy +
# scipy, no window) runs in the gt_sam env too, which has no
# ../Registration_ants editable install. Both are hard requirements for the
# actual painting GUI, which only ever runs in antsreg.
napari = QLabel = QPushButton = QVBoxLayout = QWidget = None
QCheckBox = QHBoxLayout = QLineEdit = QSpinBox = QTreeWidget = QTreeWidgetItem = Qt = None


def _import_gui():
    """Bind the napari/Qt names used by the viewer code. Called once at the
    top of the GUI entry points; import errors surface there rather than at
    module import, which is what keeps --selftest env-independent."""
    global napari, QLabel, QPushButton, QVBoxLayout, QWidget
    global QCheckBox, QHBoxLayout, QLineEdit, QSpinBox, QTreeWidget, QTreeWidgetItem, Qt
    import napari as _napari
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtWidgets import (QCheckBox as _QCheckBox, QHBoxLayout as _QHBoxLayout,
                                 QLabel as _QLabel, QLineEdit as _QLineEdit,
                                 QPushButton as _QPushButton, QSpinBox as _QSpinBox,
                                 QTreeWidget as _QTreeWidget, QTreeWidgetItem as _QTreeWidgetItem,
                                 QVBoxLayout as _QVBoxLayout, QWidget as _QWidget)
    napari, QLabel, QPushButton = _napari, _QLabel, _QPushButton
    QVBoxLayout, QWidget = _QVBoxLayout, _QWidget
    QCheckBox, QHBoxLayout, QLineEdit, QSpinBox = _QCheckBox, _QHBoxLayout, _QLineEdit, _QSpinBox
    QTreeWidget, QTreeWidgetItem, Qt = _QTreeWidget, _QTreeWidgetItem, _Qt


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
        region_ids=_normalize_region_ids(cfg.get("region_ids") or {}),
        display_scale_zyx=_normalize_display_scale(cfg.get("display_scale_zyx")),
        atlas=_atlas_reference_config(cfg),
    )


# Path to Registration_ants' atlas preset file, so the atlas/ontology this
# tool shows is literally the same one the pipeline registers against --
# there is no second place to keep those four paths in sync.
_DEFAULT_ATLAS_PRESETS = (
    Path(__file__).resolve().parent.parent / "Registration_ants" / "configs" / "atlas_presets_local.yaml")

_ATLAS_KEYS = ("template_path", "annotation_path", "ontology_path", "resolution_um", "orientation")


def _atlas_reference_config(cfg):
    """Resolve the optional atlas reference (guide mode only) -> SimpleNamespace
    or None when no atlas is configured.

    `atlas_preset` names an entry in Registration_ants' atlas_presets file
    (devccf_p04 / demba_p5 / ...); any `atlas_*` key written directly in this
    config overrides that entry's same-named field, so a one-off atlas needs
    no preset at all. Only the annotation + ontology are strictly needed (they
    are what the region picker resolves against); the template is just the
    grayscale backdrop and may be omitted.
    """
    preset_name = cfg.get("atlas_preset")
    overrides = {key: cfg.get(f"atlas_{key}") for key in _ATLAS_KEYS}
    overrides["ontology_path"] = cfg.get("ontology_path", overrides["ontology_path"])
    if not preset_name and not any(v for v in overrides.values()):
        return None

    resolved = {}
    if preset_name:
        presets_path = Path(cfg.get("atlas_presets_path") or _DEFAULT_ATLAS_PRESETS)
        if not presets_path.exists():
            raise FileNotFoundError(
                f"atlas_preset: {preset_name} 需要预设文件 {presets_path}，但它不存在。\n"
                f"要么改 atlas_presets_path 指向 Registration_ants/configs/atlas_presets_local.yaml，\n"
                f"要么删掉 atlas_preset、直接在本配置里写 atlas_annotation_path / ontology_path。")
        with open(presets_path, encoding="utf-8") as f:
            presets = yaml.safe_load(f) or {}
        if preset_name not in presets:
            raise ValueError(
                f"{presets_path} 里没有预设 {preset_name!r}；有的是: {sorted(presets)}")
        resolved = {key: presets[preset_name].get(key) for key in _ATLAS_KEYS}

    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value

    for required in ("annotation_path", "ontology_path"):
        if not resolved.get(required):
            raise ValueError(
                f"图谱参考需要 {required}（preset {preset_name!r} 里没有，配置里也没写 atlas_{required}）")
        if not Path(resolved[required]).exists():
            raise FileNotFoundError(f"图谱 {required} 不存在: {resolved[required]}")
    if resolved.get("template_path") and not Path(resolved["template_path"]).exists():
        raise FileNotFoundError(f"图谱 template_path 不存在: {resolved['template_path']}")

    downsample = int(cfg.get("atlas_downsample") or 1)
    if downsample < 1:
        raise ValueError(f"atlas_downsample 必须 >= 1，得到 {downsample}")

    return SimpleNamespace(
        preset=preset_name,
        template_path=resolved.get("template_path") or None,
        annotation_path=resolved["annotation_path"],
        ontology_path=resolved["ontology_path"],
        resolution_um=float(resolved.get("resolution_um") or 0) or None,
        orientation=resolved.get("orientation") or None,
        downsample=downsample,
    )


def _normalize_label_map(raw, key_name, coerce, describe):
    """{brush label -> [entry, ...]}, label keys forced to int, values always
    a list.

    A label maps to a LIST, not a single entry, because one guide region
    routinely needs several ontology entries: DevCCF has no single "cortex"
    structure, only 36 separate `layer N of <area>` ones, and the pipeline's
    mask.guide_regions.atlas_names/atlas_ids union a list per label for
    exactly that reason. A bare scalar is accepted as a one-element list --
    `1: cortex` is the obvious spelling and reads identically to `1: [cortex]`.

    YAML gives `1: cortex` as an int key but `"1": cortex` as a string one,
    and both spellings look identical in the file, so everything downstream
    would silently miss half the mapping if this didn't normalize.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{key_name} 应该是 {{label: {describe}}} 映射，实际是 {type(raw).__name__}")

    normalized = {}
    for key, entries in raw.items():
        try:
            label = int(key)
        except (TypeError, ValueError):
            raise ValueError(
                f"{key_name} 的 key 必须是画笔 label（整数），得到 {key!r}") from None
        if label < 1:
            raise ValueError(f"{key_name} 的 label 必须 >= 1（0 是背景/橡皮），得到 {label}")
        if label in normalized:
            raise ValueError(f"{key_name} 里 label {label} 出现了两次（int 和 str key 各一次？）")
        if isinstance(entries, (str, int)):
            entries = [entries]
        try:
            entries = [coerce(e) for e in entries]
        except (TypeError, ValueError):
            raise ValueError(f"{key_name} 里 label {label} 的值应该是 {describe}，得到 {entries!r}") from None
        if not entries:
            raise ValueError(f"{key_name} 里 label {label} 的{describe}是空的")
        normalized[label] = entries
    return normalized


def _region_name(value):
    name = str(value).strip()
    if not name:
        raise ValueError(value)
    return name


def _normalize_region_labels(raw):
    """{brush label -> [brain region name, ...]}. Whether a name actually
    resolves in the atlas ontology is checked by the ontology picker when an
    atlas is configured, and on the Registration_ants side otherwise."""
    return _normalize_label_map(raw, "region_labels", _region_name, "脑区名")


def _normalize_region_ids(raw):
    """{brush label -> [ontology structure id, ...]}. Normally written by the
    GUI's ontology picker rather than by hand; ids beat names downstream
    because they are matched exactly instead of as substrings."""
    return _normalize_label_map(raw, "region_ids", int, "ontology structure id（整数）")


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


def _read_array(path):
    """Just the voxels, with the SimpleITK image dropped immediately.

    GetArrayFromImage copies, so the ITK buffer and the numpy array are both
    live until the image goes out of scope -- and binding it to a throwaway
    `_` keeps it alive for the rest of the enclosing function. That is 1.1 GB
    of nothing for the DevCCF annotation, doubling peak memory during the
    atlas load for a handle the atlas path never uses (unlike the sample
    path, which needs the image for CopyInformation on export).
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


# =====================================================================================
# atlas reference: a read-only backdrop + the ontology the region picker resolves against
# =====================================================================================
def _atlas_helpers():
    """registration_ants.atlas_utils -- the ontology half of it is deliberately
    free of the antspyx import (see that module's docstring), which is what
    lets this display-only tool reuse it."""
    from registration_ants import atlas_utils
    return atlas_utils


def _reorient_zyx(arr_zyx, orientation, atlas_utils):
    """Apply a ClearMap-style orientation spec to a (z,y,x) array.

    atlas_utils.reorient_volume operates on (x,y,z)-ordered arrays -- the
    order ants.image_read().numpy() produces -- while everything in this file
    is (z,y,x), the order sitk.GetArrayFromImage produces (see module
    docstring fact 2). So the array is flipped into (x,y,z), reoriented, and
    flipped back. Getting this wrong would silently show the atlas permuted
    relative to the pipeline's view of it.

    Purely cosmetic here: the picker exports ontology IDS, which say nothing
    about voxel layout, so no orientation mistake in this display can reach
    the exported file. It matters only so the reference LOOKS like the atlas
    the pipeline uses.

    Returns a VIEW, not a copy -- transposes and flips are both views, and the
    caller downsamples before materializing anything. Copying here instead
    would double peak memory on a volume that is 1.1 GB to begin with.
    """
    if not orientation:
        return arr_zyx
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    arr_xyz = atlas_utils.reorient_volume(arr_xyz, orientation)
    return np.transpose(arr_xyz, (2, 1, 0))


def _compact_annotation(annotation_zyx, chunk=64):
    """(compact, present_ids) -- annotation relabelled to small consecutive
    indices into `present_ids`, so highlight lookups run on a uint8/uint16
    array instead of the raw one.

    The DevCCF P04 annotation reads back as float32 at 800x560x640: 1.1 GB
    resident, on top of a sample volume that is already several GB. It only
    carries 193 distinct labels, so an index array costs 287 MB (uint8) and
    makes np.isin cheap besides.

    Everything here runs a slab at a time, including the label scan, because
    every obvious one-shot spelling materializes another array the size of
    the input or worse: np.unique sorts a full flattened copy, and
    np.unique(return_inverse=True) / np.searchsorted over the whole array both
    return int64 -- 2.3 GB for this atlas, worse than the problem being
    solved. `annotation_zyx` is also typically a non-contiguous reoriented
    VIEW (see _reorient_zyx), so a whole-array operation would quietly
    materialize that too.
    """
    present = set()
    for z0 in range(0, annotation_zyx.shape[0], chunk):
        present.update(np.unique(annotation_zyx[z0:z0 + chunk]).tolist())
    present_ids = np.array(sorted(present))

    dtype = np.uint8 if len(present_ids) <= np.iinfo(np.uint8).max + 1 else np.uint16
    compact = np.empty(annotation_zyx.shape, dtype=dtype)
    for z0 in range(0, annotation_zyx.shape[0], chunk):
        sl = slice(z0, z0 + chunk)
        compact[sl] = np.searchsorted(present_ids, annotation_zyx[sl]).astype(dtype)
    return compact, present_ids.astype(np.int64)


def voxels_per_ontology_node(structures, present_ids, counts):
    """{structure id: voxel count including descendants} for EVERY ontology
    node, from the per-label counts of the annotation.

    Every voxel is credited to its own label and to each of that label's
    ancestors (structure_id_path is the full root->node chain), so a node at
    any depth reports the size of the whole subtree under it -- which is
    exactly what the picker highlights when you click that node.

    Nodes absent from the result have no voxels in this annotation at all.
    That is the common case, not an edge case: the DevCCF P04 annotation
    carries 193 of the ontology's 2552 structures. Assigning one of the empty
    ones would make the pipeline refuse the whole run
    (_build_guide_regions_from_labels raises when a region matches nothing),
    so the picker greys them out instead of letting you pick them.
    """
    totals = {}
    for label, count in zip(present_ids, counts):
        info = structures.get(int(label))
        if info is None:            # background (0), or a label this ontology doesn't describe
            continue
        for ancestor in info["structure_id_path"]:
            totals[ancestor] = totals.get(ancestor, 0) + int(count)
    return totals


def _load_atlas_reference(atlas_cfg):
    """Load the atlas backdrop + ontology for the region picker.

    Returns SimpleNamespace(template, compact, present_ids, index_of_id,
    structures, node_voxels, downsample) -- or raises with a message pointing
    at the offending config key. `template` is None when no template_path is
    configured; the picker works without it (highlighting only needs the
    annotation), you just get no grayscale underneath.
    """
    atlas_utils = _atlas_helpers()
    structures = atlas_utils.load_ccf_ontology_json(atlas_cfg.ontology_path)

    step = atlas_cfg.downsample
    sub = (slice(None, None, step),) * 3

    print(f"[atlas] annotation: {atlas_cfg.annotation_path}")
    annotation = _read_array(atlas_cfg.annotation_path)
    # Reorient and downsample as views, so the only full-size array alive is
    # the one SimpleITK just read; _compact_annotation reads through the view
    # a slab at a time and the 1.1 GB original is dropped immediately after.
    compact, present_ids = _compact_annotation(
        _reorient_zyx(annotation, atlas_cfg.orientation, atlas_utils)[sub])
    del annotation

    counts = np.bincount(compact.ravel(), minlength=len(present_ids))
    node_voxels = voxels_per_ontology_node(structures, present_ids, counts)

    template = None
    if atlas_cfg.template_path:
        print(f"[atlas] template:   {atlas_cfg.template_path}")
        raw_template = _read_array(atlas_cfg.template_path)
        template = np.ascontiguousarray(
            _reorient_zyx(raw_template, atlas_cfg.orientation, atlas_utils)[sub])
        del raw_template
        if template.shape != compact.shape:
            print(f"WARNING: atlas template shape {template.shape} != annotation shape "
                  f"{compact.shape}; showing the annotation only.")
            template = None

    known = sum(1 for sid in present_ids if int(sid) in structures)
    print(f"[atlas] ontology: {len(structures)} structures, {len(present_ids)} labels in the "
          f"annotation ({known} of them in the ontology), {len(node_voxels)} nodes non-empty; "
          f"grid {compact.shape}" + (f", downsampled {step}x" if step > 1 else ""))

    return SimpleNamespace(
        template=template,
        compact=compact,
        present_ids=present_ids,
        index_of_id={int(sid): i for i, sid in enumerate(present_ids)},
        structures=structures,
        node_voxels=node_voxels,
        downsample=step,
    )


def highlight_mask(atlas, structure_id):
    """Binary (z,y,x) mask of one ontology node INCLUDING all its descendants.

    Descendants are resolved through structure_id_path, never by name, so
    picking a node at any depth lights up the whole subtree -- which is the
    only way a depth-3 node can highlight anything at all here, since the
    DevCCF annotation's own labels sit at depths 2 through 12 and a parent
    node usually owns no voxels under its own id.

    Resolved through atlas_utils.descendant_ids_of, the same function the
    pipeline's atlas_ids path uses, so what lights up here is exactly what
    registration will pair against.
    """
    wanted = _atlas_helpers().descendant_ids_of(atlas.structures, [structure_id])
    indices = [atlas.index_of_id[sid] for sid in wanted if sid in atlas.index_of_id]
    if not indices:
        return np.zeros(atlas.compact.shape, dtype=np.uint8)
    return np.isin(atlas.compact, indices).astype(np.uint8)


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


def _tree_label(sid, info, voxels):
    acronym = info.get("acronym")
    name = f"{info['name']} ({acronym})" if acronym else info["name"]
    return [name, f"{voxels:,}" if voxels else "—", str(sid)]


def populate_ontology_tree(tree, structures, node_voxels):
    """Fill a QTreeWidget with the whole ontology; returns {sid: item}.

    Built shallowest-first so every node's parent (structure_id_path[-2])
    already exists when the node is added -- structure_id_path is the full
    root->node chain, so the tree shape comes straight out of it with no
    separate child lists to walk.

    Nodes with no voxels in this annotation are marked disabled rather than
    dropped: an empty node's whole subtree is necessarily empty too (a voxel
    is credited to every ancestor of its label, so a non-empty child forces a
    non-empty parent), which makes disabling safe -- it can never hide a
    pickable descendant -- and keeps the real tree shape visible instead of
    silently rewriting the ontology.
    """
    items = {}
    for sid, info in sorted(structures.items(), key=lambda kv: len(kv[1]["structure_id_path"])):
        path = info["structure_id_path"]
        voxels = node_voxels.get(sid, 0)
        item = QTreeWidgetItem(_tree_label(sid, info, voxels))
        item.setData(0, Qt.UserRole, sid)
        if not voxels:
            item.setDisabled(True)
        parent = items.get(path[-2]) if len(path) >= 2 else None
        (parent or tree.invisibleRootItem()).addChild(item)
        items[sid] = item
    return items


def visible_tree_ids(structures, node_voxels, text, hide_empty):
    """Which ontology ids a search box showing `text` should leave visible.

    Every ancestor of a match is included, otherwise a deep hit would be
    unreachable -- the tree can only show a node if the whole chain down to
    it is showing. Matching is on name and acronym, since the acronym
    (DevCCF's "SPall", "THyA", ...) is often what you actually remember.
    """
    text = text.strip().lower()
    visible = set()
    for sid, info in structures.items():
        if hide_empty and not node_voxels.get(sid):
            continue
        if text and text not in info["name"].lower() and text not in (info.get("acronym") or "").lower():
            continue
        visible.update(info["structure_id_path"])
    return visible


def format_assignment(assignment, structures):
    """The label -> region panel text. Also the thing you check before
    exporting, so it spells out ids as well as names."""
    if not assignment:
        return "还没有分配任何脑区。\n在上面的树里选一个脑区，设好 label 号，点“加到 label”。"
    lines = []
    for label in sorted(assignment):
        entries = assignment[label]
        names = ", ".join(f"{structures[sid]['name']} [{sid}]" for sid in entries)
        lines.append(f"  label {label} = {names}")
    return "brush label → 脑区：\n" + "\n".join(lines)


def guide_regions_yaml_snippet(region_ids, region_names, output_path, voxel_size_um=None):
    """A ready-to-paste mask.guide_regions block for the pipeline config.

    Emitted on export because the ids are the whole point of picking regions
    in the GUI and retyping them by hand from a JSON sidecar is exactly where
    they would get corrupted. atlas_names is emitted alongside as a comment
    only: two sources of truth that can disagree is precisely the failure
    ids were chosen to remove, so the pipeline reads the ids and the names
    stay human-facing.
    """
    voxel = list(voxel_size_um) if voxel_size_um else ["?", "?", "?"]
    lines = [
        "mask:",
        "  guide_regions:",
        f"    regions_mask: {output_path}",
        f"    voxel_size_um: [{voxel[0]}, {voxel[1]}, {voxel[2]}]   # 原图 (x,y,z) µm -- tif header 里没有，必须手填",
        "    atlas_ids:",
    ]
    for label in sorted(region_ids):
        names = ", ".join(region_names.get(label, []))
        lines.append(f"      {label}: {list(region_ids[label])}" + (f"   # {names}" if names else ""))
    lines.append("    weight: 1.0")
    return "\n".join(lines)


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
    """One label's region name(s) as display text -- a label can carry several
    (see _normalize_label_map), so this joins rather than indexes."""
    names = region_labels.get(label)
    return ", ".join(names) if names else "unnamed"


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
            f"region_labels lists label {label} ({_label_name(label, region_labels)}) but nothing was "
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
                         spacing_xyz=None, region_ids=None, atlas_info=None):
    """Write the two sidecars next to the exported outline, and return their paths.

    <stem>.regions.json is the one that matters for this tool: it is the
    only record of which brush label is which brain region, and
    Registration_ants needs exactly that to pull the matching region out of
    the atlas annotation volume.

    It records both `region_ids` (ontology structure ids, when the region was
    picked in the GUI's ontology tree) and `regions` (the names, always).
    Feed the IDS to mask.guide_regions.atlas_ids: they are matched exactly,
    descendants included, so the region registration pairs against is the one
    that was highlighted while painting. The names are for reading -- as
    mask.guide_regions.atlas_names they would be matched as case-insensitive
    substrings, which can pull in unrelated structures ("Cerebellum" also
    matches "cerebellum related fiber tracts").

    Both are {label: LIST}, because one guide region often needs several
    ontology entries -- DevCCF has no single "cortex", only 36 `layer N of
    <area>` structures.

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

    region_ids = region_ids or {}
    all_planes = sorted({z for planes in result.slices_by_label.values() for z in planes})
    regions = {str(lab): list(region_labels[lab]) for lab in sorted(region_labels)}
    regions_path.write_text(json.dumps({
        "regions": regions,
        "region_ids": {str(lab): list(region_ids[lab]) for lab in sorted(region_ids)},
        "annotated_slices": {str(lab): planes
                             for lab, planes in sorted(result.slices_by_label.items())},
        "image_path": str(image_path),
        "mask_path": str(output_path),
        "role": role,
        "total_z": int(total_z),
        "header_spacing_xyz": list(spacing_xyz) if spacing_xyz is not None else None,
        "voxel_size_um_note": VOXEL_SIZE_UM_NOTE,
        "atlas": atlas_info,
    }, indent=2, ensure_ascii=False))
    slices_path.write_text(json.dumps({
        "hand_drawn_slices": all_planes,
        "total_z": int(total_z),
        "regions": regions,
    }, indent=2, ensure_ascii=False))
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
    brush number you're about to paint with is never a guess. Only used when
    there is no atlas configured -- with one, the assignment panel built by
    _add_ontology_picker replaces this and is editable."""
    if not region_labels:
        return ("No region_labels in the config: paint one region with label 1.\n"
                "For several regions, add region_labels to the config, or configure\n"
                "an atlas (atlas_preset) to pick them from the ontology tree here --\n"
                "an unnamed label cannot be paired with an atlas region.\n")
    lines = "\n".join(f"  label {lab} = {_label_name(lab, region_labels)}"
                      for lab in sorted(region_labels))
    return f"Brush label -> brain region:\n{lines}\n"


def _seed_assignment(region_labels, region_ids, structures):
    """Pre-fill the GUI assignment from the config.

    region_ids is taken as-is. Names from region_labels are resolved to ids
    only on an exact, unique name match -- the substring matching the
    pipeline does for names is precisely what ids are here to avoid, so
    guessing on the operator's behalf would reintroduce it. A name that
    doesn't resolve is reported, not silently dropped, and stays usable as a
    name-only entry.
    """
    by_name = {}
    for sid, info in structures.items():
        by_name.setdefault(info["name"].strip().lower(), []).append(sid)

    assignment, unresolved = {}, []
    for label in sorted(set(region_labels) | set(region_ids)):
        ids = list(region_ids.get(label, []))
        for name in region_labels.get(label, []):
            matches = by_name.get(name.strip().lower(), [])
            if len(matches) == 1 and matches[0] not in ids:
                ids.append(matches[0])
            elif len(matches) != 1:
                unresolved.append((label, name, len(matches)))
        if ids:
            assignment[label] = ids
    return assignment, unresolved


def _add_ontology_picker(viewer, atlas, paint_layer, assignment, atlas_scale=None):
    """The ontology tree + label-assignment panel, as a dock on the right.

    Selecting any node highlights that structure and everything under it in
    a dedicated atlas layer (see highlight_mask), which is the only way to
    see a high-level region at all: the annotation's own labels sit at
    ontology depths 2-12, so a depth-3 node owns no voxels under its own id.

    `assignment` ({label: [structure id]}) is mutated in place -- it is the
    live state the export reads, so there is no separate "apply" step to
    forget.
    """
    scale_kwargs = {"scale": atlas_scale} if atlas_scale is not None else {}
    highlight = viewer.add_labels(
        np.zeros(atlas.compact.shape, dtype=np.uint8), name="atlas region (highlight)",
        opacity=0.55, **scale_kwargs)
    highlight.editable = False                 # a reference layer; painting here would be meaningless

    # objectNames so these are addressable from outside the closure -- napari
    # contributes its own QSpinBox/QLineEdit widgets to the same window, so
    # "the first spin box" is not this panel's brush-label box.
    search = QLineEdit()
    search.setObjectName("ontology_search")
    search.setPlaceholderText("按名字/缩写过滤脑区...")
    hide_empty = QCheckBox("只显示这份 annotation 里有体素的脑区")
    hide_empty.setObjectName("ontology_hide_empty")
    hide_empty.setChecked(True)

    tree = QTreeWidget()
    tree.setObjectName("ontology_tree")
    tree.setHeaderLabels(["脑区", "体素", "id"])
    tree.setColumnWidth(0, 260)
    tree.setMinimumHeight(320)
    items = populate_ontology_tree(tree, atlas.structures, atlas.node_voxels)

    label_spin = QSpinBox()
    label_spin.setObjectName("ontology_brush_label")
    label_spin.setRange(1, MAX_LABEL)
    add_btn = QPushButton("加到 label")
    add_btn.setObjectName("ontology_assign")
    remove_btn = QPushButton("从 label 移除")
    remove_btn.setObjectName("ontology_unassign")
    jump_btn = QPushButton("跳到该脑区中心层")
    jump_btn.setObjectName("ontology_jump")
    assign_label = QLabel()
    picker_status = QLabel()
    picker_status.setWordWrap(True)

    def selected_id():
        item = tree.currentItem()
        return None if item is None else item.data(0, Qt.UserRole)

    def refresh_filter():
        visible = visible_tree_ids(atlas.structures, atlas.node_voxels,
                                   search.text(), hide_empty.isChecked())
        for sid, item in items.items():
            item.setHidden(sid not in visible)
        if search.text().strip():
            tree.expandAll()

    def on_select():
        sid = selected_id()
        if sid is None:
            return
        info = atlas.structures[sid]
        voxels = atlas.node_voxels.get(sid, 0)
        highlight.data = highlight_mask(atlas, sid)
        highlight.name = f"atlas: {info['name']}"
        if voxels:
            picker_status.setText(f"{info['name']} [{sid}]，含后代 {voxels:,} 体素。")
        else:
            picker_status.setText(
                f"{info['name']} [{sid}] 在这份 annotation 里没有任何体素，不能分配 —— "
                f"流水线遇到匹配不到的区域会直接报错。")

    def on_jump():
        sid = selected_id()
        if sid is None or not atlas.node_voxels.get(sid):
            return
        planes = np.flatnonzero(highlight.data.any(axis=(1, 2)))
        if not planes.size:
            return
        # dims.set_point takes WORLD coordinates, so the plane index has to go
        # through the layer's own scale -- the atlas and the sample sit on
        # different grids sharing one slider, and skipping this would jump to
        # the atlas's index interpreted as the sample's microns.
        centre = int(planes[len(planes) // 2])
        viewer.dims.set_point(0, centre * float(highlight.scale[0]))

    def on_add():
        sid = selected_id()
        if sid is None:
            picker_status.setText("先在树里选一个脑区。")
            return
        if not atlas.node_voxels.get(sid):
            picker_status.setText(
                f"{atlas.structures[sid]['name']} 在这份 annotation 里没有体素，拒绝分配。")
            return
        label = label_spin.value()
        entries = assignment.setdefault(label, [])
        if sid not in entries:
            entries.append(sid)
        paint_layer.selected_label = label     # so the brush is already right for painting
        viewer.layers.selection = {paint_layer}
        refresh_assignment()

    def on_remove():
        sid = selected_id()
        label = label_spin.value()
        if sid is not None and sid in assignment.get(label, []):
            assignment[label].remove(sid)
            if not assignment[label]:
                del assignment[label]
        refresh_assignment()

    def refresh_assignment():
        assign_label.setText(format_assignment(assignment, atlas.structures))

    search.textChanged.connect(lambda _t: refresh_filter())
    hide_empty.toggled.connect(lambda _c: refresh_filter())
    tree.currentItemChanged.connect(lambda _cur, _prev: on_select())
    jump_btn.clicked.connect(on_jump)
    add_btn.clicked.connect(on_add)
    remove_btn.clicked.connect(on_remove)

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(QLabel("图谱 ontology（选中即高亮，含全部后代）"))
    layout.addWidget(search)
    layout.addWidget(hide_empty)
    layout.addWidget(tree)
    layout.addWidget(picker_status)
    layout.addWidget(jump_btn)
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.addWidget(QLabel("brush label"))
    row_layout.addWidget(label_spin)
    row_layout.addWidget(add_btn)
    row_layout.addWidget(remove_btn)
    layout.addWidget(row)
    layout.addWidget(assign_label)
    viewer.window.add_dock_widget(dock, area="right", name="Atlas / Ontology")

    refresh_filter()
    refresh_assignment()
    return SimpleNamespace(highlight=highlight, refresh_assignment=refresh_assignment)


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

    # The atlas is a read-only reference on its own grid, deliberately NOT
    # registered to the sample -- the whole point of painting a guide is that
    # the two don't line up yet. Both get an honest physical scale so each is
    # drawn undistorted; they share one z slider, so scrolling to a highlighted
    # atlas region also moves the sample view (use "跳到该脑区中心层" to get
    # there, then scroll back). Nothing about this display reaches the export,
    # which is ontology ids plus voxel indices on the sample's own grid.
    atlas = assignment = None
    if args.atlas:
        atlas = _load_atlas_reference(args.atlas)
        atlas_scale = ([args.atlas.resolution_um * atlas.downsample] * 3
                       if args.atlas.resolution_um else None)
        scale_kwargs = {"scale": atlas_scale} if atlas_scale is not None else {}
        if atlas.template is not None:
            viewer.add_image(atlas.template, name="atlas template", colormap="gray",
                             visible=False, **scale_kwargs)
        if atlas_scale is not None and args.display_scale_zyx is None:
            # Both sets of layers share one z slider, positioned in world
            # coordinates. With the atlas in microns and the sample left at
            # scale 1 (voxel indices), the sample collapses into the first
            # few percent of the slider's range and "jump to region" lands
            # nowhere near it -- so this is worth saying out loud rather than
            # leaving as a puzzling viewer.
            print(f"WARNING: 配了图谱但没设 display_scale_zyx。图谱按 {atlas_scale[0]:g} µm 定位，"
                  f"样本却按体素索引(1,1,1)，两者共用的 z 滑块会对不上，样本会被挤在滑块最前面一小段。"
                  f"\n         在配置里加上 display_scale_zyx: [z, y, x]（原图微米数，例如 "
                  f"[32.0, 2.6, 2.6]）。只影响显示，不影响导出。")

        assignment, unresolved = _seed_assignment(args.region_labels, args.region_ids,
                                                  atlas.structures)
        for label, name, n in unresolved:
            print(f"WARNING: region_labels label {label} 的 {name!r} 在 ontology 里"
                  f"{'匹配到多个' if n else '匹配不到'}结构（{n} 个），没有转成 id；"
                  f"请在树里重新选一次。")
        _add_ontology_picker(viewer, atlas, paint_layer, assignment, atlas_scale=atlas_scale)

    guess_note = "Pre-filled with the existing mask -- adjust/redraw as needed.\n" if args.existing_mask else ""
    header = ("在右侧 ontology 树里选脑区 → 设 brush label → “加到 label”，然后就用那个\n"
              "画笔号在样本上画。\n" if atlas else _region_legend(args.region_labels))
    status_label = QLabel(
        header +
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

        # The GUI assignment is authoritative when an atlas is loaded (it is
        # what was actually looked at); the config's region_labels are the
        # fallback for the no-atlas case.
        if atlas is not None:
            region_ids = {lab: list(ids) for lab, ids in assignment.items() if ids}
            region_labels = {lab: [atlas.structures[sid]["name"] for sid in ids]
                             for lab, ids in region_ids.items()}
            atlas_info = {
                "annotation_path": str(args.atlas.annotation_path),
                "ontology_path": str(args.atlas.ontology_path),
                "preset": args.atlas.preset,
                "orientation": list(args.atlas.orientation) if args.atlas.orientation else None,
                "resolution_um": args.atlas.resolution_um,
            }
        else:
            region_ids, region_labels, atlas_info = dict(args.region_ids), args.region_labels, None

        out_sitk = sitk.GetImageFromArray(result.volume)
        out_sitk.CopyInformation(base_sitk)      # keeps the source's (1,1,1) -- see module docstring
        sitk.WriteImage(out_sitk, args.output_path)
        regions_path, slices_path = write_guide_sidecars(
            args.output_path, args.image_path, result, region_labels, args.role,
            arr.shape[0], spacing_xyz=base_sitk.GetSpacing(),
            region_ids=region_ids, atlas_info=atlas_info)

        lines = [f"Wrote {args.output_path}", f"Wrote {regions_path}", f"Wrote {slices_path}"]
        for label in sorted(result.slices_by_label):
            lines.append(
                f"  label {label} ({_label_name(label, region_labels)}): "
                f"{len(result.slices_by_label[label])} painted planes "
                f"{result.slices_by_label[label]} -> {result.voxels_by_label[label]} voxels")
        lines += [f"WARNING: {w}" for w in guide_export_warnings(result, region_labels)]
        if region_ids:
            lines += ["", "把下面这段粘进流水线 config：", "",
                      guide_regions_yaml_snippet(region_ids, region_labels, args.output_path)]

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
    assert guide_export_warnings(result, {1: ["a"], 2: ["b"], 3: ["c"]}) == []
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

    warnings = guide_export_warnings(result, {1: ["cortex"], 3: ["corpus callosum"]})
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

    warnings = guide_export_warnings(result, {1: ["cortex"], 2: ["cerebellar hemisphere"]})
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

    warnings = guide_export_warnings(result, {1: ["cortex"], 2: ["cerebellar hemisphere"]})
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

    # label 1 carries TWO ontology entries, the case a single-name-per-label
    # sidecar could not express (DevCCF has no single "cortex" structure).
    region_labels = {1: ["layer 1 of A", "layer 2 of A"], 2: ["cerebellar hemisphere"]}
    region_ids = {1: [15751, 15756], 2: [15623]}
    out_path = tmp_dir / "s12t_guide_sample.nii.gz"
    regions_path, slices_path = write_guide_sidecars(
        out_path, "/data/s12t/registration.tif", result, region_labels, "sample",
        SHAPE[0], spacing_xyz=(1.0, 1.0, 1.0), region_ids=region_ids,
        atlas_info={"preset": "devccf_p04", "ontology_path": "/atlas/DevCCFv1_ontology.json"})

    assert regions_path.name == "s12t_guide_sample.regions.json", regions_path
    assert slices_path.name == "s12t_guide_sample.annotated_slices.json", slices_path

    regions = json.loads(regions_path.read_text())
    assert regions["regions"] == {"1": ["layer 1 of A", "layer 2 of A"],
                                  "2": ["cerebellar hemisphere"]}, regions["regions"]
    assert regions["region_ids"] == {"1": [15751, 15756], "2": [15623]}, regions["region_ids"]
    assert regions["annotated_slices"] == {"1": [0, 4, 8], "2": [2, 6]}, regions["annotated_slices"]
    assert regions["image_path"] == "/data/s12t/registration.tif"
    assert regions["atlas"]["preset"] == "devccf_p04", regions["atlas"]
    assert "voxel size" in regions["voxel_size_um_note"].lower()

    # The paste-ready pipeline snippet must carry the IDS (names are a comment
    # only -- two authorities that can disagree is what ids exist to remove).
    snippet = guide_regions_yaml_snippet(region_ids, region_labels, out_path,
                                         voxel_size_um=[2.6, 2.6, 32.0])
    parsed = yaml.safe_load(snippet)["mask"]["guide_regions"]
    assert parsed["atlas_ids"] == {1: [15751, 15756], 2: [15623]}, parsed
    assert parsed["voxel_size_um"] == [2.6, 2.6, 32.0], parsed
    assert "atlas_names" not in parsed, parsed

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
    print("8. config: region_labels int/str keys + multi-region values, display_scale_zyx")
    # A bare string and a one-element list must normalize identically -- both
    # spellings appear in real configs and reading them differently would be
    # a silent half-mapping.
    assert _normalize_region_labels({1: "cortex", "2": "cerebellar hemisphere"}) == \
        {1: ["cortex"], 2: ["cerebellar hemisphere"]}
    assert _normalize_region_labels({1: ["layer 1 of A", "layer 2 of A"]}) == \
        {1: ["layer 1 of A", "layer 2 of A"]}
    assert _normalize_region_labels(None) == {}
    assert _normalize_region_labels({}) == {}

    assert _normalize_region_ids({1: 15751, "2": [15623, 15666]}) == \
        {1: [15751], 2: [15623, 15666]}
    assert _normalize_region_ids(None) == {}

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


def _fake_ontology():
    """A 6-node ontology shaped exactly like load_ccf_ontology_json's output.
    Branch B is the "in the ontology but not in this annotation" case that
    the real DevCCF pairing has 2359 of."""
    return {
        1:   {"id": 1,   "name": "root",     "acronym": "R",  "structure_id_path": [1]},
        10:  {"id": 10,  "name": "branch A", "acronym": "BA", "structure_id_path": [1, 10]},
        100: {"id": 100, "name": "leaf A1",  "acronym": "A1", "structure_id_path": [1, 10, 100]},
        101: {"id": 101, "name": "leaf A2",  "acronym": "A2", "structure_id_path": [1, 10, 101]},
        20:  {"id": 20,  "name": "branch B", "acronym": "BB", "structure_id_path": [1, 20]},
        200: {"id": 200, "name": "leaf B1",  "acronym": "B1", "structure_id_path": [1, 20, 200]},
    }


def selftest_ontology_node_voxels():
    print("10. ontology: voxels roll up to every ancestor; empty subtrees stay empty")
    structures = _fake_ontology()
    # Label 0 is background and has no ontology entry -- it must not crash and
    # must not be credited to anything.
    totals = voxels_per_ontology_node(structures, [0, 100, 101], [50, 3, 7])

    assert totals == {1: 10, 10: 10, 100: 3, 101: 7}, totals
    assert 20 not in totals and 200 not in totals, totals

    # The invariant populate_ontology_tree relies on to disable empty nodes
    # without ever hiding a pickable descendant: a node with no voxels can
    # have no non-empty descendant, because every voxel is credited to the
    # whole ancestor chain.
    for sid, info in structures.items():
        if not totals.get(sid):
            descendants = [d for d, i in structures.items() if sid in i["structure_id_path"]]
            assert not any(totals.get(d) for d in descendants), (sid, descendants)
    print("   ok")


def selftest_ontology_tree_filter():
    print("11. ontology: search reveals ancestors, matches acronyms, respects hide-empty")
    structures = _fake_ontology()
    node_voxels = {1: 10, 10: 10, 100: 3, 101: 7}

    assert visible_tree_ids(structures, node_voxels, "", True) == {1, 10, 100, 101}
    assert visible_tree_ids(structures, node_voxels, "", False) == {1, 10, 100, 101, 20, 200}

    # A deep hit is unreachable unless its whole ancestor chain shows too.
    assert visible_tree_ids(structures, node_voxels, "leaf A2", True) == {1, 10, 101}
    # Acronyms are searchable: "SPall"/"THyA" is often what you remember.
    assert visible_tree_ids(structures, node_voxels, "a1", True) == {1, 10, 100}

    # An empty region stays hidden while hide-empty is on, and is reachable
    # (to look at, not to assign) once it's off.
    assert visible_tree_ids(structures, node_voxels, "leaf B1", True) == set()
    assert visible_tree_ids(structures, node_voxels, "leaf B1", False) == {1, 20, 200}
    print("   ok")


def selftest_seed_assignment():
    print("12. config -> GUI assignment: ids kept, names resolved only when unambiguous")
    structures = _fake_ontology()
    structures[300] = {"id": 300, "name": "leaf A1",   # deliberate duplicate name
                       "acronym": "DUP", "structure_id_path": [1, 20, 300]}

    assignment, unresolved = _seed_assignment(
        region_labels={1: ["branch A"], 2: ["nope"], 3: ["leaf A1"]},
        region_ids={1: [101], 4: [200]},
        structures=structures)

    # Ids pass through; a unique name resolves and is appended alongside them.
    assert assignment[1] == [101, 10], assignment
    assert assignment[4] == [200], assignment
    # A name matching nothing, and one matching two structures, both refuse to
    # guess -- substring/ambiguous matching is exactly what ids exist to avoid.
    assert 2 not in assignment and 3 not in assignment, assignment
    assert sorted((lab, n) for lab, _name, n in unresolved) == [(2, 0), (3, 2)], unresolved
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


def selftest_compact_annotation():
    print("13. atlas: chunked relabelling is exact, and small enough to be worth it")
    rng = np.random.default_rng(3)
    ids = np.array([0, 7, 15564, 21558], dtype=np.float32)   # real DevCCF-scale ids
    annotation = ids[rng.integers(0, len(ids), size=(70, 12, 9))].astype(np.float32)

    compact, present_ids = _compact_annotation(annotation, chunk=16)   # chunk < z, so it wraps

    assert np.array_equal(present_ids, ids.astype(np.int64)), present_ids
    assert compact.dtype == np.uint8, compact.dtype
    # Round-tripping through the index must reproduce the annotation exactly:
    # a chunk-boundary bug would corrupt only some planes, which is precisely
    # the kind of thing that would go unnoticed in the GUI.
    assert np.array_equal(present_ids[compact], annotation.astype(np.int64))

    counts = np.bincount(compact.ravel(), minlength=len(present_ids))
    assert counts.sum() == annotation.size
    for i, sid in enumerate(present_ids):
        assert counts[i] == int((annotation == sid).sum()), (sid, counts[i])

    # >255 distinct labels has to widen, or ids would silently collide.
    many = np.arange(300, dtype=np.float32).reshape(300, 1, 1) * np.ones((300, 2, 2), np.float32)
    wide, wide_ids = _compact_annotation(many, chunk=32)
    assert wide.dtype == np.uint16, wide.dtype
    assert np.array_equal(wide_ids[wide], many.astype(np.int64))
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
    selftest_ontology_node_voxels()
    selftest_ontology_tree_filter()
    selftest_seed_assignment()
    selftest_compact_annotation()
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
                                region_labels=cfg.region_labels, region_ids=cfg.region_ids,
                                display_scale_zyx=cfg.display_scale_zyx, atlas=cfg.atlas)
        _run_guide(args)
    else:
        raise ValueError(f"Unknown kind: {cfg.kind!r} (expected 'mask' or 'guide')")

    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
