"""Interactive tool: paint a guide outline on a 3D sample volume.

A guide outline marks a structure that is genuinely present in both images
but needs help being aligned correctly (e.g. a bulged/deformed patch of
cortex that keeps ending up mapped to background). It is NOT an
inclusion/exclusion mask -- it marks tissue to *actively align*, consumed
as a paired sample+atlas outline feeding ants.registration()'s
multivariate_extras (see register.register_to_atlas's `guide_regions`
parameter and ../Registration_ants/scripts/project_outline.py), never as a
`mask`/`moving_mask` argument.

Sparse keyframes only: you paint a handful of representative planes and the
rest is interpolated between them, since a guide outline is a bounded 3D
blob rather than something that needs a meaningful value on every plane.

  Only the SAMPLE side is painted. The atlas side is not drawn by hand at
    all: the atlas ships a complete annotation volume (e.g.
    P04_DevCCF_Annotations_20um.nii.gz) from which Registration_ants builds
    the matching atlas-side outline by looking the region up. That is what
    makes the label->region mapping load-bearing rather than cosmetic -- it
    is the only thing tying "the blob I painted" to "which atlas structure
    to pair it with".

  Picking the region: point `atlas_annotation_path` + `ontology_path` at an
    atlas and the viewer grows an ontology tree. Selecting any node
    highlights that structure AND all its descendants -- the only way a
    high-level node shows anything, since the annotation's own labels sit at
    ontology depths 2-12 and a depth-3 node owns no voxels under its own id.
    Assign the selected region to a brush label right there; the export
    records the ontology IDS (see write_guide_sidecars for why ids rather
    than names).

  A "Relabel" panel: click-to-fill a single already-painted blob into
    another label, or renumber one label across the whole volume. A bulk
    renumber carries that label's ontology assignment with it, so the number
    keeps meaning the same region.

    The atlas is drawn in a SECOND, closable window (see _open_atlas_window
    for the measured reason it must not share the sample's) as three stacked
    layers -- the grayscale template, the complete annotation in colour, and
    the region currently selected in the tree -- each toggleable from napari's
    own layer list, and in three synced ortho panes so a region's shape is
    visible in all three planes even though the sample is only cut in one
    (`atlas_ortho_views: false` gives the old single pane back). Left-clicking
    any pane recentres all three on that point; hovering shows the full
    ontology chain of the voxel under the cursor, not just the leaf structure
    the annotation stores.

    The atlas grid is INDEPENDENT of the sample's: a half-brain sample against
    a whole-brain atlas is the normal case, and needs nothing but pointing
    atlas_template_path / atlas_annotation_path at the whole-brain files. The
    atlas is a reference only, never registered to the sample -- what is
    exported is ids plus voxel indices on the sample's own grid, so nothing
    about how the atlas is displayed, oriented, cropped or downsampled can
    reach the output. The one thing that must match the pipeline's atlas is
    the ONTOLOGY, since ids are what crosses over.

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

  2. Axis order: images are read via SimpleITK
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
QCheckBox = QHBoxLayout = QLineEdit = QScrollArea = QSpinBox = None
QTreeWidget = QTreeWidgetItem = Qt = QSplitter = None
ViewerModel = QtViewer = None


def _import_gui():
    """Bind the napari/Qt names used by the viewer code. Called once at the
    top of the GUI entry points; import errors surface there rather than at
    module import, which is what keeps --selftest env-independent."""
    global napari, QLabel, QPushButton, QVBoxLayout, QWidget
    global QCheckBox, QHBoxLayout, QLineEdit, QScrollArea, QSpinBox, QSplitter
    global QTreeWidget, QTreeWidgetItem, Qt
    global ViewerModel, QtViewer
    import napari as _napari
    from napari.components import ViewerModel as _ViewerModel
    from napari.qt import QtViewer as _QtViewer
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtWidgets import (QCheckBox as _QCheckBox, QHBoxLayout as _QHBoxLayout,
                                 QLabel as _QLabel, QLineEdit as _QLineEdit,
                                 QPushButton as _QPushButton, QScrollArea as _QScrollArea,
                                 QSpinBox as _QSpinBox, QSplitter as _QSplitter,
                                 QTreeWidget as _QTreeWidget, QTreeWidgetItem as _QTreeWidgetItem,
                                 QVBoxLayout as _QVBoxLayout, QWidget as _QWidget)
    napari, QLabel, QPushButton = _napari, _QLabel, _QPushButton
    QVBoxLayout, QWidget = _QVBoxLayout, _QWidget
    QCheckBox, QHBoxLayout, QLineEdit, QSpinBox = _QCheckBox, _QHBoxLayout, _QLineEdit, _QSpinBox
    QScrollArea, QSplitter = _QScrollArea, _QSplitter
    QTreeWidget, QTreeWidgetItem, Qt = _QTreeWidget, _QTreeWidgetItem, _Qt
    # ViewerModel + QtViewer are the "a napari canvas without its own window"
    # pair the ortho panes are built from (see _open_atlas_window).
    ViewerModel, QtViewer = _ViewerModel, _QtViewer


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
        required=("image_path", "output_path"),
        legacy_paths=_LEGACY_CONFIG_PATHS)
    return SimpleNamespace(
        image_path=cfg["image_path"],
        output_path=cfg["output_path"],
        existing_mask_path=cfg.get("existing_mask_path") or None,
        region_labels=_normalize_region_labels(cfg.get("region_labels") or {}),
        region_ids=_normalize_region_ids(cfg.get("region_ids") or {}),
        display_scale_zyx=_normalize_display_scale(cfg.get("display_scale_zyx")),
        atlas=_atlas_reference_config(cfg),
    )


def _atlas_reference_config(cfg):
    """Resolve the optional atlas reference -> SimpleNamespace, or None when no
    atlas is configured.

    Only the annotation + ontology are strictly needed -- they are what the
    region picker resolves against. The template is just the grayscale
    backdrop and may be omitted; resolution_um and orientation only affect how
    the atlas is drawn.

    The ontology has to be the SAME one the pipeline registers against, since
    what the export carries over is ontology ids. The paths themselves do not:
    the atlas here is never registered to the sample, so a whole-brain atlas
    beside a half-brain sample is the normal case.
    """
    paths = {
        "template_path": cfg.get("atlas_template_path"),
        "annotation_path": cfg.get("atlas_annotation_path"),
        "ontology_path": cfg.get("ontology_path") or cfg.get("atlas_ontology_path"),
    }
    if not any(paths.values()):
        return None

    for required in ("annotation_path", "ontology_path"):
        if not paths[required]:
            key = "ontology_path" if required == "ontology_path" else f"atlas_{required}"
            raise ValueError(f"图谱参考需要 {key}（配置里没写）")
        if not Path(paths[required]).exists():
            raise FileNotFoundError(f"图谱 {required} 不存在: {paths[required]}")
    if paths["template_path"] and not Path(paths["template_path"]).exists():
        raise FileNotFoundError(f"图谱 template_path 不存在: {paths['template_path']}")

    downsample = cfg.get("atlas_downsample")
    downsample = 1 if downsample is None else int(downsample)
    if downsample < 1:
        raise ValueError(f"atlas_downsample 必须 >= 1，得到 {downsample}")

    return SimpleNamespace(
        template_path=paths["template_path"] or None,
        annotation_path=paths["annotation_path"],
        ontology_path=paths["ontology_path"],
        resolution_um=float(cfg.get("atlas_resolution_um") or 0) or None,
        orientation=cfg.get("atlas_orientation") or None,
        downsample=downsample,
        # Three synced canvases instead of one. Off is the escape hatch for a
        # small screen, or for a machine where three GL canvases over the same
        # volume is more than the driver wants to do.
        ortho=bool(cfg.get("atlas_ortho_views", True)),
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


def _launch_viewer(arr, prefill, scale=None):
    """The sample window: the grayscale volume plus the layer painted on.

    scale: optional (z, y, x) physical size per voxel, applied to BOTH layers
    so they stay registered to each other. Without it a raw 2.6/2.6/32 um
    stack is drawn as if it were isotropic, i.e. squashed 12x along z, which
    makes the orthogonal views unusable. Purely a display transform -- layer
    .data, and therefore the export, is untouched."""
    viewer = napari.Viewer(title="Paint guide outline")
    scale_kwargs = {"scale": scale} if scale is not None else {}
    viewer.add_image(arr, name="sample", colormap="gray", **scale_kwargs)
    paint_layer = viewer.add_labels(prefill.copy(), name="guide outline (paint here)",
                                    **scale_kwargs)
    return viewer, paint_layer


def annotation_features(atlas):
    """Hover text for the full-annotation layer, one row per compact label.

    That layer holds COMPACT indices (see _compact_annotation), which mean
    nothing on their own -- index 37 is not structure 37. napari's Labels
    layer appends a features row to the status bar next to the value under
    the cursor, matched through the 'index' column, so mapping compact index
    -> the real structure's name and id here is what makes the layer readable
    without hunting for the label in the tree.
    """
    names, ids = [], []
    for sid in atlas.present_ids:
        sid = int(sid)
        info = atlas.structures.get(sid)
        if sid == 0:
            names.append("背景")
        elif info is None:
            names.append(f"ontology 里没有 id {sid}")
        else:
            acronym = info.get("acronym")
            names.append(f"{info['name']} ({acronym})" if acronym else info["name"])
        ids.append(sid)
    return {"index": np.arange(len(atlas.present_ids)), "脑区": names, "id": ids}


# (dims.order, pane title). napari displays the LAST `ndisplay` axes of
# dims.order, so each entry names the axis kept on the slider, i.e. the axis
# the plane is perpendicular to. Index 0 is the sample's own stack axis, so
# the first pane is the view the sample is painted in and the other two are
# the reconstructions.
_ORTHO_PANES = (
    ((0, 1, 2), "轴0 切面（样本视角）"),
    ((1, 0, 2), "轴1 切面"),
    ((2, 0, 1), "轴2 切面"),
)

# Layer attributes mirrored from the main pane onto the ortho panes. Only the
# main pane has a layer list, so these are the knobs a user can actually reach;
# everything else about the sub-panes is fixed at construction.
_MIRRORED_LAYER_ATTRS = ("visible", "opacity", "contrast_limits", "gamma", "colormap")

# Starting height in px of the bottom dock holding the two reconstructed
# views, leaving the main canvas the rest. Draggable afterwards.
_ORTHO_DOCK_HEIGHT = 300

# How far the mouse may travel between press and release and still count as a
# click rather than the start of a camera pan (see _click_moves_crosshair).
_CLICK_SLOP_PX = 4


def crosshair_vectors(centre, shape):
    """napari Vectors data (3, 2, 3): one full-extent line along each axis,
    all three crossing at voxel `centre`.

    Two of the three land in any given pane and the third does not, which is
    exactly the classic ortho crosshair and costs no per-pane bookkeeping. A
    pane slicing axis k draws the axes it displays: those lines start at 0 on
    the axes it slices, i.e. at `centre` on the slider axis, so they are in the
    current slice. The line ALONG axis k starts at 0 on axis k, so it only
    matches slice 0 and is otherwise (correctly) invisible -- it is the line
    pointing straight at the viewer.
    """
    ndim = len(shape)
    vectors = np.zeros((ndim, 2, ndim), dtype=float)
    for axis in range(ndim):
        start = list(centre)
        start[axis] = 0
        vectors[axis, 0] = start
        vectors[axis, 1, axis] = shape[axis]      # length: the whole extent
    return vectors


def _add_atlas_layers(model, atlas, scale_kwargs, features):
    """The atlas's four layers, added to one ViewerModel in draw order.

    Called once per ortho pane. Every pane gets its OWN Layer objects -- one
    Layer cannot belong to two ViewerModels -- but they are all backed by the
    same numpy arrays, so three panes cost three sets of slice textures, not
    three copies of a 287 MB annotation.
    """
    template = None
    if atlas.template is not None:
        template = model.add_image(atlas.template, name="reference（模板灰度）",
                                   colormap="gray", **scale_kwargs)
    # atlas.compact itself, not a copy: 190-odd labels over an 800^3 grid is
    # the biggest array in the process, and the layer is read-only anyway.
    annotation = model.add_labels(atlas.compact, name="annotation（全部脑区）",
                                  opacity=0.45, features=features, **scale_kwargs)
    annotation.editable = False
    highlight = model.add_labels(np.zeros(atlas.compact.shape, dtype=np.uint8),
                                 name="selection（选中的脑区）", opacity=0.85,
                                 colormap=napari.utils.DirectLabelColormap(
                                     color_dict={None: "transparent", 1: "red"}),
                                 **scale_kwargs)
    highlight.editable = False              # a reference; painting here would mean nothing
    centre = tuple(dim // 2 for dim in atlas.compact.shape)
    cross = model.add_vectors(crosshair_vectors(centre, atlas.compact.shape),
                              name="十字准线", edge_color="cyan", edge_width=1.5,
                              vector_style="line", opacity=0.9, **scale_kwargs)
    return SimpleNamespace(template=template, annotation=annotation,
                           highlight=highlight, cross=cross)


def mask_centre_index(mask):
    """(i0, i1, i2) voxel index at the middle of `mask`'s extent on each axis,
    or None when the mask is empty.

    The median-of-occupied-planes per axis, not the centre of mass: for a
    C-shaped or bilateral structure the centre of mass can land in a plane the
    structure never touches, and the whole point of jumping there is to see it
    in all three panes at once.
    """
    centre = []
    for axis in range(mask.ndim):
        planes = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis)))
        if not planes.size:
            return None
        centre.append(int(planes[len(planes) // 2]))
    return tuple(centre)


def _open_atlas_window(atlas, resolution_um, ortho=True):
    """A SECOND napari window holding the atlas alone.

    THREE LAYERS, bottom to top: the grayscale template, the COMPLETE
    annotation in colour, and the region the ontology tree currently selects.
    All three are ordinary napari layers, so the layer list's own checkboxes
    and opacity slider are the controls -- annotation off to read the plain
    template, annotation on to see where the selection sits among its
    neighbours. The annotation starts semi-transparent and the selection is
    forced to solid red rather than taking a colormap colour, so it stays
    distinguishable on top of 190-odd annotation colours.

    THREE PANES (`ortho`), one per anatomical axis, sharing one slice position:
    the sample is usually cut in one plane only, but the atlas is a volume and
    a region's shape in the other two planes is what tells you whether the
    tree selection is the structure you meant. napari has no ortho mode, so
    each pane is its own ViewerModel with its own dims.order, and
    `_sync_ortho_panes` keeps their sliders together; pane 0 is the real napari
    Viewer (it owns the window, the layer list and the layer controls) and the
    other two are bare QtViewer canvases in a dock beneath it.

    ONE CROSSHAIR over all of them: left-clicking any pane recentres all three
    on the clicked point, the way every other ortho viewer behaves. The
    crosshair is derived from dims.current_step rather than stored, so clicks,
    slider drags and "jump to region" all drive it through the same path and
    it cannot end up pointing at a plane no pane is showing.

    A HOVER PANEL (_add_ancestry_panel) on the right, because the one structure
    a voxel is labelled with is usually a leaf ("layer 5 of primary motor
    area") and the level you are actually picking is one of its ancestors.

    The atlas grid is INDEPENDENT of the sample's -- a half-brain sample
    against a whole-brain atlas is the normal case here, and works because
    nothing in this window is registered to the sample. What the export
    records is ontology ids plus voxel indices on the sample's own grid, so
    the atlas's own extent, orientation and downsampling cannot reach it.

    Kept out of the sample's window on purpose. Sharing one viewer means
    sharing one z slider, whose step is the smallest scale across all layers
    -- so an atlas at 20um beside a sample at 32um put slider positions on
    fractional sample planes (0.62, 1.25, 1.88, ...), where napari 0.8.0's
    renderer and its paint tool round differently: measured, 5 of every 12
    slider positions painted one plane BEFORE the plane on screen, silently.
    Separate windows remove the shared slider, and with it that whole class of
    bug -- the sample's slider is driven by the sample alone again. It also
    means the atlas keeps its own true voxel size here instead of being
    rescaled to stay commensurate with the sample's.

    The window is disposable: nothing here is needed to paint or export (the
    ontology tree and the label assignment live in the sample's window), so
    closing it loses no state, and reopening rebuilds it from `atlas`.
    """
    scale_kwargs = {"scale": [float(resolution_um) * atlas.downsample] * 3} if resolution_um else {}
    features = annotation_features(atlas)
    panes = tuple(_ORTHO_PANES if ortho else _ORTHO_PANES[:1])

    viewer = napari.Viewer(title="Atlas reference (read-only)")
    main = SimpleNamespace(model=viewer, qt=None, order=panes[0][0],
                           layers=_add_atlas_layers(viewer, atlas, scale_kwargs, features))
    built = [main]

    for order, title in panes[1:]:
        model = ViewerModel(ndisplay=2)
        # Canvas BEFORE layers: napari 0.8.0's QtViewer.__init__ walks the
        # layers already in the model and reorders them against a visual map
        # it has not filled in yet, which raises KeyError on the second layer.
        # Adding to an empty model routes through the same code path one layer
        # at a time, where the map is always current.
        qt = QtViewer(model)
        built.append(SimpleNamespace(
            model=model, qt=qt, order=order,
            layers=_add_atlas_layers(model, atlas, scale_kwargs, features)))
        model.dims.order = order
        model.reset_view()

    if len(built) > 1:
        splitter = QSplitter(Qt.Horizontal)
        for pane, (_order, title) in zip(built[1:], panes[1:]):
            wrapper = QWidget()
            layout = QVBoxLayout(wrapper)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QLabel(title))
            layout.addWidget(pane.qt)
            # A bare QtViewer asks for 800x626 and Qt would honour both of
            # them: two side by side in a bottom dock open a ~1600px-wide
            # window whose main canvas is a letterbox. A small floor plus the
            # explicit resizeDocks below makes the ortho row a strip that the
            # user can then drag to whatever they actually want.
            pane.qt.setMinimumSize(160, 120)
            splitter.addWidget(wrapper)
        dock = viewer.window.add_dock_widget(splitter, area="bottom", name="正交视图（同步）")
        viewer.window._qt_window.resizeDocks([dock], [_ORTHO_DOCK_HEIGHT], Qt.Vertical)
        _sync_ortho_panes(built)
        _mirror_layer_attrs(built)
        # QtViewer holds a vispy canvas and its GL context; napari only cleans
        # up the ones its own Window created, so these two have to be closed by
        # hand or the context outlives the window it was drawn in.
        viewer.window._qt_window.destroyed.connect(
            lambda *_a: [pane.qt.close() for pane in built[1:]])

    def set_highlight(mask, name=None):
        """The selection layer's data, on every pane at once.

        One array shared by all three layers rather than one copy each: the
        mask is a full-volume allocation and highlight_mask already made it.
        """
        for pane in built:
            pane.layers.highlight.data = mask
            if name:
                pane.layers.highlight.name = name

    def centre_on(index):
        """Point every pane at voxel `index` -- so the two reconstructions land
        on it too, not just the pane whose slider was moved or clicked."""
        for axis, value in enumerate(index):
            value = int(np.clip(round(float(value)), 0, atlas.compact.shape[axis] - 1))
            viewer.dims.set_point(axis, value * float(main.layers.annotation.scale[axis]))

    def refresh_cross(*_args):
        """The crosshair is DERIVED from the slice position, never stored
        separately: slider drag, click, and 'jump to region' then all reach it
        through one path, and none of them can leave it pointing somewhere the
        panes are not showing."""
        step = viewer.dims.current_step
        if len(step) != atlas.compact.ndim:
            # Closing the window empties the layer list, which collapses
            # dims.ndim to 2 and emits current_step on the way down. There is
            # nothing to draw a crosshair on any more.
            return
        data = crosshair_vectors(tuple(int(v) for v in step), atlas.compact.shape)
        for pane in built:
            pane.layers.cross.data = data

    # Connected to the MAIN pane only: _sync_ortho_panes pushes every pane's
    # change into it, so its event is the one place all of them converge.
    viewer.dims.events.current_step.connect(refresh_cross)
    for pane in built:
        pane.model.mouse_drag_callbacks.append(_click_moves_crosshair(pane, centre_on))
    refresh_cross()

    hover = _add_ancestry_panel(viewer, atlas, built)

    return SimpleNamespace(viewer=viewer, panes=built, highlight=main.layers.highlight,
                           annotation=main.layers.annotation, hover=hover,
                           set_highlight=set_highlight, centre_on=centre_on)


def _click_moves_crosshair(pane, centre_on):
    """Left-click anywhere in a pane -> crosshair there, in every pane.

    A generator callback, which is napari's way of telling a click from the
    start of a drag: napari resumes it on every mouse_move and once more on
    release, so the decision can be made after the fact. A plain press handler
    cannot -- the same press that begins a camera pan would jump the crosshair
    on its way, and the atlas would slide out from under the pan.
    """
    def callback(_model, event):
        if event.button != 1 or event.modifiers:
            return                          # right-click menus and modified drags stay napari's
        origin = np.asarray(event.pos, dtype=float)
        dragged = False
        yield
        while event.type == "mouse_move":
            if np.abs(np.asarray(event.pos, dtype=float) - origin).max() > _CLICK_SLOP_PX:
                dragged = True
            yield
        if not dragged:
            # event.position is world coords on ALL axes -- the two the pane
            # draws come from the cursor, the third from its own slider -- so
            # this is a full 3D point, not just the 2D one that was clicked.
            centre_on(pane.layers.annotation.world_to_data(event.position))
    return callback


def _sync_ortho_panes(panes):
    """Make every pane's slider position follow every other one's.

    All panes carry the same layers on the same grid and scale, so
    `dims.current_step` is one shared (i0, i1, i2) and can be copied across
    verbatim -- what differs between panes is only dims.order, i.e. which of
    the three a pane draws instead of slides.

    The `busy` flag is what stops the obvious infinite loop: assigning
    current_step on pane B emits B's own current_step event, which would push
    straight back at A.
    """
    busy = {"in": False}

    def follow(source):
        def _handler(*_args):
            if busy["in"]:
                return
            busy["in"] = True
            try:
                step = source.model.dims.current_step
                for pane in panes:
                    if pane is source or pane.model.dims.current_step == step:
                        continue
                    if pane.model.dims.ndim != len(step):
                        continue        # a pane already torn down by window close
                    pane.model.dims.current_step = step
            finally:
                busy["in"] = False
        return _handler

    for pane in panes:
        pane.model.dims.events.current_step.connect(follow(pane))


def _mirror_layer_attrs(panes):
    """Push the main pane's layer settings onto the ortho panes.

    Only the main pane has a layer list and layer controls, so a visibility
    checkbox or opacity slider there has to reach the other two canvases or
    they drift out of agreement with the window they live in. Attributes are
    connected by name and skipped where the layer type has no such event
    (an Image has contrast_limits, a Labels does not), so the same list covers
    the template, the annotation and the selection.
    """
    main, followers = panes[0], panes[1:]
    for role in ("template", "annotation", "highlight", "cross"):
        source = getattr(main.layers, role)
        if source is None:
            continue
        targets = [getattr(pane.layers, role) for pane in followers]
        for attr in _MIRRORED_LAYER_ATTRS:
            emitter = getattr(source.events, attr, None)
            if emitter is None or not all(hasattr(t, attr) for t in targets):
                continue

            def _push(*_args, _src=source, _targets=targets, _attr=attr):
                value = getattr(_src, _attr)
                for target in _targets:
                    if getattr(target, _attr) != value:
                        setattr(target, _attr, value)

            emitter.connect(_push)


def format_ancestry(structures, structure_id):
    """The whole root -> leaf chain of `structure_id`, one level per line.

    What the status bar can show is the ONE structure a voxel is labelled
    with, which for a fine-grained annotation is a leaf like "layer 5 of
    primary motor area" -- true, and useless for deciding whether you are
    looking at cortex. The ontology already carries the answer in
    structure_id_path; this just renders it, so the level you actually care
    about is on screen next to the level the annotation happens to store.

    The voxel's own structure is marked, since it is usually NOT the last line
    a user cares about, and every line carries its id -- ids are what the
    export records, so being able to read one off directly is the point.
    """
    info = structures.get(structure_id)
    if info is None:
        return f"id {structure_id}：不在 ontology 里（annotation 用了这个 label，但字典里查不到）"
    lines = []
    for depth, sid in enumerate(info["structure_id_path"]):
        node = structures.get(sid)
        marker = "▶" if sid == structure_id else "·"
        indent = "  " * depth
        if node is None:
            lines.append(f"{indent}{marker} [{sid}] ?")
            continue
        acronym = node.get("acronym")
        name = f"{node['name']} ({acronym})" if acronym else node["name"]
        lines.append(f"{indent}{marker} {name}  [{sid}]")
    return "\n".join(lines)


def _add_ancestry_panel(viewer, atlas, panes):
    """A dock on the atlas window showing the full ontology chain of whatever
    the mouse is over, updated live from every pane.

    Returns SimpleNamespace(label=, show=) so the behaviour is testable without
    synthesising Qt mouse events: `show` takes a compact index and is the whole
    of what the mouse callbacks do.
    """
    label = QLabel("把鼠标放到图谱上，这里显示该体素所在脑区的完整层级。")
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.setWordWrap(False)

    # mouse_move fires continuously; re-rendering the same chain on every pixel
    # of travel is pure waste, and the flicker is visible.
    last = {"index": -1}

    def show(compact_index):
        if compact_index is None:
            compact_index = 0
        compact_index = int(compact_index)
        if compact_index == last["index"]:
            return
        last["index"] = compact_index
        if compact_index == 0:
            label.setText("（背景，没有脑区）")
            return
        sid = int(atlas.present_ids[compact_index])
        voxels = atlas.node_voxels.get(sid, 0)
        label.setText(format_ancestry(atlas.structures, sid)
                      + f"\n\n该结构（含后代）共 {voxels:,} 体素")

    def watcher(pane):
        def on_move(_model, event):
            dims_displayed = getattr(event, "dims_displayed", None)
            # get_value concatenates this onto a list internally, so it has to
            # BE a list -- napari's own canvas passes one, but dims.displayed
            # itself is a tuple and reaches here unchanged on other paths.
            if dims_displayed is not None:
                dims_displayed = list(dims_displayed)
            show(pane.layers.annotation.get_value(
                event.position,
                view_direction=getattr(event, "view_direction", None),
                dims_displayed=dims_displayed,
                world=True))
        return on_move

    for pane in panes:
        pane.model.mouse_move_callbacks.append(watcher(pane))

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(QLabel("鼠标所在脑区的完整层级（▶ 是 annotation 实际存的那一级）"))
    layout.addWidget(_scrollable(label, 220))
    viewer.window.add_dock_widget(dock, area="right", name="脑区层级")
    return SimpleNamespace(label=label, show=show)


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


def voxel_size_um_from_display_scale(display_scale_zyx):
    """display_scale_zyx (z,y,x) -> the pipeline's voxel_size_um (x,y,z), or
    None when no display scale was configured.

    The two configs describe the same physical voxel in OPPOSITE axis orders
    -- display_scale_zyx matches sitk.GetArrayFromImage's (z,y,x), while
    mask.guide_regions.voxel_size_um is (x,y,z) like every other micron
    triple in the pipeline. Reversing it by hand is exactly the kind of step
    that gets silently miscopied, and a reversed voxel_size_um does not
    error: it just resamples the outline against the wrong physical size.
    """
    return list(reversed(display_scale_zyx)) if display_scale_zyx else None


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
    note = ("# 原图 (x,y,z) µm，由 display_scale_zyx 反序得到 -- 核对一下"
            if voxel_size_um else
            "# 原图 (x,y,z) µm -- tif header 里没有，必须手填")
    lines = [
        "mask:",
        "  guide_regions:",
        f"    regions_mask: {output_path}",
        f"    voxel_size_um: [{voxel[0]}, {voxel[1]}, {voxel[2]}]   {note}",
        "    atlas_ids:",
    ]
    for label in sorted(region_ids):
        names = ", ".join(region_names.get(label, []))
        lines.append(f"      {label}: {list(region_ids[label])}" + (f"   # {names}" if names else ""))
    lines.append("    weight: 1.0")
    return "\n".join(lines)


def _scrollable(label, min_height):
    """A growing text label wrapped in a scroll area, so its content length
    stops driving the whole window's minimum size.

    Load-bearing, not cosmetic. A word-wrapped QLabel reports its ENTIRE
    laid-out text as its minimumSizeHint, and Qt propagates that up the chain
    dock content -> QDockWidget -> napari's QMainWindow. Measured on this
    panel layout: filling the export label with a normal 4-label report took
    the main window's minimum height from 854 px to 1238 px, i.e. past the
    work area of a 1080p screen. Qt then resizes the window to satisfy the
    new minimum, and resizing a maximized/fullscreen window drops it back to
    normal -- that is the "clicking Export exits fullscreen" bug -- after
    which the window can never be made shorter again, so the z slider at the
    bottom sits under the taskbar for good.

    Widening/narrowing is no escape either, it is the wrong way round: a
    word-wrapped label has heightForWidth, so a NARROWER dock needs MORE
    height (measured on the same report: 648 px at 600 px wide, 1201 px at
    260 px wide). The one dimension still free makes it worse.

    A QScrollArea's own minimum is a small fixed size, so the report scrolls
    inside the panel instead of pushing the window around.
    """
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    scroll = QScrollArea()
    scroll.setWidget(label)
    scroll.setWidgetResizable(True)
    scroll.setMinimumHeight(min_height)
    return scroll


def _make_export_dock(viewer, status_label, on_export, button_text, panel_name):
    # Selectable because the guide export prints a ready-to-paste
    # guide_regions block into this label (guide_regions_yaml_snippet) --
    # retyping it out of the terminal is exactly the corruption those ids
    # exist to prevent.
    status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    export_btn = QPushButton(button_text)
    export_btn.clicked.connect(on_export)

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(_scrollable(status_label, 120))
    layout.addWidget(export_btn)
    viewer.window.add_dock_widget(dock, area="right", name=panel_name)


def _add_relabel_panel(viewer, paint_layer, on_change=None):
    """A "change what an already-painted region is labelled" panel.

    Two operations, because "the label of this blob is wrong" splits into
    two different jobs and only one of them is a bulk edit:

      pick + fill: click a blob, it takes the target label. This is napari's
        own FILL mode, exposed as a button because the tool is otherwise a
        keyboard shortcut people do not find. n_edit_dimensions is forced to
        1 first: the default would flood-fill across z, and on a sparse
        keyframe stack the planes are not connected anyway, so a 3D fill
        either does nothing extra or -- once the interpolated resume of a
        previous export is loaded -- silently eats neighbouring planes.

      relabel all: renumber every voxel of one label at once, for when a
        whole region was drawn under the wrong number.

    on_change is called after a bulk relabel so the caller can refresh
    anything keyed on label numbers (the guide mode's assignment panel).
    """
    from_spin, to_spin = QSpinBox(), QSpinBox()
    for spin in (from_spin, to_spin):
        spin.setRange(0, MAX_LABEL)
    from_spin.setValue(1)
    to_spin.setValue(2)

    note = QLabel("选中已画的区域改 label")
    status = QLabel("")
    status.setWordWrap(True)

    def start_fill():
        # selected_label is what FILL paints with, so set it from `to`.
        paint_layer.n_edit_dimensions = 1
        paint_layer.selected_label = int(to_spin.value())
        paint_layer.mode = "fill"
        status.setText(f"填充模式：点任意一块，它就变成 label {int(to_spin.value())}。"
                       "（改完记得切回画笔模式再继续画）")

    def relabel_all():
        src, dst = int(from_spin.value()), int(to_spin.value())
        if src == dst:
            status.setText("from 和 to 一样，没什么可改的。")
            return
        n = relabel_volume(paint_layer.data, src, dst)
        paint_layer.refresh()
        if on_change is not None:
            on_change(src, dst)
        status.setText(f"label {src} -> {dst}：改了 {n} 个体素。"
                       + ("" if n else "（那个 label 一个体素都没画过）"))

    fill_btn = QPushButton("点击填充改 label")
    fill_btn.clicked.connect(start_fill)
    all_btn = QPushButton("整个 label 全改")
    all_btn.clicked.connect(relabel_all)

    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.addWidget(QLabel("from"))
    row_layout.addWidget(from_spin)
    row_layout.addWidget(QLabel("to"))
    row_layout.addWidget(to_spin)

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(note)
    layout.addWidget(row)
    layout.addWidget(fill_btn)
    layout.addWidget(all_btn)
    layout.addWidget(_scrollable(status, 60))
    viewer.window.add_dock_widget(dock, area="right", name="Relabel")


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


def write_guide_sidecars(output_path, image_path, result, region_labels, total_z,
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


def load_guide_resume(existing_path, expected_shape):
    """Restore a previous guide export as EDITABLE keyframes, or None if it
    can't be resumed that way.

    Reloading the exported volume directly is wrong twice over, which is why
    this exists rather than reusing _load_mask_array:

      1. That function binarizes (`> 0`), collapsing a multi-label outline
         into a single label -- every region you separated would merge.
      2. The exported volume is DENSE: interpolation already filled every
         plane between the first and last keyframe. Re-reading it makes all
         of those look hand-drawn, so the next export interpolates on top of
         the previous export's own guess instead of on your real keyframes.
         A 5-plane job comes back as 11 planes and drifts further every round.

    So only the planes the `.regions.json` sidecar recorded as hand-drawn are
    restored, at their original label values -- the same "overlay only the
    real keyframes onto a fresh baseline" rule edit_sample_labels.py's
    _load_prior_hand_drawn follows, for the same reason.

    Note the restored plane holds what SURVIVED export: where two labels'
    interpolations collided, the higher label id won (see
    interpolate_labels_separately), so a keyframe overlapped by a
    higher-numbered region comes back missing those voxels. The export
    warning about contested voxels is what flags that at the time.

    Returns SimpleNamespace(prefill, region_ids, region_labels,
    slices_by_label) -- region_* being what the sidecar recorded, ready to
    re-seed the assignment panel so the label numbers keep their meaning.
    """
    sidecar = _output_stem(existing_path)
    sidecar = sidecar.with_name(sidecar.name + ".regions.json")
    if not sidecar.exists():
        return None

    meta = json.loads(sidecar.read_text())
    annotated = meta.get("annotated_slices") or {}
    if not annotated:
        return None

    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(existing_path)))
    if arr.shape != expected_shape:
        print(f"WARNING: 续画文件 shape {arr.shape} != 图像 shape {expected_shape}，不预填。")
        return None

    prefill = np.zeros(expected_shape, dtype=np.uint8)
    slices_by_label = {}
    for raw_label, planes in annotated.items():
        label = int(raw_label)
        planes = [z for z in planes if 0 <= z < expected_shape[0]]
        for z in planes:
            prefill[z][arr[z] == label] = label
        slices_by_label[label] = sorted(planes)

    return SimpleNamespace(
        prefill=prefill,
        slices_by_label=slices_by_label,
        region_ids=_normalize_region_ids(meta.get("region_ids") or {}),
        region_labels=_normalize_region_labels(meta.get("regions") or {}),
        sidecar=sidecar,
    )


def relabel_volume(volume, from_label, to_label):
    """Renumber one brush label across a whole painted volume, in place.

    Returns the number of voxels changed. Separate from the GUI so the
    selftests can pin the semantics: it is a pure renumber, so pointing two
    labels at the same number MERGES them rather than erroring -- the
    keyframe bookkeeping downstream is per-label, and a merge is a thing you
    might actually want (two halves of one region drawn separately).
    """
    if not (0 <= to_label <= MAX_LABEL):
        raise ValueError(f"to_label 必须在 0..{MAX_LABEL} 之间（导出是 uint8），得到 {to_label}")
    hit = volume == from_label
    n = int(np.count_nonzero(hit))
    volume[hit] = to_label
    return n


def _region_legend(region_labels):
    """The label -> region-name mapping, shown in the side panel so the
    brush number you're about to paint with is never a guess. Only used when
    there is no atlas configured -- with one, the assignment panel built by
    _add_ontology_picker replaces this and is editable."""
    if not region_labels:
        return ("No region_labels in the config: paint one region with label 1.\n"
                "For several regions, add region_labels to the config, or configure\n"
                "an atlas (atlas_annotation_path) to pick them from the tree here --\n"
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


def _add_ontology_picker(viewer, atlas, paint_layer, assignment, resolution_um=None, ortho=True):
    """The ontology tree + label-assignment panel, as a dock on the SAMPLE
    viewer's right side.

    Selecting any node highlights that structure and everything under it (see
    highlight_mask) in the separate atlas window, in all three of its synced
    ortho panes at once -- the only way to see a high-level region at all,
    since the annotation's own labels sit at ontology depths 2-12 and a
    depth-3 node owns no voxels under its own id.

    Everything needed to paint and export lives HERE, in the sample's window:
    the atlas window is a disposable viewer that can be closed and reopened at
    will (see _open_atlas_window) without touching the assignment.

    `assignment` ({label: [structure id]}) is mutated in place -- it is the
    live state the export reads, so there is no separate "apply" step to
    forget.
    """
    # {"w": SimpleNamespace | None}. napari clears a closed viewer's layers
    # but leaves the Python object alive, so the Qt window's own destroyed
    # signal is what actually tells us the user shut it.
    atlas_window = {"w": None}

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
    # A floor, not a target: the tree is what absorbs spare vertical space, so
    # it fills the dock on a tall window regardless. Keep the floor low enough
    # that the two right-hand docks' minimum heights, which napari stacks and
    # therefore ADDS, still leave the window shorter than the screen -- see
    # _scrollable for what happens when they don't.
    tree.setMinimumHeight(160)
    items = populate_ontology_tree(tree, atlas.structures, atlas.node_voxels)

    label_spin = QSpinBox()
    label_spin.setObjectName("ontology_brush_label")
    label_spin.setRange(1, MAX_LABEL)
    add_btn = QPushButton("加到 label")
    add_btn.setObjectName("ontology_assign")
    remove_btn = QPushButton("从 label 移除")
    remove_btn.setObjectName("ontology_unassign")
    jump_btn = QPushButton("三视角跳到该脑区中心")
    jump_btn.setObjectName("ontology_jump")
    window_btn = QPushButton("打开图谱窗口")
    window_btn.setObjectName("ontology_window")
    # Both grow with use -- picker_status with the selected region's blurb,
    # assign_label with one line per assigned brush label -- so both go in
    # scroll areas for the reason _scrollable spells out.
    assign_label = QLabel()
    picker_status = QLabel()

    def selected_id():
        item = tree.currentItem()
        return None if item is None else item.data(0, Qt.UserRole)

    def window_open():
        return atlas_window["w"] is not None

    def ensure_window():
        """The atlas window, opening one if it isn't up."""
        if not window_open():
            win = _open_atlas_window(atlas, resolution_um, ortho=ortho)
            win.viewer.window._qt_window.destroyed.connect(_on_window_closed)
            atlas_window["w"] = win
            window_btn.setText("关闭图谱窗口")
        return atlas_window["w"]

    def _on_window_closed(*_args):
        atlas_window["w"] = None
        window_btn.setText("打开图谱窗口")

    def on_toggle_window():
        if window_open():
            atlas_window["w"].viewer.close()   # destroyed signal resets the state
            _on_window_closed()
        else:
            ensure_window()
            on_select()                        # show whatever is selected right now

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
        if voxels:
            picker_status.setText(f"{info['name']} [{sid}]，含后代 {voxels:,} 体素。"
                                  + ("" if window_open() else " 打开图谱窗口可以看到它。"))
        else:
            picker_status.setText(
                f"{info['name']} [{sid}] 在这份 annotation 里没有任何体素，不能分配 —— "
                f"流水线遇到匹配不到的区域会直接报错。")
        # Only compute the mask when there is somewhere to show it: it is a
        # full-volume np.isin over the annotation, too expensive to run on
        # every arrow-key move through the tree with no window open.
        if window_open():
            atlas_window["w"].set_highlight(highlight_mask(atlas, sid),
                                            name=f"selection: {info['name']}")

    def on_jump():
        sid = selected_id()
        if sid is None or not atlas.node_voxels.get(sid):
            return
        win = ensure_window()
        if win.highlight.data.max() == 0:
            on_select()
        # All three axes, not just the slider axis: with the ortho panes up,
        # jumping only pane 0 leaves the other two showing a plane the region
        # may not even reach.
        centre = mask_centre_index(win.highlight.data)
        if centre is not None:
            win.centre_on(centre)

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
    window_btn.clicked.connect(on_toggle_window)

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(QLabel("图谱 ontology（选中即在图谱窗口三个视角同时高亮，含全部后代）"))
    layout.addWidget(search)
    layout.addWidget(hide_empty)
    layout.addWidget(tree)
    layout.addWidget(_scrollable(picker_status, 56))
    layout.addWidget(window_btn)
    layout.addWidget(jump_btn)
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.addWidget(QLabel("brush label"))
    row_layout.addWidget(label_spin)
    row_layout.addWidget(add_btn)
    row_layout.addWidget(remove_btn)
    layout.addWidget(row)
    layout.addWidget(_scrollable(assign_label, 80))
    viewer.window.add_dock_widget(dock, area="right", name="Atlas / Ontology")

    refresh_filter()
    refresh_assignment()
    ensure_window()          # open it once at startup; closing it is then up to you
    return SimpleNamespace(refresh_assignment=refresh_assignment,
                           atlas_window=atlas_window, ensure_window=ensure_window)


def _run_guide(args):
    _import_gui()
    base_sitk, arr = _read_sitk_array(args.image_path)

    prefill = np.zeros(arr.shape, dtype=np.uint8)
    resume = load_guide_resume(args.existing_mask, arr.shape) if args.existing_mask else None
    if resume is not None:
        prefill = resume.prefill
        planes = sum(len(p) for p in resume.slices_by_label.values())
        print(f"[resume] 从 {resume.sidecar.name} 恢复了 {planes} 个手画层"
              f"（{ {lab: p for lab, p in sorted(resume.slices_by_label.items())} }），"
              f"插值出来的层已丢弃，继续画即可。")
    elif args.existing_mask:
        # No sidecar: all this can do is binarize, which merges every region
        # into label 1 and treats interpolated planes as hand-drawn. Usable as
        # a rough tracing backdrop, not as a resume.
        loaded = _load_mask_array(args.existing_mask, arr.shape)
        if loaded is not None:
            prefill = loaded
            print(f"WARNING: {args.existing_mask} 旁边没有 .regions.json，无法按关键帧续画。"
                  f"\n         已按二值预填：所有脑区被合并成 label 1，且插值出来的层会被当成"
                  f"手画层。\n         只适合当描图底稿；要真正续画，请用带 sidecar 的导出结果。")

    viewer, paint_layer = _launch_viewer(arr, prefill, scale=args.display_scale_zyx)

    # The atlas is a read-only reference on its own grid, deliberately NOT
    # registered to the sample -- the whole point of painting a guide is that
    # the two don't line up yet. It lives in its OWN window (see
    # _open_atlas_window for why sharing one was actively harmful), so this
    # viewer's z slider is driven by the sample alone. Nothing about the atlas
    # display reaches the export, which is ontology ids plus voxel indices on
    # the sample's own grid.
    atlas = assignment = picker = None
    if args.atlas:
        atlas = _load_atlas_reference(args.atlas)
        if args.display_scale_zyx is None:
            print("WARNING: 没设 display_scale_zyx，样本按各向同性绘制。原图 z 间距通常是面内的 "
                  "10 倍以上，正交视图会被压扁到没法用。\n         建议加上 display_scale_zyx: "
                  "[z, y, x]（原图微米数，例如 [32.0, 2.6, 2.6]）。只影响显示，不影响导出。")

        # A resumed file's own sidecar wins over the config: it records what
        # those label numbers actually meant last session, and painting more
        # planes under a label that silently changed region would be worse
        # than any config convenience.
        seed_labels = resume.region_labels if resume is not None else args.region_labels
        seed_ids = resume.region_ids if resume is not None else args.region_ids
        assignment, unresolved = _seed_assignment(seed_labels, seed_ids, atlas.structures)
        for label, name, n in unresolved:
            print(f"WARNING: region_labels label {label} 的 {name!r} 在 ontology 里"
                  f"{'匹配到多个' if n else '匹配不到'}结构（{n} 个），没有转成 id；"
                  f"请在树里重新选一次。")
        picker = _add_ontology_picker(viewer, atlas, paint_layer, assignment,
                                       resolution_um=args.atlas.resolution_um,
                                       ortho=args.atlas.ortho)

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
                "orientation": list(args.atlas.orientation) if args.atlas.orientation else None,
                "resolution_um": args.atlas.resolution_um,
            }
        else:
            region_ids, region_labels, atlas_info = dict(args.region_ids), args.region_labels, None

        out_sitk = sitk.GetImageFromArray(result.volume)
        out_sitk.CopyInformation(base_sitk)      # keeps the source's (1,1,1) -- see module docstring
        sitk.WriteImage(out_sitk, args.output_path)
        regions_path, slices_path = write_guide_sidecars(
            args.output_path, args.image_path, result, region_labels,
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
                      guide_regions_yaml_snippet(
                          region_ids, region_labels, args.output_path,
                          voxel_size_um=voxel_size_um_from_display_scale(args.display_scale_zyx))]

        msg = "\n".join(lines)
        status_label.setText(msg)
        print(msg)

    _make_export_dock(viewer, status_label, export, "Export Outline", "Guide Outline Export")

    # A bulk relabel has to carry the region assignment with it, or the label
    # keeps its voxels and loses its meaning -- the exact thing the ontology
    # picker exists to prevent. Merging onto a label that already has a region
    # keeps the destination's, since that is the one the user just pointed at.
    def _assignment_follows_relabel(src, dst):
        if assignment is None:
            return
        ids = assignment.pop(src, None)
        if ids and dst not in assignment:
            assignment[dst] = ids
        if picker is not None:
            picker.refresh_assignment()

    _add_relabel_panel(viewer, paint_layer, on_change=_assignment_follows_relabel)


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


def selftest_relabel_volume(interp):
    print("9. relabel: renumber one label, merge two, reject out-of-range")
    canvas = _canvas()
    _box(canvas, [1, 4], 1, 5, 12, 5, 12)
    _box(canvas, [1, 4], 2, 20, 27, 20, 27)
    before_1 = int((canvas == 1).sum())
    before_2 = int((canvas == 2).sum())

    moved = canvas.copy()
    n = relabel_volume(moved, 1, 3)
    assert n == before_1, (n, before_1)
    assert (moved == 1).sum() == 0 and (moved == 3).sum() == before_1
    assert (moved == 2).sum() == before_2, "an unrelated label was touched"

    merged = canvas.copy()
    relabel_volume(merged, 1, 2)
    assert (merged == 2).sum() == before_1 + before_2, "merge onto an existing label lost voxels"

    assert relabel_volume(canvas.copy(), 99, 5) == 0, "a label nobody painted should be a no-op"
    try:
        relabel_volume(canvas.copy(), 1, MAX_LABEL + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range to_label was accepted (the export is uint8)")
    print(f"   ok ({before_1} voxels moved, merge and no-op behave)")


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
        out_path, "/data/s12t/registration.tif", result, region_labels,
        SHAPE[0], spacing_xyz=(1.0, 1.0, 1.0), region_ids=region_ids,
        atlas_info={"annotation_path": "/atlas/P04_annotations.nii.gz",
                    "ontology_path": "/atlas/DevCCFv1_ontology.json"})

    assert regions_path.name == "s12t_guide_sample.regions.json", regions_path
    assert slices_path.name == "s12t_guide_sample.annotated_slices.json", slices_path

    regions = json.loads(regions_path.read_text())
    assert regions["regions"] == {"1": ["layer 1 of A", "layer 2 of A"],
                                  "2": ["cerebellar hemisphere"]}, regions["regions"]
    assert regions["region_ids"] == {"1": [15751, 15756], "2": [15623]}, regions["region_ids"]
    assert regions["annotated_slices"] == {"1": [0, 4, 8], "2": [2, 6]}, regions["annotated_slices"]
    assert regions["image_path"] == "/data/s12t/registration.tif"
    assert regions["atlas"]["ontology_path"].endswith("DevCCFv1_ontology.json"), regions["atlas"]
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


def selftest_resume_restores_only_hand_drawn_planes(interp, tmp_dir):
    print("8. resume: export -> reload -> add planes, without inheriting the interpolation")
    canvas = _canvas()
    _box(canvas, [0, 4, 8], 1, 2, 10, 2, 10)        # label 1: 3 real keyframes
    _box(canvas, [2, 6], 2, 20, 30, 4, 12)          # label 2: 2 real keyframes
    result = interpolate_labels_separately(sparse_keyframes_by_label(canvas), SHAPE,
                                           interpolate=interp)

    out_path = tmp_dir / "resume_guide.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(result.volume), str(out_path))
    write_guide_sidecars(out_path, "/data/registration.tif", result,
                         {1: ["cortex"], 2: ["cerebellar hemisphere"]}, SHAPE[0],
                         region_ids={1: [15751], 2: [15623]})

    resumed = load_guide_resume(out_path, SHAPE)
    assert resumed is not None

    # The whole point: exactly the planes that were hand-drawn come back, NOT
    # the dense interpolation between them. Reloading the volume directly
    # would give label 1 planes 0..8 and label 2 planes 2..6.
    assert resumed.slices_by_label == {1: [0, 4, 8], 2: [2, 6]}, resumed.slices_by_label
    assert sparse_keyframes_by_label(resumed.prefill).keys() == {1, 2}
    assert {lab: sorted(planes) for lab, planes in
            sparse_keyframes_by_label(resumed.prefill).items()} == {1: [0, 4, 8], 2: [2, 6]}

    # Labels stay distinct (the old `> 0` path merged them into one).
    assert set(np.unique(resumed.prefill).tolist()) == {0, 1, 2}, np.unique(resumed.prefill)
    # And the label -> region mapping survives, so label numbers keep meaning.
    assert resumed.region_ids == {1: [15751], 2: [15623]}, resumed.region_ids
    assert resumed.region_labels == {1: ["cortex"], 2: ["cerebellar hemisphere"]}

    # Re-exporting the untouched resume must reproduce the original file
    # byte-for-byte -- otherwise every save/reload cycle would drift.
    again = interpolate_labels_separately(sparse_keyframes_by_label(resumed.prefill), SHAPE,
                                          interpolate=interp)
    assert np.array_equal(again.volume, result.volume), \
        f"{int(np.sum(again.volume != result.volume))} voxels drifted across a resume cycle"

    # Adding a keyframe extends that label's span and leaves the other alone.
    grown = resumed.prefill.copy()
    _box(grown, [12], 1, 2, 10, 2, 10)
    third = interpolate_labels_separately(sparse_keyframes_by_label(grown), SHAPE,
                                          interpolate=interp)
    assert third.slices_by_label == {1: [0, 4, 8, 12], 2: [2, 6]}, third.slices_by_label
    assert third.voxels_by_label[1] == 64 * 13, third.voxels_by_label
    assert third.voxels_by_label[2] == result.voxels_by_label[2], third.voxels_by_label

    # No sidecar next to the file -> refuse to pretend it's resumable.
    plain = tmp_dir / "no_sidecar.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(result.volume), str(plain))
    assert load_guide_resume(plain, SHAPE) is None
    print("   ok")


def selftest_config_normalizers():
    print("10. config: region_labels int/str keys + multi-region values, display_scale_zyx")
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

    # (z,y,x) display scale -> (x,y,z) pipeline voxel size. Reversed, not
    # copied: the two configs use opposite axis orders for the same voxel.
    assert voxel_size_um_from_display_scale([32.0, 2.6, 2.6]) == [2.6, 2.6, 32.0]
    assert voxel_size_um_from_display_scale(None) is None

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
    print("12. ontology: voxels roll up to every ancestor; empty subtrees stay empty")
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
    print("13. ontology: search reveals ancestors, matches acronyms, respects hide-empty")
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
    print("14. config -> GUI assignment: ids kept, names resolved only when unambiguous")
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
    print("11. local reference interpolator == registration_ants.mask_utils (when available)")
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
    print("15. atlas: chunked relabelling is exact, and small enough to be worth it")
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


def selftest_atlas_view_geometry():
    print("16. atlas 三视角: 每个轴各被切一次，跳转中心落在脑区里")
    # Every axis is the slider axis in exactly one pane, and is drawn in the
    # other two -- i.e. the three panes really are the three orthogonal views
    # and not two copies of one.
    orders = [order for order, _title in _ORTHO_PANES]
    assert len(orders) == 3, orders
    sliced = [order[0] for order in orders]
    assert sorted(sliced) == [0, 1, 2], sliced
    for order in orders:
        assert sorted(order) == [0, 1, 2], order

    # A C-shaped structure: the centre of MASS of this mask sits at z=2 in the
    # hollow, which is exactly the plane where nothing would be visible.
    mask = np.zeros((5, 9, 7), dtype=np.uint8)
    mask[0, 1:8, 1:6] = 1
    mask[4, 1:8, 1:6] = 1
    centre = mask_centre_index(mask)
    assert mask[centre] == 1, (centre, "跳转落在了脑区外面")
    assert centre == (4, 4, 3), centre

    solid = np.zeros((5, 9, 7), dtype=np.uint8)
    solid[1:4, 2:5, 3:6] = 1
    assert mask_centre_index(solid) == (2, 3, 4), mask_centre_index(solid)
    assert mask_centre_index(np.zeros((3, 3, 3), dtype=np.uint8)) is None
    print("   ok")


def selftest_annotation_features():
    print("17. atlas 图层: compact 索引 -> 真实脑区名/id 的悬停表")
    atlas = SimpleNamespace(
        present_ids=np.array([0, 5, 7, 999], dtype=np.int64),
        structures={5: {"name": "Cortex", "acronym": "CTX"}, 7: {"name": "Thalamus"}})
    feats = annotation_features(atlas)
    # napari matches a features row to the value under the cursor through the
    # 'index' column, so this has to be the COMPACT index, not the real id.
    assert list(feats["index"]) == [0, 1, 2, 3], feats["index"]
    assert feats["id"] == [0, 5, 7, 999], feats["id"]
    assert feats["脑区"][1] == "Cortex (CTX)", feats["脑区"]
    assert feats["脑区"][2] == "Thalamus", feats["脑区"]
    assert "999" in feats["脑区"][3], feats["脑区"]     # in the annotation, not in the ontology
    print("   ok")


def selftest_crosshair_vectors():
    print("18. atlas 十字准线: 每个面正好画到两条线，且交点就是当前切片位置")
    shape = (6, 8, 10)
    vectors = crosshair_vectors((4, 5, 7), shape)
    assert vectors.shape == (3, 2, 3), vectors.shape

    # Each line starts at 0 on its own axis and at the crosshair on the others,
    # and is exactly as long as the volume, so it spans the whole pane.
    assert np.array_equal(vectors[:, 0], [[0, 5, 7], [4, 0, 7], [4, 5, 0]]), vectors[:, 0]
    assert np.array_equal(vectors[:, 1], np.diag(shape)), vectors[:, 1]

    # A pane slicing axis k shows a line iff that line's start sits in the
    # current slice -- true for the two lines lying in the plane, false for the
    # one pointing at the viewer. Two lines per pane is the whole trick.
    for sliced_axis in range(3):
        in_slice = [i for i in range(3) if vectors[i, 0, sliced_axis] == (4, 5, 7)[sliced_axis]]
        assert len(in_slice) == 2, (sliced_axis, in_slice)
        assert sliced_axis not in in_slice, (sliced_axis, in_slice)
    print("   ok")


def selftest_format_ancestry():
    print("19. atlas 悬停: 显示的是整条祖先链，不是只有最细那一级")
    structures = {
        1: {"name": "root", "acronym": "root", "structure_id_path": [1]},
        2: {"name": "Cerebrum", "acronym": "CH", "structure_id_path": [1, 2]},
        5: {"name": "Cortex", "acronym": "CTX", "structure_id_path": [1, 2, 5]},
        9: {"name": "layer 5", "acronym": None, "structure_id_path": [1, 2, 5, 9]},
    }
    text = format_ancestry(structures, 9)
    lines = text.splitlines()
    assert len(lines) == 4, lines
    # Root first, leaf last, every ancestor present with its id -- the ids are
    # what the export records, so they have to be readable off this panel.
    for sid in (1, 2, 5, 9):
        assert f"[{sid}]" in text, (sid, text)
    assert lines[0].startswith("· root"), lines[0]
    assert "▶" in lines[-1] and lines[-1].count("▶") == 1, lines[-1]
    assert sum("▶" in line for line in lines) == 1, lines
    assert "(CTX)" in lines[2], lines[2]      # acronym shown when the node has one
    assert "(None)" not in lines[3], lines[3] # and not faked when it does not

    # A mid-level pick marks itself, not the deepest node it knows about.
    assert "▶ Cortex" in format_ancestry(structures, 5)
    # A label the annotation carries but the ontology never describes.
    assert "不在 ontology" in format_ancestry(structures, 4242)
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
        selftest_resume_restores_only_hand_drawn_planes(interp, Path(tmp))
    selftest_relabel_volume(interp)
    selftest_config_normalizers()
    selftest_interpolator_matches_registration_ants()
    selftest_ontology_node_voxels()
    selftest_ontology_tree_filter()
    selftest_seed_assignment()
    selftest_compact_annotation()
    selftest_atlas_view_geometry()
    selftest_annotation_features()
    selftest_crosshair_vectors()
    selftest_format_ancestry()
    print("=== all selftests passed ===")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Paint a guide outline on a sample volume")
    _local_config.add_config_arg(parser, "paint_mask")
    parser.add_argument("--selftest", action="store_true",
                        help="run the built-in synthetic tests (no GUI, no config) and exit")
    args_cli = parser.parse_args()

    if args_cli.selftest:
        return run_selftests()

    cfg = _load_local_config(args_cli.config)
    _run_guide(SimpleNamespace(
        image_path=cfg.image_path, output_path=cfg.output_path,
        existing_mask=cfg.existing_mask_path,
        region_labels=cfg.region_labels, region_ids=cfg.region_ids,
        display_scale_zyx=cfg.display_scale_zyx, atlas=cfg.atlas))

    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
