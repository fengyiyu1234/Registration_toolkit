"""Interactive tool: paint a guide outline on a 3D sample volume.

TWO MODES, chosen by `mode:` in configs/paint_mask.yaml. Both export a guide
for mask.guide_regions; they differ in what you start from. The config is
`mode:` plus a `common:` section and one section per mode -- only the running
mode's section is read (flatten_config_sections), so both can stay filled in
and switching modes is a one-line edit.

  mode: guide (default) -- paint on the raw sample, from blank planes.
    You trace each region by hand and say which atlas structure(s) each brush
    number stands for, in the ontology tree. Use it when there is no
    registration yet, or when the one you have is too far off to correct.
    Everything below this header describes this mode.

  mode: labels -- paint on a registration RESULT, and correct it.
    Starts from <name>_labels_in_sample.nii.gz collapsed into a partition of
    brush labels, so the whole brain is already outlined and you only fix
    what came out wrong, on a handful of planes. Exports two volumes: a
    sparse guide to re-register with, and a dense one to re-open and carry
    on from. The partition is refined per region rather than at a fixed
    ontology depth -- expand Hippocampal formation into CA/DG without
    touching how coarsely the cerebellum is described. See the
    "mode: labels" section further down, and shared/label_partition.py for
    the measured reason a uniform ontology depth is not a usable knob.

    THE CANVAS IS STILL THE RAW STACK. The registration output arrives on
    the isotropic grid registration ran on (the pipeline's fine_target_um --
    20 um for both atlas presets here, 25 if it is left unset), and is
    regridded up onto image_path's grid to be overlaid -- never the other
    way round. Painting on the isotropic grid would mean drawing at 20 um on
    planes that were interpolated into existence, instead of at 2.6 um on
    the planes that were actually imaged; pipeline.py's _build_guide_regions_from_labels
    says the same thing about where a painted volume has to live. So
    image_path, the exported grid, and the voxel_size_um that goes into the
    pipeline config are identical to guide mode's.

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
    atlas and the tool grows an ontology tree in its own dedicated panel
    (see "A large, dedicated region panel" below). Selecting any node
    assigns that structure AND all its descendants to a brush label -- the
    only way a high-level node means anything, since the annotation's own
    labels sit at ontology depths 2-12 and a depth-3 node owns no voxels
    under its own id. The export records the ontology IDS (see
    write_guide_sidecars for why ids rather than names).

    This panel only picks and assigns -- it draws nothing. To actually SEE
    the atlas (three synced ortho panes, a region highlighted among its
    neighbours, hover-to-read-the-full-ontology-chain), run the separate
    tools/atlas_view.py against the same atlas_annotation_path / ontology_path.
    The two tools used to be one window pair, with the atlas side driven
    live from this tree; they are independent scripts now, so nothing
    painted or assigned here reaches tools/atlas_view.py and nothing selected
    there reaches this tool.

    The atlas grid is INDEPENDENT of the sample's: a half-brain sample
    against a whole-brain atlas is the normal case. The atlas is never
    registered to the sample -- what is exported is ids plus voxel indices
    on the sample's own grid, so nothing about the atlas (its extent,
    orientation, downsampling) can reach the output. The one thing that
    must match the pipeline's atlas is the ONTOLOGY, since ids are what
    crosses over.

  A large, dedicated region panel: the ontology tree lives on its own side
    of the window (left) rather than sharing a column with Relabel/Export
    (right), because it is 2-12 levels deep and a tree squeezed into a
    fraction of a shared dock leaves most of it scrolled out of view. Under
    it, behind a draggable splitter, is what has been assigned so far --
    brush label -> its regions, as a tree whose rows are the handles: select
    any single region there and Remove takes just that one off the label,
    without hunting it down in the ontology again. A label whose last region
    is removed does not disappear, it stays listed as empty with a warning,
    because something is probably still painted with that number.

  Everything is TABBED, one tab bar per side (see _tab_the_panels): five
    panels stacked down a column arrive as five slivers, none of them usable
    without dragging the others shut first. Drag a tab out of the bar if two
    panels really are needed at once.

  A "Relabel" panel: click-to-fill a single already-painted blob into
    another label, or renumber one label across the whole volume. A bulk
    renumber carries that label's ontology assignment with it, so the number
    keeps meaning the same region.

  An "Erase" panel: lasso a polygon around a mistake and everything inside
    it is erased on that plane, whatever label it carried -- rubbing a whole
    wrong blob out with the eraser brush is the slow way to do the same
    thing. The brush/eraser size slider is also widened past napari's own
    1..40 ceiling (see MAX_BRUSH_SIZE), since 40 voxels is a dot on a plane
    several hundred voxels across.

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
     the config's `voxel_size_um: [x, y, z]` is the ONE place it is stated:
     napari's display aspect is that triple reversed, the pasteable pipeline
     snippet quotes it verbatim, and `mode: labels` regrids with it. It
     never changes the exported values (those are voxel indices).

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
configs/paint_mask.example.yaml the first time, which has every key filled in
under common:/guide:/labels: rather than commented out), then just run the
file -- no command-line arguments.

    conda activate antsreg
    python paint_mask.py
    python paint_mask.py configs/paint_mask.guide_s12t.yaml  # or point at another config

To actually look at the atlas (not just assign regions to brush labels),
run the separate tools/atlas_view.py -- see its own docstring.

The export logic is separately runnable with no display and no config, on
purely synthetic data:

    python paint_mask.py --selftest

shared/atlas_reference.py --selftest and tools/atlas_view.py --selftest cover
the atlas loading / ontology math and the ortho-view geometry that used to be
tested here.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk
import yaml

from shared import atlas_reference   # GUI-free atlas loading + ontology math
from shared import local_config      # configs/<tool>.yaml
from shared import label_partition   # brush-label <-> ontology-region partitions
from shared import ontology_tree_ui  # the shared Qt ontology tree widget

# napari/PyQt5 are imported lazily by _import_gui() rather than here, and
# mask_utils by _interpolate_sparse_mask(), so that --selftest (pure numpy +
# scipy, no window) runs with no display and without the ../Registration_ants
# editable install. Both are hard requirements for the actual painting GUI,
# which only ever runs in antsreg.
napari = QLabel = QPushButton = QVBoxLayout = QWidget = None
QCheckBox = QHBoxLayout = QLineEdit = QSpinBox = QListWidget = None
QTreeWidget = QTreeWidgetItem = QSplitter = Qt = None


def _import_gui():
    """Bind the napari/Qt names used by the viewer code. Called once at the
    top of the GUI entry points; import errors surface there rather than at
    module import, which is what keeps --selftest env-independent."""
    global napari, QLabel, QPushButton, QVBoxLayout, QWidget
    global QCheckBox, QHBoxLayout, QLineEdit, QSpinBox, QListWidget
    global QTreeWidget, QTreeWidgetItem, QSplitter, Qt
    import napari as _napari
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtWidgets import (QCheckBox as _QCheckBox, QHBoxLayout as _QHBoxLayout,
                                 QLabel as _QLabel, QLineEdit as _QLineEdit,
                                 QListWidget as _QListWidget,
                                 QPushButton as _QPushButton,
                                 QSpinBox as _QSpinBox,
                                 QSplitter as _QSplitter,
                                 QTreeWidget as _QTreeWidget,
                                 QTreeWidgetItem as _QTreeWidgetItem,
                                 QVBoxLayout as _QVBoxLayout, QWidget as _QWidget)
    napari, QLabel, QPushButton = _napari, _QLabel, _QPushButton
    QVBoxLayout, QWidget = _QVBoxLayout, _QWidget
    QCheckBox, QHBoxLayout, QLineEdit, QSpinBox = _QCheckBox, _QHBoxLayout, _QLineEdit, _QSpinBox
    QListWidget = _QListWidget
    QTreeWidget, QTreeWidgetItem, QSplitter, Qt = _QTreeWidget, _QTreeWidgetItem, _QSplitter, _Qt


def _interpolate_sparse_mask():
    """registration_ants.mask_utils.interpolate_sparse_mask -- resolved
    through the editable install of ../Registration_ants (pip install -e
    there puts it on sys.path for the antsreg env), no path hacking needed.
    mask_utils is pure numpy/scipy, so this import does NOT drag in
    antspyx."""
    from registration_ants import mask_utils
    return mask_utils.interpolate_sparse_mask


def _interpolate_sparse_label_correction():
    """registration_ants.mask_utils.interpolate_sparse_label_correction --
    the multi-label sibling of the above, used by `mode: labels`. Same lazy
    import for the same reason (--selftest must not need the editable
    install)."""
    from registration_ants import mask_utils
    return mask_utils.interpolate_sparse_label_correction

# This config used to live in the repo root; it now sits in configs/ like every
# other tool's. The old location is still read, with a migration note printed.
_LEGACY_CONFIG_PATHS = (Path(__file__).resolve().parent / "paint_mask_local.yaml",)


MODES = ("guide", "labels")


def flatten_config_sections(cfg, mode):
    """`common:` plus the running mode's own section, flattened into one dict.

    The config file is written as mode + common/guide/labels sections because a
    single flat list of every key gave no way to see which of them the mode you
    are about to run actually reads -- the mode-specific ones had to be left
    commented out to stay out of the way, which is not a state a config file
    should have to be in. Sections make "filled in but not used right now" the
    normal state: the INACTIVE mode's section is dropped rather than merged, so
    labels_path can sit there with a real path while mode: guide runs, and
    switching modes is a one-line edit.

    Top-level keys are still read (that is what every config looked like before
    the sections existed, and a one-mode config needs no ceremony); common:
    overrides them and the mode section overrides both, so the more specific
    place always wins.
    """
    flat = {k: v for k, v in cfg.items() if k not in ("mode", "common") + MODES}
    for name in ("common", mode):
        block = cfg.get(name)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(f"`{name}:` in the config should be a key: value mapping, "
                             f"got {type(block).__name__}")
        flat.update(block)
    return flat


def _load_local_config(cli_path=None):
    """Paths live in a gitignored configs/paint_mask.yaml instead of constants
    here, so editing them for a new sample never shows up as a git diff."""
    raw = local_config.load_config(
        "paint_mask", cli_path=cli_path, legacy_paths=_LEGACY_CONFIG_PATHS)
    mode = (raw.get("mode") or "guide").strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode must be 'guide' or 'labels', got {mode!r}")
    # After flattening, not before: image_path/output_path normally live under
    # `common:` now, and load_config's own required= check only sees the top
    # level. Same message it would have printed.
    cfg = flatten_config_sections(raw, mode)
    missing = [k for k in ("image_path", "output_path") if not cfg.get(k)]
    if missing:
        raise ValueError(
            f"config is missing: {', '.join(missing)} (looked at the top level, in "
            f"`common:` and in `{mode}:`)\n"
            f"(what each key means: {local_config.example_path('paint_mask')})")
    return SimpleNamespace(
        mode=mode,
        image_path=cfg["image_path"],
        output_path=cfg["output_path"],
        existing_mask_path=cfg.get("existing_mask_path") or None,
        region_labels=_normalize_region_labels(cfg.get("region_labels") or {}),
        region_ids=_normalize_region_ids(cfg.get("region_ids") or {}),
        atlas=atlas_reference.atlas_reference_config(cfg),
        # mode: labels only -- see the "painting on a registration result"
        # section of the module docstring.
        labels_path=cfg.get("labels_path") or None,
        atlas_output_path=cfg.get("atlas_output_path") or None,
        partition_path=cfg.get("partition_path") or None,
        min_region_mm3=float(cfg.get("min_region_mm3") or label_partition.DEFAULT_MIN_MM3),
        # (x,y,z) um for image_path. Optional in mode: guide (only the
        # display aspect and the pasteable snippet want it), REQUIRED in
        # mode: labels, where the registration output has to be regridded
        # onto the raw stack before it can be overlaid on it and the raw
        # stack's header does not carry a voxel size (module docstring).
        voxel_size_um=_config_voxel_size_um(cfg),
        labels_voxel_size_um=(list(cfg["labels_voxel_size_um"])
                              if cfg.get("labels_voxel_size_um") else None),
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
        raise ValueError(f"{key_name} should be a {{label: {describe}}} mapping, "
                         f"got {type(raw).__name__}")

    normalized = {}
    for key, entries in raw.items():
        try:
            label = int(key)
        except (TypeError, ValueError):
            raise ValueError(
                f"{key_name} keys must be brush labels (integers), got {key!r}") from None
        if label < 1:
            raise ValueError(f"{key_name} labels must be >= 1 (0 is background/eraser), "
                             f"got {label}")
        if label in normalized:
            raise ValueError(f"{key_name} lists label {label} twice "
                             "(once as an int key and once as a str key?)")
        if isinstance(entries, (str, int)):
            entries = [entries]
        try:
            entries = [coerce(e) for e in entries]
        except (TypeError, ValueError):
            raise ValueError(f"{key_name} label {label} should map to {describe}, "
                             f"got {entries!r}") from None
        if not entries:
            raise ValueError(f"{key_name} label {label} has an empty {describe}")
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
    return _normalize_label_map(raw, "region_labels", _region_name, "brain region name")


def _normalize_region_ids(raw):
    """{brush label -> [ontology structure id, ...]}. Normally written by the
    GUI's ontology picker rather than by hand; ids beat names downstream
    because they are matched exactly instead of as substrings."""
    return _normalize_label_map(raw, "region_ids", int, "ontology structure id (integer)")


def _normalize_voxel_size_um(raw, key="voxel_size_um"):
    """Optional (x, y, z) micron voxel size -- None if unset.

    (x,y,z) like every other micron triple in the pipeline configs, and the
    REVERSE of the (z,y,x) axis order SimpleITK hands the arrays back in;
    display_scale_from_voxel_size does that one reversal, so nobody has to
    keep two spellings of the same voxel in the config in sync.
    """
    if raw is None or raw == "":
        return None
    try:
        size = [float(v) for v in raw]
    except (TypeError, ValueError):
        raise ValueError(f"{key} should be three numbers [x, y, z], got {raw!r}") from None
    if len(size) != 3:
        raise ValueError(f"{key} needs exactly 3 numbers [x, y, z], got {len(size)}")
    if any(v <= 0 for v in size):
        raise ValueError(f"{key} must be all positive numbers, got {size}")
    return size


def _config_voxel_size_um(cfg):
    """voxel_size_um from the config, accepting the retired display_scale_zyx.

    display_scale_zyx was a second spelling of the same physical voxel in the
    opposite axis order, so a config could carry both and have them disagree.
    Only voxel_size_um survives; an old config's display_scale_zyx is still
    read (reversed) with a note, and having both is an error rather than a
    silent pick, because which one won would decide whether mode: labels
    regrids against 2.6 um or 32 um planes.
    """
    voxel = _normalize_voxel_size_um(cfg.get("voxel_size_um"))
    legacy = _normalize_voxel_size_um(
        list(reversed(cfg["display_scale_zyx"]))
        if cfg.get("display_scale_zyx") else None, key="display_scale_zyx (reversed)")
    if legacy and voxel and legacy != voxel:
        raise ValueError(
            f"config has both voxel_size_um {voxel} (x,y,z) and display_scale_zyx "
            f"{list(reversed(legacy))} (z,y,x), and they describe different voxels. "
            f"display_scale_zyx is retired -- delete it and keep voxel_size_um.")
    if legacy and not voxel:
        print(f"NOTE: display_scale_zyx is retired; using it as voxel_size_um: {legacy} "
              f"(x,y,z). Rename it in the config -- the display scale is now derived "
              f"from voxel_size_um.")
        return legacy
    return voxel


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


# napari 0.8 builds its brush-size slider with a hardcoded 1..40 range
# (_qt/layer_controls/widgets/_labels/qt_brush_size_slider.py). A guide
# outline is painted on planes several hundred voxels across, so 40 is a
# small dot -- both to fill a region and, mostly, to rub one out again.
# Starting width of the region panels, in px. A STARTING width, not a cap:
# the panels are ontology_tree_ui.shrinkable, so both edges stay draggable --
# a 12-deep tree of region names needs whatever width the names need, and
# that is not something a constant can know.
_ONTOLOGY_PANEL_START_PX = 380

MAX_BRUSH_SIZE = 100


def _widen_brush_size_slider(paint_layer, maximum=MAX_BRUSH_SIZE):
    """Raise the ceiling of napari's brush/eraser size slider to `maximum`.

    Two paths, because each on its own has a hole:

      the layer side -- the slider widens its own maximum whenever the layer
        reports a brush_size above it (QtBrushSizeSliderControl.
        _on_brush_size_change) and never narrows it again, so pushing the
        value up and putting it straight back leaves a 1..maximum slider
        behind. That only reaches controls that already exist.

      the class side -- napari builds a fresh controls widget per layer, and
        a new one starts from the hardcoded 40 again, so the widget class
        itself is patched to widen on construction. That is a private module
        path, hence the guarded import: if it ever moves, the layer-side bump
        still covers the one layer this tool creates.
    """
    try:
        from napari._qt.layer_controls.widgets._labels.qt_brush_size_slider import (
            QtBrushSizeSliderControl)
    except ImportError:
        QtBrushSizeSliderControl = None

    if QtBrushSizeSliderControl is not None and not getattr(
            QtBrushSizeSliderControl, "_paint_mask_widened", False):
        original_init = QtBrushSizeSliderControl.__init__

        def _init(self, parent, layer, _original=original_init):
            _original(self, parent, layer)
            if self.brush_size_slider.maximum() < maximum:
                self.brush_size_slider.setMaximum(maximum)

        QtBrushSizeSliderControl.__init__ = _init
        QtBrushSizeSliderControl._paint_mask_widened = True

    previous = paint_layer.brush_size
    paint_layer.brush_size = maximum
    paint_layer.brush_size = previous


def _launch_viewer(arr, prefill, scale=None, title="Paint guide outline",
                   layer_name="guide outline (paint here)"):
    """The sample window: the grayscale volume plus the layer painted on.

    scale: optional (z, y, x) physical size per voxel, applied to BOTH layers
    so they stay registered to each other. Without it a raw 2.6/2.6/32 um
    stack is drawn as if it were isotropic, i.e. squashed 12x along z, which
    makes the orthogonal views unusable. Purely a display transform -- layer
    .data, and therefore the export, is untouched."""
    viewer = napari.Viewer(title=title)
    scale_kwargs = {"scale": scale} if scale is not None else {}
    viewer.add_image(arr, name="sample", colormap="gray", **scale_kwargs)
    paint_layer = viewer.add_labels(prefill.copy(), name=layer_name, **scale_kwargs)
    _widen_brush_size_slider(paint_layer)
    return viewer, paint_layer


def assignment_rows(assignment, structures):
    """The assignment panel's rows: [(label, [(sid, name), ...]), ...].

    Split out of the widget so the one thing worth checking -- that a label
    whose last region was removed is still REPORTED rather than silently
    vanishing -- is testable without a window. A label maps to an empty list
    exactly when it was assigned regions and they were all removed again;
    see empty_assignment_labels.
    """
    return [(label, [(sid, structures[sid]["name"]) for sid in assignment[label]])
            for label in sorted(assignment)]


def empty_assignment_labels(assignment):
    """Brush labels left with no region at all.

    These are kept in `assignment` rather than deleted on the way out,
    precisely so the panel can say so: a label that is still being painted
    with but has lost its region exports an outline nothing downstream can
    pair with an atlas structure (guide_export_warnings says the same thing
    at export time, which is far too late to be the first mention of it).
    Removing an already-empty label forgets it for good.
    """
    return sorted(label for label, ids in assignment.items() if not ids)


def display_scale_from_voxel_size(voxel_size_um):
    """voxel_size_um (x,y,z) -> the napari layer scale (z,y,x), or None when
    no voxel size was configured.

    The array axes and the pipeline's micron triples run in OPPOSITE orders
    -- sitk.GetArrayFromImage gives (z,y,x), while mask.guide_regions.
    voxel_size_um is (x,y,z) like every other micron triple here. Doing the
    reversal here rather than asking the config for both spellings is the
    point: a reversed voxel_size_um does not error, it just resamples the
    outline against the wrong physical size.
    """
    return list(reversed(voxel_size_um)) if voxel_size_um else None


def guide_regions_yaml_snippet(region_ids, region_names, output_path, voxel_size_um=None,
                               atlas_exclude_ids=None, voxel_size_note=None):
    """A ready-to-paste mask.guide_regions block for the pipeline config.

    Emitted on export because the ids are the whole point of picking regions
    in the GUI and retyping them by hand from a JSON sidecar is exactly where
    they would get corrupted. atlas_names is emitted alongside as a comment
    only: two sources of truth that can disagree is precisely the failure
    ids were chosen to remove, so the pipeline reads the ids and the names
    stay human-facing.
    """
    voxel = list(voxel_size_um) if voxel_size_um else ["?", "?", "?"]
    note = voxel_size_note or (
        "# source image (x,y,z) um, copied from the paint_mask config's voxel_size_um"
        if voxel_size_um else
        "# source image (x,y,z) um -- not in the tif header, fill it in by hand")
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
    if atlas_exclude_ids:
        # Only `mode: labels` emits this, because only a nested partition can
        # produce it -- see label_partition.Partition.atlas_exclude_ids for
        # why a residual parent's atlas outline has to have its split-out
        # children subtracted back out.
        lines.append("    # subtract each split-out child back out of its parent's atlas")
        lines.append("    # outline -- without this the same atlas voxels are pulled towards")
        lines.append("    # two different sample outlines at once.")
        lines.append("    atlas_exclude_ids:")
        for label in sorted(atlas_exclude_ids):
            names = ", ".join(region_names.get(label, []))
            lines.append(f"      {label}: {list(atlas_exclude_ids[label])}"
                         + (f"   # out of {names}" if names else ""))
    lines.append("    weight: 1.0")
    return "\n".join(lines)


def _tab_the_panels(viewer, left, right):
    """Fold every side panel into two TAB BARS -- one per side -- instead of
    stacking them down their columns.

    napari stacks docks vertically, and stacking is only usable while there
    are two of them. This tool adds four or five on the right and shares the
    left with napari's own layer list and layer controls, so opening the
    window used to mean being handed a pile of slivers, each of which had to
    be dragged open (at the cost of the ones above it) before it could be
    used. Tabbed, whichever panel is in front gets the whole column, and the
    rest are one click away -- and any of them can still be dragged out of
    the tab bar into its own dock, or floated, if two really are needed at
    once.

    The layer-controls height unclamp goes here too, because it is the same
    complaint: that dock stops shrinking while it still fills half the column
    (see ontology_tree_ui.free_layer_controls_height), which is precisely
    what makes a stacked left column unusable.

    Which tab starts in front: the panel this tool exists for, i.e. the
    region panel on the left and the export/status panel on the right, since
    that is where every message the tool prints ends up.
    """
    ontology_tree_ui.free_layer_controls_height(viewer)
    left = [dock for dock in left if dock is not None]
    right = [dock for dock in right if dock is not None]
    ontology_tree_ui.tabify(viewer, ontology_tree_ui.napari_layer_docks(viewer) + left,
                            current=left[0] if left else None)
    ontology_tree_ui.tabify(viewer, right, current=right[0] if right else None)


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
    layout.addWidget(ontology_tree_ui.scrollable(status_label, 120))
    layout.addWidget(export_btn)
    ontology_tree_ui.shrinkable(dock)
    return viewer.window.add_dock_widget(dock, area="right", name=panel_name)


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

    note = QLabel("Change the label of an already-painted region")
    status = QLabel("")
    status.setWordWrap(True)

    def start_fill():
        # selected_label is what FILL paints with, so set it from `to`.
        paint_layer.n_edit_dimensions = 1
        paint_layer.selected_label = int(to_spin.value())
        paint_layer.mode = "fill"
        status.setText(f"Fill mode: click any blob and it becomes label {int(to_spin.value())}. "
                       "(switch back to the paint brush before drawing again)")

    def relabel_all():
        src, dst = int(from_spin.value()), int(to_spin.value())
        if src == dst:
            status.setText("from and to are the same -- nothing to change.")
            return
        n = relabel_volume(paint_layer.data, src, dst)
        paint_layer.refresh()
        if on_change is not None:
            on_change(src, dst)
        status.setText(f"label {src} -> {dst}: {n} voxels changed."
                       + ("" if n else " (nothing was ever painted with that label)"))

    fill_btn = QPushButton("Click-to-fill one blob")
    fill_btn.clicked.connect(start_fill)
    all_btn = QPushButton("Relabel the whole label")
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
    layout.addWidget(ontology_tree_ui.scrollable(status, 60))
    ontology_tree_ui.shrinkable(dock)
    return viewer.window.add_dock_widget(dock, area="right", name="Relabel")


def _add_display_panel(viewer, layers):
    """A fill/outline switch for the region layers.

    napari's Labels layers are FILLED by default (contour = 0) and this tool
    has never changed that -- filled is what shows which region a blob
    actually is. The switch is here for the other half of the job: dropping
    to a 1-voxel contour uncovers the raw stack underneath, which is how you
    check whether a boundary sits where the tissue boundary sits. napari's
    own layer controls carry the same `contour` field per layer; this drives
    every region layer at once and puts it where it gets used.

    single_sample.py has the same checkbox, with the same default, so the two
    tools agree about what "showing a region" looks like.
    """
    layers = [layer for layer in layers if layer is not None]

    def on_toggled(checked):
        for layer in layers:
            layer.contour = 1 if checked else 0

    checkbox = QCheckBox("Region outline only (unchecked = filled)")
    checkbox.setChecked(False)
    checkbox.toggled.connect(on_toggled)

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(checkbox)
    layout.addWidget(QLabel(
        "Filled shows what each region IS; outline uncovers the image under it,\n"
        "for checking a boundary against the tissue."))
    ontology_tree_ui.shrinkable(dock)
    return viewer.window.add_dock_widget(dock, area="right", name="Display")


def _add_erase_panel(viewer, paint_layer):
    """An "erase what the brush is clumsy at" panel: lasso a polygon and
    everything inside it, on the plane you are looking at, is erased.

    napari's own eraser is the brush painting label 0 -- fine for nudging an
    edge, painful for taking out a whole wrong blob, which is exactly what a
    keyframe drawn on the wrong plane or a region that bled into its
    neighbour needs. napari 0.8's Labels layer already carries a POLYGON mode
    (click the corners, double-click or Enter to close); pointing it at label
    0 turns it into an eraser, which is all this panel wires up -- no new
    editing path, so undo/redo and the keyframe bookkeeping are unchanged.

    n_edit_dimensions is forced back to 2 on the way in, for two reasons:
    polygon painting is 2D-only (Labels._get_polygon_mask_and_bbox raises
    otherwise) and the Relabel panel's click-to-fill leaves it at 1, so
    erasing after a fill would otherwise throw. 2 also means the erase stays
    on the plane you can see -- every other plane is hand-drawn work that a
    polygon dragged somewhere else must not touch.

    It erases every label inside the polygon, not just the selected one:
    that is what "eraser" means everywhere else in the tool, and the blob
    you are rubbing out is often exactly the one that came out under the
    wrong number. Set napari's own "preserve labels" if you need the other
    behaviour.
    """
    # The label to come back to. Remembered rather than read on the way out,
    # because by then selected_label is 0 (the eraser) and the number the
    # user was painting with would be lost.
    last_label = {"value": max(1, int(paint_layer.selected_label))}

    status = QLabel("")
    status.setWordWrap(True)

    def start_polygon_erase():
        if paint_layer.selected_label:
            last_label["value"] = int(paint_layer.selected_label)
        viewer.layers.selection = {paint_layer}   # modes belong to the active layer
        paint_layer.n_edit_dimensions = 2
        paint_layer.selected_label = 0            # 0 = background = erase
        paint_layer.mode = "polygon"
        status.setText(
            "Polygon erase on this plane: left-click each corner, then double-click "
            "(or press Enter) to close it -- everything inside is erased, whatever "
            "label it had. Right-click drops the last corner, Esc drops the whole "
            "polygon, Ctrl+Z undoes a finished erase.")

    def back_to_brush():
        viewer.layers.selection = {paint_layer}
        # 2 = napari's default, i.e. the brush paints the plane on screen.
        # Restored here because the Relabel panel's click-to-fill drops it to
        # 1 and leaves it there.
        paint_layer.n_edit_dimensions = 2
        paint_layer.selected_label = last_label["value"]
        paint_layer.mode = "paint"
        status.setText(f"Back to the brush, painting label {last_label['value']}.")

    polygon_btn = QPushButton("Polygon erase")
    polygon_btn.clicked.connect(start_polygon_erase)
    brush_btn = QPushButton("Back to brush")
    brush_btn.clicked.connect(back_to_brush)

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(QLabel(
        "Erase a mistake with a polygon instead of scrubbing it out with the\n"
        f"eraser brush. The brush/eraser size slider goes up to {MAX_BRUSH_SIZE}."))
    layout.addWidget(polygon_btn)
    layout.addWidget(brush_btn)
    layout.addWidget(ontology_tree_ui.scrollable(status, 80))
    ontology_tree_ui.shrinkable(dock)
    return viewer.window.add_dock_widget(dock, area="right", name="Erase")


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
        raise ValueError(f"labels must be within 1..{MAX_LABEL} (the export is uint8), "
                         f"got {too_big}")

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
    .nii.gz is special-cased the same way tools/edit_sample_labels.py's
    _annotation_sidecar_path does it, so the names line up with the sidecar
    convention already in use."""
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
    sidecar (written by tools/edit_sample_labels.py, read by
    registration_eval.py's load_region_annotation_hint) -- same
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
    real keyframes onto a fresh baseline" rule tools/edit_sample_labels.py's
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
        print(f"WARNING: resume file shape {arr.shape} != image shape {expected_shape}, "
              f"not pre-filling.")
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
        raise ValueError(f"to_label must be within 0..{MAX_LABEL} (the export is uint8), "
                         f"got {to_label}")
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


def _add_ontology_picker(viewer, atlas, paint_layer, assignment):
    """The ontology tree + label-assignment panel, as its own DEDICATED dock
    on the SAMPLE viewer's LEFT side.

    Selecting a node assigns it (and everything under it) to a brush label;
    nothing here is displayed anywhere. To actually SEE the atlas -- a
    region highlighted among its neighbours in three synced panes, hover
    ancestry -- run the separate tools/atlas_view.py against the same
    atlas_annotation_path / ontology_path. The two tools no longer share any
    state: this panel used to drive a second napari window live (see
    highlight_mask / _open_atlas_window, both now in tools/atlas_view.py /
    shared/atlas_reference.py); it just assigns now.

    A dedicated left-side dock, not squeezed in with Relabel/Export on the
    right: the ontology sits 2-12 levels deep, so a tree squeezed into a
    fraction of a shared column leaves most of it scrolled out of view.

    TWO trees, split by a draggable QSplitter. The lower one is what has been
    assigned so far, as brush label -> its regions, and it is a TREE rather
    than the text block it used to be for one reason: a label routinely
    carries a dozen regions (DevCCF has no single "cortex", only 36 `layer N
    of <area>` structures), and taking one of them back out used to mean
    hunting that structure down in the 12-deep ontology above and pressing
    "Remove from label" -- with nothing on screen to click even though the
    thing to remove was right there in the list. Now the region row itself is
    the handle. The splitter is there because a fixed-height text box that
    folds after two labels was the other half of the same complaint.

    `assignment` ({label: [structure id]}) is mutated in place -- it is the
    live state the export reads, so there is no separate "apply" step to
    forget. A label whose last region is removed stays in it as an EMPTY
    entry, on purpose: see empty_assignment_labels.
    """
    # objectNames so these are addressable from outside the closure -- napari
    # contributes its own QSpinBox/QLineEdit widgets to the same window, so
    # "the first spin box" is not this panel's brush-label box.
    search = QLineEdit()
    search.setObjectName("ontology_search")
    search.setPlaceholderText("Filter regions by name/acronym...")
    hide_empty = QCheckBox("Only regions with voxels in this annotation")
    hide_empty.setObjectName("ontology_hide_empty")
    hide_empty.setChecked(True)

    tree = QTreeWidget()
    tree.setObjectName("ontology_tree")
    tree.setHeaderLabels(["Region", "Voxels", "id"])
    tree.setColumnWidth(0, 260)
    # No minimum height, deliberately: the tree is the only widget in its
    # half of the splitter carrying a stretch factor, so it already takes
    # every pixel that half is given beyond what the search box, the status
    # box and the buttons need -- see the module docstring's "large,
    # dedicated region panel" note for why that space was the point of this
    # dock. A floor would only fight the splitter handle.
    items = ontology_tree_ui.populate_ontology_tree(tree, atlas.structures, atlas.node_voxels)

    label_spin = QSpinBox()
    label_spin.setObjectName("ontology_brush_label")
    label_spin.setRange(1, MAX_LABEL)
    add_btn = QPushButton("Assign to label")
    add_btn.setObjectName("ontology_assign")
    remove_btn = QPushButton("Remove from label")
    remove_btn.setObjectName("ontology_unassign")
    picker_status = QLabel()

    # The lower half: what is assigned, one expandable row per brush label.
    assign_tree = QTreeWidget()
    assign_tree.setObjectName("assignment_tree")
    assign_tree.setHeaderLabels(["Brush label / region", "id"])
    assign_tree.setColumnWidth(0, 200)
    assign_tree.setMinimumHeight(90)
    assign_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
    drop_btn = QPushButton("Remove selected region(s)")
    drop_btn.setObjectName("assignment_remove")
    empty_note = QLabel()
    empty_note.setObjectName("assignment_empty_note")
    empty_note.setWordWrap(True)
    empty_note.setStyleSheet("color: #ffb86b;")   # a warning, not a caption
    empty_note.setVisible(False)

    def selected_id():
        item = tree.currentItem()
        return None if item is None else item.data(0, Qt.UserRole)

    def refresh_filter():
        visible = atlas_reference.visible_tree_ids(
            atlas.structures, atlas.node_voxels, search.text(), hide_empty.isChecked())
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
            picker_status.setText(
                f"{info['name']} [{sid}]: {voxels:,} voxels including descendants.")
        else:
            picker_status.setText(
                f"{info['name']} [{sid}] has no voxels in this annotation and cannot be "
                f"assigned -- the pipeline errors out on a region it cannot match.")

    def on_add():
        sid = selected_id()
        if sid is None:
            picker_status.setText("Select a region in the tree first.")
            return
        if not atlas.node_voxels.get(sid):
            picker_status.setText(
                f"{atlas.structures[sid]['name']} has no voxels in this annotation -- "
                f"refusing to assign it.")
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
            # Left as an empty entry rather than deleted: the label is still
            # what the brush is painting with, and losing it silently is the
            # failure empty_assignment_labels exists to make visible.
            assignment[label].remove(sid)
        refresh_assignment()

    def drop_selected():
        """Remove whatever is selected in the LOWER tree.

        A region row drops that one region. A label row drops every region
        under it (leaving the empty-label reminder), and dropping a label row
        that is ALREADY empty forgets the label -- so the reminder has an
        obvious way out that is not "assign something you don't want".
        """
        picked = assign_tree.selectedItems()
        if not picked:
            picker_status.setText(
                "Select a region (or a brush-label row) in the assignment tree below first.")
            return
        dropped, forgotten = 0, []
        for item in picked:
            label = item.data(0, Qt.UserRole)
            sid = item.data(1, Qt.UserRole)
            ids = assignment.get(label)
            if label is None or ids is None:
                continue
            if sid is None:
                if ids:
                    dropped += len(ids)
                    assignment[label] = []
                else:
                    del assignment[label]
                    forgotten.append(label)
            elif sid in ids:
                ids.remove(sid)
                dropped += 1
        refresh_assignment()
        parts = []
        if dropped:
            parts.append(f"removed {dropped} region(s) from the assignment")
        if forgotten:
            parts.append(f"forgot brush label(s) {', '.join(str(lab) for lab in forgotten)}")
        picker_status.setText(("; ".join(parts) + ".") if parts else "Nothing to remove.")

    def on_assignment_selected():
        """Clicking a row picks that brush label, so removing a region and
        carrying on painting with the same label needs no second control."""
        picked = assign_tree.selectedItems()
        label = picked[0].data(0, Qt.UserRole) if picked else None
        if label is None:
            return
        label_spin.setValue(int(label))
        paint_layer.selected_label = int(label)

    def refresh_assignment():
        assign_tree.clear()
        rows = assignment_rows(assignment, atlas.structures)
        for label, regions in rows:
            head = QTreeWidgetItem([f"label {label}    ({len(regions)} region(s))"
                                    if regions else
                                    f"label {label}    -- NO REGION LEFT", ""])
            head.setData(0, Qt.UserRole, label)
            head.setData(1, Qt.UserRole, None)
            assign_tree.addTopLevelItem(head)
            for sid, name in regions:
                child = QTreeWidgetItem([name, str(sid)])
                child.setData(0, Qt.UserRole, label)
                child.setData(1, Qt.UserRole, sid)
                head.addChild(child)
        if not rows:
            hint = QTreeWidgetItem(["No region assigned yet -- pick one above, set a brush "
                                    "label, then Assign to label.", ""])
            hint.setDisabled(True)
            assign_tree.addTopLevelItem(hint)
        assign_tree.expandAll()

        empties = empty_assignment_labels(assignment)
        empty_note.setVisible(bool(empties))
        if empties:
            empty_note.setText(
                "! brush label(s) " + ", ".join(str(lab) for lab in empties) + ": no region left. "
                "Painting with them exports an outline nothing can be paired with. Assign "
                "one, or Remove the label row again to forget it.")

    search.textChanged.connect(lambda _t: refresh_filter())
    hide_empty.toggled.connect(lambda _c: refresh_filter())
    tree.currentItemChanged.connect(lambda _cur, _prev: on_select())
    add_btn.clicked.connect(on_add)
    remove_btn.clicked.connect(on_remove)
    drop_btn.clicked.connect(drop_selected)
    assign_tree.itemSelectionChanged.connect(on_assignment_selected)

    upper = QWidget()
    upper_layout = QVBoxLayout(upper)
    upper_layout.setContentsMargins(0, 0, 0, 0)
    upper_layout.addWidget(QLabel("Atlas ontology -- selecting a node assigns it to the brush "
                                  "label below. The atlas itself is not shown here; run "
                                  "tools/atlas_view.py to look at it."))
    upper_layout.addWidget(search)
    upper_layout.addWidget(hide_empty)
    upper_layout.addWidget(tree, 1)      # the stretch: spare height is the tree's
    # Pinned to its own height (it scrolls inside), so the blurb about the
    # selected region cannot quietly take a hundred pixels off the tree --
    # the panel being too small for the regions in it is the complaint this
    # whole splitter exists to answer. A height cap only; the WIDTH stays
    # free, which is the one set_dock_width/shrinkable insist on.
    status_box = ontology_tree_ui.scrollable(picker_status, 56)
    status_box.setMaximumHeight(56)
    upper_layout.addWidget(status_box)
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.addWidget(QLabel("brush label"))
    row_layout.addWidget(label_spin)
    row_layout.addWidget(add_btn)
    row_layout.addWidget(remove_btn)
    upper_layout.addWidget(row)

    lower = QWidget()
    lower_layout = QVBoxLayout(lower)
    lower_layout.setContentsMargins(0, 0, 0, 0)
    lower_layout.addWidget(QLabel("Assigned so far -- select any region and remove it here."))
    lower_layout.addWidget(assign_tree, 1)
    lower_layout.addWidget(drop_btn)
    lower_layout.addWidget(empty_note)

    # A splitter, not two stacked widgets: how much of the column the
    # assignment is worth depends on how many labels there are (five or six
    # is normal), and that is exactly what a fixed split cannot know.
    splitter = QSplitter(Qt.Vertical)
    splitter.addWidget(upper)
    splitter.addWidget(lower)
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 2)
    splitter.setSizes([560, 320])

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(splitter)
    for widget in (dock, tree, assign_tree, upper, lower, splitter, empty_note):
        ontology_tree_ui.shrinkable(widget)
    dock_widget = viewer.window.add_dock_widget(dock, area="left", name="Atlas / Ontology")
    ontology_tree_ui.set_dock_width(dock_widget, _ONTOLOGY_PANEL_START_PX)

    refresh_filter()
    refresh_assignment()
    return SimpleNamespace(refresh_assignment=refresh_assignment, dock=dock_widget)


def _run_guide(args):
    _import_gui()
    base_sitk, arr = _read_sitk_array(args.image_path)

    prefill = np.zeros(arr.shape, dtype=np.uint8)
    resume = load_guide_resume(args.existing_mask, arr.shape) if args.existing_mask else None
    if resume is not None:
        prefill = resume.prefill
        planes = sum(len(p) for p in resume.slices_by_label.values())
        print(f"[resume] restored {planes} hand-drawn planes from {resume.sidecar.name} "
              f"({ {lab: p for lab, p in sorted(resume.slices_by_label.items())} }); "
              f"the interpolated planes were dropped -- just keep painting.")
    elif args.existing_mask:
        # No sidecar: all this can do is binarize, which merges every region
        # into label 1 and treats interpolated planes as hand-drawn. Usable as
        # a rough tracing backdrop, not as a resume.
        loaded = _load_mask_array(args.existing_mask, arr.shape)
        if loaded is not None:
            prefill = loaded
            print(f"WARNING: no .regions.json next to {args.existing_mask}, so it cannot be "
                  f"resumed as keyframes.\n"
                  f"         Pre-filled binarized instead: every region is merged into label 1, "
                  f"and interpolated planes count as hand-drawn.\n"
                  f"         Good as a tracing backdrop only; to really resume, use an export "
                  f"that still has its sidecar.")

    viewer, paint_layer = _launch_viewer(
        arr, prefill, scale=display_scale_from_voxel_size(args.voxel_size_um))

    # The atlas ontology is loaded here only to populate the region-assignment
    # tree and check which structures this annotation actually has voxels
    # for -- nothing about it is displayed. To look at the atlas itself, run
    # tools/atlas_view.py against the same atlas_annotation_path / ontology_path.
    atlas = assignment = picker = None
    if args.atlas:
        atlas = atlas_reference.load_atlas_reference(args.atlas, include_template=False)

        # A resumed file's own sidecar wins over the config: it records what
        # those label numbers actually meant last session, and painting more
        # planes under a label that silently changed region would be worse
        # than any config convenience.
        seed_labels = resume.region_labels if resume is not None else args.region_labels
        seed_ids = resume.region_ids if resume is not None else args.region_ids
        assignment, unresolved = _seed_assignment(seed_labels, seed_ids, atlas.structures)
        for label, name, n in unresolved:
            print(f"WARNING: region_labels label {label}: {name!r} matches "
                  f"{'several' if n else 'no'} structures in the ontology ({n}), so it was not "
                  f"resolved to an id; pick it again in the tree.")
        picker = _add_ontology_picker(viewer, atlas, paint_layer, assignment)

    guess_note = "Pre-filled with the existing mask -- adjust/redraw as needed.\n" if args.existing_mask else ""
    header = ("Pick a region in the ontology tree on the left, set a brush label, click\n"
              "Assign to label, then paint the sample with that brush number.\n"
              if atlas else _region_legend(args.region_labels))
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
            lines += ["", "Paste this into the pipeline config:", "",
                      guide_regions_yaml_snippet(
                          region_ids, region_labels, args.output_path,
                          voxel_size_um=args.voxel_size_um)]

        msg = "\n".join(lines)
        status_label.setText(msg)
        print(msg)

    export_dock = _make_export_dock(viewer, status_label, export, "Export Outline",
                                    "Guide Outline Export")

    # A bulk relabel has to carry the region assignment with it, or the label
    # keeps its voxels and loses its meaning -- the exact thing the ontology
    # picker exists to prevent. Merging onto a label that already has a region
    # keeps the destination's, since that is the one the user just pointed at.
    # An EMPTY entry travels too (`is not None`, not truthiness): the reminder
    # that a label lost its regions belongs to whichever number now carries
    # those voxels.
    def _assignment_follows_relabel(src, dst):
        if assignment is None:
            return
        ids = assignment.pop(src, None)
        if ids is not None and dst not in assignment:
            assignment[dst] = ids
        if picker is not None:
            picker.refresh_assignment()

    relabel_dock = _add_relabel_panel(viewer, paint_layer,
                                      on_change=_assignment_follows_relabel)
    erase_dock = _add_erase_panel(viewer, paint_layer)
    display_dock = _add_display_panel(viewer, [paint_layer])
    _tab_the_panels(viewer,
                    left=[picker.dock] if picker is not None else [],
                    right=[export_dock, relabel_dock, erase_dock, display_dock])


# =====================================================================================
# mode: labels -- painting on a registration RESULT rather than on blank planes
# =====================================================================================
# `mode: guide` above starts from an empty paint layer: you trace regions on the
# raw sample and every plane you do not touch stays background. This mode starts
# from <name>_labels_in_sample.nii.gz -- a finished registration -- collapsed into
# the current partition's brush labels, and you correct where it came out wrong.
#
# Three things differ, and all three follow from "the layer arrives pre-filled":
#
#   1. A keyframe is a WHOLE PLANE, not the pixels you touched. In guide mode an
#      untouched pixel means "no outline here"; here it means "the registration
#      was already right here", which is a positive statement about that plane's
#      anatomy and belongs in the guide. So a plane counts as hand-drawn as soon
#      as it differs from the baseline collapse anywhere, and the whole plane --
#      every region on it, corrected or not -- becomes the keyframe.
#
#   2. Interpolation is mask_utils.interpolate_sparse_label_correction, not
#      interpolate_labels_separately. Every region shares the same keyframe
#      planes here (they are whole planes), so the interleaving problem that
#      forces per-label interpolation in guide mode cannot arise; what is needed
#      instead is for neighbouring regions to COMPETE for the voxels between two
#      keyframes, which is exactly that function's per-label signed-distance
#      contest.
#
#   3. Two volumes come out, not one:
#        <output_path>        sparse guide, empty outside the keyframe span, for
#                             mask.guide_regions -- i.e. for re-registering.
#        <atlas_output_path>  dense, every plane filled, for re-opening and
#                             drawing more. Same keyframes, baseline swapped from
#                             zeros to the full collapse.
#      The dense one must never be used as the baseline for its own next export,
#      or each session interpolates on top of the last one's guess; the
#      .keyframes.json sidecar records which planes were real and where the true
#      baseline lives, and load_labels_resume enforces it.


def plane_keyframes(paint, baseline):
    """{z: (all-True mask, paint[z])} for every plane that differs from the
    baseline collapse anywhere.

    The mask is all-True on purpose -- see point 1 above. It is still passed
    explicitly rather than assumed, because interpolate_sparse_label_correction
    is shared with tools/edit_sample_labels.py, where the mask really is the
    sparse set of touched pixels.
    """
    changed = np.any(paint != baseline, axis=(1, 2))
    full = np.ones(paint.shape[1:], dtype=bool)
    return {int(z): (full, paint[int(z)]) for z in np.flatnonzero(changed)}


def recollapse_keeping_edits(paint, old_baseline, new_baseline):
    """Re-derive the paint layer after the partition changed, keeping the
    hand edits and refreshing everything else.

    Expanding a group renumbers most of the volume, so the layer has to be
    rebuilt -- but a voxel the user actually repainted must survive verbatim,
    or expanding would quietly discard the correction that motivated it. A
    voxel counts as edited exactly when it disagreed with the OLD baseline.

    The useful consequence: the parts of a keyframe plane you never touched
    are refined to the new partition automatically, so expanding
    Hippocampal formation gives you CA/DG boundaries on planes you had
    already corrected at the coarse level, without redrawing them.
    """
    edited = paint != old_baseline
    return np.where(edited, paint, new_baseline).astype(np.uint8)


def labels_export(paint, baseline, interpolate=None):
    """Whole-plane keyframes -> (sparse guide volume, dense atlas volume).

    Returns SimpleNamespace(guide, atlas, hand_drawn_slices, slices_by_label,
    voxels_by_label, overlap_pairs, n_contested) -- the last four shaped like
    interpolate_labels_separately's result so write_guide_sidecars and
    guide_export_warnings can be reused unchanged. overlap_pairs is always
    empty and n_contested always 0: a napari Labels layer is a single-valued
    raster, so a partition cannot have two labels claim one voxel the way
    separately-interpolated outlines can.
    """
    interpolate = interpolate or _interpolate_sparse_label_correction()
    keyframes = plane_keyframes(paint, baseline)
    if not keyframes:
        return None

    guide = interpolate(keyframes, np.zeros_like(baseline))
    atlas = interpolate(keyframes, baseline)

    slices_by_label = {}
    for z, (_mask, plane) in sorted(keyframes.items()):
        for label in np.unique(plane):
            if label:
                slices_by_label.setdefault(int(label), []).append(int(z))
    return SimpleNamespace(
        guide=guide.astype(np.uint8),
        atlas=atlas.astype(np.uint8),
        hand_drawn_slices=sorted(keyframes),
        slices_by_label={lab: sorted(zs) for lab, zs in sorted(slices_by_label.items())},
        voxels_by_label={lab: int(np.count_nonzero(guide == lab)) for lab in sorted(slices_by_label)},
        overlap_pairs={},
        n_contested=0,
    )


def labels_export_warnings(result, partition, structures, own_voxels, total_z,
                           node_voxels=None, voxel_mm3=None,
                           min_mm3=label_partition.DEFAULT_MIN_MM3):
    """The mode-specific checks, on top of guide_export_warnings'."""
    warnings = []

    painted = set(result.slices_by_label)
    empty = [lab for lab in partition.empty_atlas_side(structures, own_voxels) if lab in painted]
    for label in empty:
        group = partition.groups[label]
        warnings.append(
            f"label {label} ({group.name}) is still painted on planes "
            f"{result.slices_by_label[label]}, but every one of its atlas regions has been "
            f"split out into a child label, so its atlas outline is EMPTY. The pipeline "
            f"aborts the whole run on that (_build_guide_regions_from_labels raises). "
            f"Repaint those voxels with the child labels, or merge the children back.")

    if node_voxels is not None and voxel_mm3:
        for label in sorted(painted):
            group = partition.groups.get(label)
            if group is None:
                continue
            mm3 = sum(node_voxels.get(i, 0) for i in group.ids) * voxel_mm3
            if mm3 < min_mm3:
                warnings.append(
                    f"label {label} ({group.name}) is only ~{mm3:.2f} mm3 in the atlas. A guide "
                    f"region that small usually drags the deformation the wrong way -- the "
                    f"hand-drawn boundary error is a large fraction of the structure.")

    n = len(result.hand_drawn_slices)
    if total_z and n > total_z / 2:
        warnings.append(
            f"{n} of {total_z} planes count as hand-drawn. That is most of the volume, which "
            f"usually means a bulk edit (Relabel the whole label) touched planes you never "
            f"looked at -- every one of them is now a keyframe. Check the plane list above.")
    return warnings


def _labels_sidecar_path(atlas_output_path):
    return _output_stem(atlas_output_path).with_name(
        _output_stem(atlas_output_path).name + ".keyframes.json")


def write_labels_sidecar(atlas_output_path, guide_output_path, labels_path, result,
                         partition, structures, total_z, grids=None):
    """<atlas_output>.keyframes.json -- what makes the dense volume safely
    re-openable.

    It records the true baseline's path, NOT just the plane list, because the
    dense volume is mostly interpolation: re-opening it as its own baseline
    would promote this session's guesses to next session's ground truth and
    compound every round. load_labels_resume reads the baseline back from
    here and overlays only these planes.

    The partition is stored with its parent links so an expand can be merged
    back after a resume -- the nesting is what atlas_exclude_ids is derived
    from, and losing it would silently drop those subtractions.
    """
    path = _labels_sidecar_path(atlas_output_path)
    path.write_text(json.dumps({
        "hand_drawn_slices": result.hand_drawn_slices,
        "baseline_labels_path": str(Path(labels_path).resolve()),
        "guide_path": str(guide_output_path),
        "atlas_path": str(atlas_output_path),
        "total_z": int(total_z),
        "region_ids": {str(g.label): list(g.ids) for g in partition},
        "region_names": {str(g.label): g.name for g in partition},
        "parents": {str(g.label): g.parent.label for g in partition if g.parent is not None},
        "atlas_exclude_ids": {str(k): v for k, v in partition.atlas_exclude_ids(structures).items()},
        # Both grids, because the volumes here are on the raw stack's while
        # baseline_labels_path points at one on the registration's -- a resume
        # has to regrid again and the two voxel sizes are in neither header.
        "grids": grids,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_labels_resume(atlas_output_path, structures, expected_shape):
    """Restore a previous mode-labels export, or None if there is nothing to
    resume. Returns SimpleNamespace(partition, hand_drawn_slices, planes,
    baseline_labels_path, sidecar) where `planes` is {z: 2D uint8}.

    Only the recorded planes come back, at their exported values -- same rule
    as load_guide_resume and tools/edit_sample_labels.py's
    _load_prior_hand_drawn, for the same reason: everything else in that file
    is this tool's own interpolation.
    """
    atlas_output_path = Path(atlas_output_path)
    sidecar = _labels_sidecar_path(atlas_output_path)
    if not (atlas_output_path.exists() and sidecar.exists()):
        return None

    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(atlas_output_path)))
    if arr.shape != expected_shape:
        print(f"WARNING: resume file shape {arr.shape} != labels shape {expected_shape}, "
              f"not resuming.")
        return None

    partition = label_partition.Partition.from_region_ids(
        {int(k): v for k, v in meta["region_ids"].items()}, structures)
    for child, parent in (meta.get("parents") or {}).items():
        child, parent = int(child), int(parent)
        if child in partition.groups and parent in partition.groups:
            partition.groups[child].parent = partition.groups[parent]

    planes = [z for z in meta["hand_drawn_slices"] if 0 <= z < expected_shape[0]]
    return SimpleNamespace(
        partition=partition,
        hand_drawn_slices=sorted(planes),
        planes={int(z): arr[int(z)].astype(np.uint8) for z in planes},
        baseline_labels_path=meta.get("baseline_labels_path"),
        sidecar=sidecar,
    )


def labels_voxel_size_um(spacing_xyz, override=None):
    """The (x,y,z) micron voxel size of a labels_in_sample.nii.gz.

    Every image this codebase writes carries spacing DIRECTLY IN MICRONS
    (io_utils.load_tiff_stack_as_ants / resample_to_isotropic are handed
    micron values and never divide), so a pipeline output reads back as
    25.0, not 0.025. Files from elsewhere follow the NIfTI convention and
    are in millimetres -- the DevCCF downloads read back as 0.02. Both have
    to work here, and getting it wrong by 1000x silently regrids the labels
    to a sliver of the stack rather than erroring, so the two cases are told
    apart by magnitude and the choice is announced.

    override wins outright, for the case where neither guess is right.
    """
    if override:
        return [float(v) for v in override]
    spacing = [float(s) for s in spacing_xyz]
    if all(abs(s - 1.0) < 1e-6 for s in spacing):
        raise ValueError(
            "labels_path has no voxel size in its header (spacing is 1,1,1). Set "
            "labels_voxel_size_um in the config -- it is needed to overlay the labels on "
            "the raw stack, so it cannot be guessed.")
    if max(spacing) < 1.0:
        print(f"[labels] header spacing {spacing} is below 1, reading it as MILLIMETRES "
              f"-> {[s * 1000 for s in spacing]} um. Set labels_voxel_size_um to override.")
        return [s * 1000.0 for s in spacing]
    return spacing


def regrid_nearest(arr_zyx, src_spacing_zyx, dst_shape_zyx, dst_spacing_zyx):
    """Nearest-neighbour regrid of a label volume between two axis-aligned
    grids that share physical origin 0 and identity direction.

    That precondition is not an assumption, it is this codebase's invariant:
    io_utils.load_tiff_stack_as_ants and resample_to_isotropic never pass a
    nonzero origin or a non-default direction, and io_utils.crop_to_bounds
    shifts origin precisely so a crop stays in the same physical space. So
    the mapping is one multiplication per axis and needs no transform.

    Done as a gather rather than through ants.resample_image_to_target
    because the destination here is the raw stack -- 2273x3974x157 for the
    s12t sample. A float32 ANTs round trip of that is ~5.7 GB per copy;
    indexing a uint8 array straight into place is one output-sized
    allocation and no interpolation to get wrong on discrete ids.
    """
    idx = []
    for axis in range(3):
        scale = float(dst_spacing_zyx[axis]) / float(src_spacing_zyx[axis])
        pos = np.rint(np.arange(dst_shape_zyx[axis]) * scale).astype(np.int64)
        idx.append(np.clip(pos, 0, arr_zyx.shape[axis] - 1))
    return arr_zyx[np.ix_(*idx)]


def _seed_partition(args, structures, resume):
    """Where the starting partition comes from, most specific first: a resumed
    session, then partition_path (a .regions.json -- e.g. the one an earlier
    `mode: guide` export already wrote), then the config's own region_ids."""
    if resume is not None:
        return resume.partition, f"resumed from {resume.sidecar.name}"
    if args.partition_path:
        return (label_partition.Partition.from_regions_json(args.partition_path, structures),
                f"seeded from {Path(args.partition_path).name}")
    if args.region_ids:
        return (label_partition.Partition.from_region_ids(args.region_ids, structures),
                "seeded from the config's region_ids")
    raise ValueError(
        "mode: labels needs a starting partition. Set partition_path to a .regions.json "
        "(the sidecar any guide export writes), or list region_ids in the config.")


def _add_partition_panel(viewer, paint_layer, partition, structures, node_voxels,
                         own_voxels, voxel_mm3, min_mm3, on_partition_changed):
    """The panel that replaces guide mode's ontology tree.

    Guide mode picks a region and assigns it to a free brush number -- a flat
    mapping, built by hand. Here the mapping already exists (it came from the
    registration) and what you do to it is REFINE it, one node at a time, so
    the control that matters is expand/merge on the selected group rather
    than a 12-deep tree to hunt through. Depth is per-group on purpose: see
    label_partition's docstring for the measured reason a uniform ontology
    depth is not usable on CCFv3.
    """
    status = QLabel("")
    status.setWordWrap(True)
    status.setTextInteractionFlags(Qt.TextSelectableByMouse)

    listing = QListWidget()

    def selected_label():
        row = listing.currentRow()
        return listing._labels[row] if 0 <= row < len(getattr(listing, "_labels", [])) else None

    def refresh(message=""):
        keep = selected_label()
        listing.clear()
        listing._labels = []
        for group in partition:
            mm3 = sum(node_voxels.get(i, 0) for i in group.ids) * voxel_mm3
            kids = partition.children_of(group.label)
            note = f"  [residual, {len(kids)} split out]" if kids else ""
            listing.addItem(f"{group.label:>3}  {group.name}   ~{mm3:.1f} mm3{note}")
            listing._labels.append(group.label)
        if keep in listing._labels:
            listing.setCurrentRow(listing._labels.index(keep))
        empty = partition.empty_atlas_side(structures, own_voxels)
        tail = (f"\nResidual labels with an EMPTY atlas side (fine unless still painted): "
                f"{empty}" if empty else "")
        status.setText((message or "Pick a group; the brush switches to its label.") + tail)

    def on_row_changed(_row):
        label = selected_label()
        if label is not None:
            paint_layer.selected_label = label

    def expand():
        label = selected_label()
        if label is None:
            return
        try:
            kept, skipped = partition.expand(label, structures, node_voxels, voxel_mm3, min_mm3)
        except ValueError as exc:
            refresh(f"Cannot expand: {exc}")
            return
        if not kept:
            refresh(f"label {label} has no child region big enough to split out "
                    f"(skipped: {[n for _i, n, _m in skipped]}).")
            return
        on_partition_changed()
        msg = (f"Expanded label {label} -> " +
               ", ".join(f"{n} ({m:.1f} mm3)" for _i, n, m in kept))
        if skipped:
            msg += ("\nLeft with the parent (under "
                    f"{min_mm3} mm3): " + ", ".join(f"{n} ({m:.2f})" for _i, n, m in skipped))
        refresh(msg)

    def merge():
        label = selected_label()
        if label is None:
            return
        removed = partition.merge_back(label)
        if not removed:
            refresh(f"label {label} has nothing split out of it.")
            return
        on_partition_changed()
        refresh(f"Merged {len(removed)} group(s) back into label {label}: "
                + ", ".join(g.name for g in removed))

    listing.currentRowChanged.connect(on_row_changed)
    expand_btn = QPushButton("Expand one level")
    expand_btn.clicked.connect(expand)
    merge_btn = QPushButton("Merge children back")
    merge_btn.clicked.connect(merge)

    isolate = QCheckBox("Show only the selected group")
    isolate.toggled.connect(lambda checked: setattr(paint_layer, "show_selected_label", checked))

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(QLabel(
        "Brush label -> atlas region. Expanding splits one group into its\n"
        f"ontology children; children under {min_mm3} mm3 stay with the parent,\n"
        "because a guide region that small drags the deformation the wrong way."))
    # The stretch, plus a status box pinned to its own height: the group list
    # is what this panel is FOR and a partition routinely runs to a dozen
    # groups, so spare height belongs to it rather than to the blank half of
    # a message box. (Height only -- the width stays draggable, see
    # ontology_tree_ui.shrinkable.)
    layout.addWidget(listing, 1)
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.addWidget(expand_btn)
    row_layout.addWidget(merge_btn)
    layout.addWidget(row)
    layout.addWidget(isolate)
    status_box = ontology_tree_ui.scrollable(status, 100)
    status_box.setMaximumHeight(100)
    layout.addWidget(status_box)
    ontology_tree_ui.shrinkable(dock)
    ontology_tree_ui.shrinkable(listing)
    dock_widget = viewer.window.add_dock_widget(dock, area="left", name="Partition")
    ontology_tree_ui.set_dock_width(dock_widget, _ONTOLOGY_PANEL_START_PX)
    refresh()
    return SimpleNamespace(refresh=refresh, dock=dock_widget)


def _run_labels(args):
    """`mode: labels` -- correct a finished registration, export a guide to
    re-register with plus a dense volume to carry on from.

    Everything happens on the RAW stack's grid, not on the isotropic grid the
    registration ran on. That is not a preference: the resample to
    fine_target_um throws away ~8x of the in-plane detail (2.6 um pixels
    become 20 um ones at the fine_target_um used here) and replaces the real
    imaging planes with interpolated ones, so the
    boundaries being corrected are no longer resolvable by eye and the plane
    being drawn on is not a plane that was ever imaged.
    pipeline.py's _build_guide_regions_from_labels already states this as the
    convention for the painted volume; the registration output is brought TO
    that grid here, rather than the painting being dragged down to it.
    """
    _import_gui()
    if not args.labels_path:
        raise ValueError("mode: labels needs labels_path (the <name>_labels_in_sample.nii.gz "
                         "a completed registration wrote)")
    if not args.atlas:
        raise ValueError("mode: labels needs atlas_annotation_path + ontology_path: the "
                         "partition is expressed in that ontology's ids")
    raw_voxel_um = args.voxel_size_um
    if not raw_voxel_um:
        raise ValueError(
            "mode: labels needs voxel_size_um: [x, y, z] for image_path -- the raw stack's "
            "header does not carry one, and it is what puts the registration output onto the "
            "same grid.")

    atlas_output_path = args.atlas_output_path or str(
        _output_stem(args.output_path).with_name(_output_stem(args.output_path).name + "_atlas.nii.gz"))

    raw_sitk, sample_arr = _read_sitk_array(args.image_path)
    labels_sitk, fine_labels = _read_sitk_array(args.labels_path)
    fine_labels = fine_labels.astype(np.uint32)
    fine_voxel_um = labels_voxel_size_um(labels_sitk.GetSpacing(), args.labels_voxel_size_um)

    raw_spacing_zyx = list(reversed(raw_voxel_um))
    fine_spacing_zyx = list(reversed(fine_voxel_um))
    print(f"[grids] raw stack   {sample_arr.shape} (z,y,x) @ {raw_spacing_zyx} um\n"
          f"[grids] registration {fine_labels.shape} (z,y,x) @ {fine_spacing_zyx} um")
    covered = [f * n for f, n in zip(fine_spacing_zyx, fine_labels.shape)]
    extent = [s * n for s, n in zip(raw_spacing_zyx, sample_arr.shape)]
    if any(c < 0.9 * e for c, e in zip(covered, extent)):
        print(f"WARNING: the registration grid spans {[round(c) for c in covered]} um but the raw "
              f"stack spans {[round(e) for e in extent]} um. Either the two are not the same "
              f"sample, or a voxel size is wrong -- check voxel_size_um and labels_voxel_size_um "
              f"before painting, because the overlay will be silently offset.")

    atlas = atlas_reference.load_atlas_reference(args.atlas, include_template=False)
    structures = atlas.structures
    counts = np.bincount(atlas.compact.ravel(), minlength=len(atlas.present_ids))
    own_voxels = {int(sid): int(n) for sid, n in zip(atlas.present_ids, counts)}
    res_um = args.atlas.resolution_um
    if not res_um:
        res_um = 25.0
        print("WARNING: atlas_resolution_um is not set, assuming 25 um. Every mm3 shown in the "
              "partition panel -- and therefore which children are big enough to split out -- "
              "scales with its cube, so set it if the atlas is not 25 um. Both presets here are "
              "20 um (off by (25/20)^3 ~ 1.95x), and a TIFF annotation like DeMBA's carries no "
              "spacing to read it from, so it has to come from the config.")
    voxel_mm3 = (res_um / 1000.0) ** 3 * (atlas.downsample ** 3)

    resume = load_labels_resume(atlas_output_path, structures, sample_arr.shape)
    partition, seed_note = _seed_partition(args, structures, resume)
    if resume is not None and resume.baseline_labels_path and \
            Path(resume.baseline_labels_path).resolve() != Path(args.labels_path).resolve():
        print(f"WARNING: {resume.sidecar.name} was made against\n"
              f"           {resume.baseline_labels_path}\n"
              f"         but labels_path is\n           {args.labels_path}\n"
              f"         Resuming anyway, but the restored planes describe the other volume.")

    def baseline_for(partition):
        """The registration output in brush space, on the RAW grid.

        Collapsed first and regridded second, deliberately: collapsing works
        on the small isotropic volume (~20M voxels) and turns uint32 ids into
        uint8 brush labels, so the one array that reaches the raw grid's ~1.4e9
        voxels is a byte per voxel instead of four.
        """
        return regrid_nearest(partition.collapse(fine_labels, structures), fine_spacing_zyx,
                              sample_arr.shape, raw_spacing_zyx)

    state = {"baseline": baseline_for(partition)}
    prefill = state["baseline"].copy()
    if resume is not None:
        for z, plane in resume.planes.items():
            prefill[z] = plane
        print(f"[resume] restored {len(resume.planes)} hand-drawn plane(s) "
              f"{resume.hand_drawn_slices} from {resume.sidecar.name}; the interpolated "
              f"planes were re-derived from {Path(args.labels_path).name}.")

    viewer, paint_layer = _launch_viewer(
        sample_arr, prefill, scale=display_scale_from_voxel_size(raw_voxel_um),
        title="Correct a registration result", layer_name="regions (paint here)")
    paint_layer.opacity = 0.5
    scale_kwargs = {"scale": display_scale_from_voxel_size(raw_voxel_um)}
    # The untouched registration, to compare a keyframe against once
    # interpolation has overwritten the planes between two of them in the
    # dense volume. In brush space rather than raw ontology ids: that is the
    # comparison that matters here, and a uint32 copy of the raw grid would
    # cost 4x this one for no extra information. Hidden by default.
    reference = viewer.add_labels(state["baseline"], name="registration as-is (read-only)",
                                  visible=False, opacity=0.4, **scale_kwargs)
    reference.editable = False

    status_label = QLabel("")
    status_label.setWordWrap(True)
    hover_label = QLabel("Under cursor: -")

    def on_partition_changed():
        new_baseline = baseline_for(partition)
        paint_layer.data = recollapse_keeping_edits(
            paint_layer.data, state["baseline"], new_baseline)
        state["baseline"] = new_baseline
        reference.data = new_baseline

    panel = _add_partition_panel(viewer, paint_layer, partition, structures, atlas.node_voxels,
                                 own_voxels, voxel_mm3, args.min_region_mm3, on_partition_changed)

    def on_mouse_move(_layer, event):
        data = paint_layer.data
        if data.ndim != 3:
            return
        z, y, x = (int(round(c)) for c in paint_layer.world_to_data(event.position))
        if not all(0 <= c < n for c, n in zip((z, y, x), data.shape)):
            return
        def described(value):
            group = partition.groups.get(value)
            if group is not None:
                return group.name
            return "background" if not value else f"unassigned label {value}"

        label = int(data[z, y, x])
        was = int(state["baseline"][z, y, x])
        # Showing what the registration said, not just what is there now, is
        # what tells a correction apart from a region you have not touched --
        # the whole plane looks hand-drawn once it becomes a keyframe.
        note = "" if label == was else f"   (registration said: {described(was)})"
        hover_label.setText(f"Under cursor: {described(label)} [{label}]{note}")

    paint_layer.mouse_move_callbacks.append(on_mouse_move)

    def describe():
        planes = sorted(plane_keyframes(paint_layer.data, state["baseline"]))
        return (f"{seed_note}; {len(partition)} groups. Painting on the raw stack "
                f"{sample_arr.shape} (z,y,x).\n"
                f"Planes that differ from the registration so far ({len(planes)}): {planes}\n"
                "Correct a plane anywhere and the WHOLE plane becomes a keyframe -- every\n"
                "region on it, not just what you repainted. Planes between two keyframes\n"
                "are interpolated; planes outside them stay empty in the guide.")

    status_label.setText(describe())

    def export():
        planes = sorted(plane_keyframes(paint_layer.data, state["baseline"]))
        if not planes:
            status_label.setText("Nothing differs from the registration yet -- nothing to "
                                 "export.\n" + describe())
            return
        # Said out loud because on the raw grid this is minutes, not seconds:
        # the interpolation runs a signed-distance transform per region per
        # pair of neighbouring keyframes, on planes of several megapixels.
        note = (f"Exporting {len(planes)} keyframe planes at {sample_arr.shape[1]}x"
                f"{sample_arr.shape[2]}. On the raw grid this takes a while (one distance "
                f"transform per region per keyframe gap) -- watch the terminal.")
        status_label.setText(note)
        print(note)

        result = labels_export(paint_layer.data, state["baseline"])
        region_ids = partition.region_ids()
        region_names = partition.region_names(structures)
        painted = set(result.slices_by_label)
        for volume, path in ((result.guide, args.output_path), (result.atlas, atlas_output_path)):
            out = sitk.GetImageFromArray(volume)
            out.CopyInformation(raw_sitk)   # the raw stack's own (1,1,1) -- see the module docstring
            sitk.WriteImage(out, str(path))

        atlas_info = {
            "annotation_path": str(args.atlas.annotation_path),
            "ontology_path": str(args.atlas.ontology_path),
            "orientation": list(args.atlas.orientation) if args.atlas.orientation else None,
            "resolution_um": args.atlas.resolution_um,
        }
        regions_path, slices_path = write_guide_sidecars(
            args.output_path, args.image_path, result,
            {lab: names for lab, names in region_names.items() if lab in painted},
            sample_arr.shape[0], spacing_xyz=raw_sitk.GetSpacing(),
            region_ids={lab: ids for lab, ids in region_ids.items() if lab in painted},
            atlas_info=atlas_info)
        keyframes_path = write_labels_sidecar(
            atlas_output_path, args.output_path, args.labels_path, result, partition,
            structures, sample_arr.shape[0],
            grids={"raw_shape_zyx": list(sample_arr.shape),
                   "raw_voxel_size_um_xyz": list(raw_voxel_um),
                   "labels_shape_zyx": list(fine_labels.shape),
                   "labels_voxel_size_um_xyz": list(fine_voxel_um)})

        lines = [f"Wrote {args.output_path}          (sparse guide -- re-register with this)",
                 f"Wrote {atlas_output_path}   (dense -- re-open this to keep drawing)",
                 f"Wrote {regions_path}", f"Wrote {slices_path}", f"Wrote {keyframes_path}", "",
                 f"Keyframe planes ({len(result.hand_drawn_slices)}): {result.hand_drawn_slices}"]
        for label in sorted(result.slices_by_label):
            lines.append(f"  label {label} ({_label_name(label, region_names)}):  "
                         f"{len(result.slices_by_label[label])} planes -> "
                         f"{result.voxels_by_label[label]} voxels")
        # Only the painted groups: guide mode warns about a configured label
        # with no outline because there it means "you forgot to draw it", but
        # a partition legitimately covers the whole brain while any one sample
        # only spans part of it.
        painted_names = {lab: names for lab, names in region_names.items() if lab in painted}
        lines += [f"WARNING: {w}" for w in guide_export_warnings(result, painted_names)]
        lines += [f"WARNING: {w}" for w in labels_export_warnings(
            result, partition, structures, own_voxels, sample_arr.shape[0],
            node_voxels=atlas.node_voxels, voxel_mm3=voxel_mm3, min_mm3=args.min_region_mm3)]

        exclude = {lab: ids for lab, ids in partition.atlas_exclude_ids(structures).items()
                   if lab in painted}
        lines += ["", "Paste this into the pipeline config:", "",
                  guide_regions_yaml_snippet(
                      {lab: ids for lab, ids in region_ids.items() if lab in painted},
                      region_names, args.output_path, voxel_size_um=raw_voxel_um,
                      atlas_exclude_ids=exclude,
                      voxel_size_note="# the raw stack's own (x,y,z) um, same grid as mode: guide")]

        msg = "\n".join(lines)
        status_label.setText(msg)
        print(msg)
        panel.refresh()

    dock = QWidget()
    QVBoxLayout(dock).addWidget(hover_label)
    ontology_tree_ui.shrinkable(dock)
    hover_dock = viewer.window.add_dock_widget(dock, area="right", name="Under cursor")
    export_dock = _make_export_dock(viewer, status_label, export, "Export Guide + Atlas",
                                    "Registration Correction Export")
    relabel_dock = _add_relabel_panel(
        viewer, paint_layer, on_change=lambda src, dst: status_label.setText(
            f"Bulk relabel {src} -> {dst} touched every plane it appears on -- each of those is "
            f"now a keyframe.\n" + describe()))
    erase_dock = _add_erase_panel(viewer, paint_layer)
    display_dock = _add_display_panel(viewer, [paint_layer, reference])
    _tab_the_panels(viewer, left=[panel.dock],
                    right=[export_dock, hover_dock, relabel_dock, erase_dock, display_dock])


# =====================================================================================
# selftests -- synthetic arrays only, no GUI, no config, no image on disk
# =====================================================================================
def _reference_interpolate_sparse_mask(keyframe_planes, full_shape):
    """Deliberate standalone minimal copy of
    registration_ants.mask_utils.interpolate_sparse_mask, used by the
    selftests ONLY when registration_ants isn't importable -- it lives in
    ../Registration_ants and is pip-installed-editable in the antsreg env, so
    a checkout without that install (or an env that never had it) can still
    run --selftest.
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
    # registration_ants.transforms -> antspyx, which --selftest does without.
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
    print("10. config: region_labels int/str keys + multi-region values, voxel_size_um,\n"
          "    mode sections")
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

    # (x,y,z) config voxel size -> (z,y,x) napari scale. Reversed, not
    # copied: the array axes and the config run in opposite orders.
    assert display_scale_from_voxel_size([2.6, 2.6, 32.0]) == [32.0, 2.6, 2.6]
    assert display_scale_from_voxel_size(None) is None

    def rejects(fn, value, needle):
        try:
            fn(value)
        except ValueError as exc:
            assert needle in str(exc), f"{value!r}: wrong reason {exc}"
        else:
            raise AssertionError(f"{value!r} should have been rejected")

    rejects(_normalize_region_labels, {"cortex": 1}, "integers")
    rejects(_normalize_region_labels, {0: "background"}, ">= 1")
    rejects(_normalize_region_labels, {1: "cortex", "1": "cortex"}, "twice")
    rejects(_normalize_region_labels, ["cortex"], "mapping")

    assert _normalize_voxel_size_um([2.6, 2.6, 32.0]) == [2.6, 2.6, 32.0]
    assert _normalize_voxel_size_um(None) is None
    rejects(_normalize_voxel_size_um, [2.6, 2.6], "exactly 3 numbers")
    rejects(_normalize_voxel_size_um, [2.6, 0.0, 32.0], "positive")
    rejects(_normalize_voxel_size_um, "2.6,2.6,32", "three numbers")

    # The retired spelling is still read, reversed -- but never alongside a
    # voxel_size_um that contradicts it.
    assert _config_voxel_size_um({"display_scale_zyx": [32.0, 2.6, 2.6]}) == [2.6, 2.6, 32.0]
    assert _config_voxel_size_um({"voxel_size_um": [2.6, 2.6, 32.0],
                                  "display_scale_zyx": [32.0, 2.6, 2.6]}) == [2.6, 2.6, 32.0]
    assert _config_voxel_size_um({}) is None
    rejects(lambda cfg: _config_voxel_size_um(cfg),
            {"voxel_size_um": [2.6, 2.6, 32.0], "display_scale_zyx": [25.0, 25.0, 25.0]},
            "retired")

    # Sections: the mode you are NOT running is dropped, not merged -- that is
    # what lets both sections stay filled in.
    sectioned = {"mode": "guide",
                 "common": {"image_path": "raw.tif", "voxel_size_um": [2.6, 2.6, 32.0]},
                 "guide": {"existing_mask_path": "prev.nii.gz"},
                 "labels": {"labels_path": "labels.nii.gz", "image_path": "WRONG.tif"}}
    assert flatten_config_sections(sectioned, "guide") == {
        "image_path": "raw.tif", "voxel_size_um": [2.6, 2.6, 32.0],
        "existing_mask_path": "prev.nii.gz"}, flatten_config_sections(sectioned, "guide")
    assert flatten_config_sections(sectioned, "labels") == {
        "image_path": "WRONG.tif", "voxel_size_um": [2.6, 2.6, 32.0],
        "labels_path": "labels.nii.gz"}
    # A flat config (every config predates the sections) still reads, and the
    # more specific place wins over the top level.
    assert flatten_config_sections({"image_path": "raw.tif", "mode": "guide"}, "guide") == \
        {"image_path": "raw.tif"}
    assert flatten_config_sections(
        {"output_path": "top.nii.gz", "common": {"output_path": "common.nii.gz"}},
        "guide") == {"output_path": "common.nii.gz"}
    rejects(lambda cfg: flatten_config_sections(cfg, "guide"),
            {"common": ["image_path"]}, "mapping")

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


def selftest_assignment_rows():
    print("20. assignment panel rows: an emptied brush label is reported, not dropped")
    structures = _fake_ontology()
    assignment = {1: [101, 10], 3: []}

    rows = assignment_rows(assignment, structures)
    assert [label for label, _regions in rows] == [1, 3], rows
    assert [sid for sid, _name in rows[0][1]] == [101, 10], rows
    assert all(name for _sid, name in rows[0][1]), "a row must carry the region's name"
    assert rows[1][1] == [], "an emptied label still gets a row of its own"

    # THE point: a label whose regions were all removed is still visible, so
    # "I painted with 3 and it exports as nothing" is caught in the panel
    # rather than in guide_export_warnings after the export.
    assert empty_assignment_labels(assignment) == [3]
    assert empty_assignment_labels({1: [101]}) == []
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


def _labels_interpolator():
    """The real multi-label interpolator, or None when registration_ants is
    not importable here (the mode-labels tests are then skipped rather than
    re-implemented: interpolate_sparse_label_correction is 30 lines of
    per-label signed-distance contest and a second copy would drift)."""
    try:
        return _interpolate_sparse_label_correction()
    except ImportError:
        return None


def selftest_plane_keyframes_are_whole_planes():
    print("13. mode labels: a plane that differs anywhere becomes a WHOLE keyframe")
    baseline = np.zeros((6, 8, 8), dtype=np.uint8)
    baseline[:, :4, :] = 1
    baseline[:, 4:, :] = 2
    paint = baseline.copy()
    paint[3, 3, 3] = 2                      # one pixel repainted on one plane

    keyframes = plane_keyframes(paint, baseline)
    assert list(keyframes) == [3], list(keyframes)
    mask, plane = keyframes[3]
    assert mask.all(), "the keyframe must cover the whole plane, not just the edit"
    # ...and it carries the regions that were NOT edited, which is the point:
    # the guide describes that plane's whole anatomy.
    assert set(np.unique(plane)) == {1, 2}, np.unique(plane)
    assert not plane_keyframes(baseline, baseline), "an untouched volume has no keyframes"
    print("   ok")


def selftest_labels_export_sparse_vs_dense():
    print("14. mode labels: sparse guide is bounded by the keyframes, dense one is not")
    interp = _labels_interpolator()
    if interp is None:
        print("   skipped (no registration_ants in this env)")
        return
    baseline = np.zeros((10, 12, 12), dtype=np.uint8)
    baseline[:, :6, :] = 1
    baseline[:, 6:, :] = 2
    paint = baseline.copy()
    for z in (2, 6):
        paint[z, 5, :] = 2                  # move the 1/2 boundary on two planes

    result = labels_export(paint, baseline, interpolate=interp)
    assert result.hand_drawn_slices == [2, 6], result.hand_drawn_slices

    # sparse: empty before the first and after the last keyframe, filled between
    assert not result.guide[:2].any(), "planes before the first keyframe must stay empty"
    assert not result.guide[7:].any(), "planes after the last keyframe must stay empty"
    assert result.guide[4].any(), "planes between two keyframes must be interpolated"

    # dense: every plane carries the registration, keyframes carry the edit
    assert result.atlas[0].any() and result.atlas[9].any(), "the dense volume has no empty planes"
    assert np.array_equal(result.atlas[0], baseline[0]), \
        "a plane outside the keyframe span must be the untouched registration"
    assert np.array_equal(result.atlas[2], paint[2]), "a keyframe must survive verbatim"
    assert np.array_equal(result.guide[2], paint[2]), "...in both outputs"

    assert set(result.slices_by_label) == {1, 2}, result.slices_by_label
    assert result.slices_by_label[1] == [2, 6], result.slices_by_label
    print("   ok")


def selftest_recollapse_keeps_edits():
    print("15. mode labels: expanding refines the untouched pixels, keeps the edited ones")
    old = np.zeros((3, 4, 4), dtype=np.uint8)
    old[:] = 1                              # everything was "cortex"
    new = old.copy()
    new[:, :2, :] = 8                       # expand: half of it is now "cortical plate"

    paint = old.copy()
    paint[1, 0, 0] = 3                      # a hand correction, inside the refined half
    paint[1, 3, 3] = 3                      # ...and one outside it

    out = recollapse_keeping_edits(paint, old, new)
    assert out[1, 0, 0] == 3 and out[1, 3, 3] == 3, "hand edits must survive an expand"
    assert out[0, 0, 0] == 8, "untouched pixels must pick up the finer label"
    assert out[1, 0, 1] == 8, "...including on a plane that was edited elsewhere"
    assert out[2, 3, 3] == 1, "and stay coarse where the expand does not reach"

    # The plane stays a keyframe afterwards, i.e. the expand did not silently
    # drop it from the export.
    assert list(plane_keyframes(out, new)) == [1], list(plane_keyframes(out, new))
    print("   ok")


def selftest_labels_sidecar_roundtrip(tmp_dir):
    print("16. mode labels: resume restores only the hand-drawn planes, and the true baseline")
    interp = _labels_interpolator()
    if interp is None:
        print("   skipped (no registration_ants in this env)")
        return
    structures = {
        10: {"name": "cortex", "structure_id_path": [1, 10]},
        100: {"name": "plate", "structure_id_path": [1, 10, 100]},
        20: {"name": "cerebellum", "structure_id_path": [1, 20]},
    }
    partition = label_partition.Partition.from_region_ids({1: [10], 2: [20]}, structures)
    partition.groups[3] = label_partition.Group(3, [100], "plate", parent=partition.groups[1])

    baseline = np.full((8, 6, 6), 1, dtype=np.uint8)
    baseline[:, 3:, :] = 2
    paint = baseline.copy()
    paint[2, 2, :] = 3
    paint[5, 2, :] = 3
    result = labels_export(paint, baseline, interpolate=interp)

    atlas_path = Path(tmp_dir) / "s_corrected_atlas.nii.gz"
    out = sitk.GetImageFromArray(result.atlas)
    sitk.WriteImage(out, str(atlas_path))
    sidecar = write_labels_sidecar(atlas_path, Path(tmp_dir) / "s_guide.nii.gz",
                                   Path(tmp_dir) / "s_labels_in_sample.nii.gz",
                                   result, partition, structures, 8)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["hand_drawn_slices"] == [2, 5], meta["hand_drawn_slices"]
    assert meta["baseline_labels_path"].endswith("s_labels_in_sample.nii.gz"), meta
    assert meta["parents"] == {"3": 1}, meta["parents"]
    assert meta["atlas_exclude_ids"] == {"1": [100]}, meta["atlas_exclude_ids"]

    resumed = load_labels_resume(atlas_path, structures, (8, 6, 6))
    assert resumed.hand_drawn_slices == [2, 5], resumed.hand_drawn_slices
    assert set(resumed.planes) == {2, 5}, "an interpolated plane must NOT come back as hand-drawn"
    assert np.array_equal(resumed.planes[2], paint[2]), "a restored keyframe must be exact"
    assert resumed.partition.children_of(1)[0].ids == (100,), "the nesting must survive"
    assert resumed.partition.atlas_exclude_ids(structures) == {1: [100]}
    print("   ok")


def selftest_yaml_snippet_carries_exclusions():
    print("17. mode labels: the emitted config block includes atlas_exclude_ids")
    snippet = guide_regions_yaml_snippet(
        {1: [688], 8: [695]}, {1: ["Cerebral cortex"], 8: ["Cortical plate"]},
        "/tmp/guide.nii.gz", voxel_size_um=[25.0, 25.0, 25.0],
        atlas_exclude_ids={1: [695]})
    assert "atlas_exclude_ids:" in snippet, snippet
    assert "1: [695]" in snippet, snippet
    assert "voxel_size_um: [25.0, 25.0, 25.0]" in snippet, snippet
    # guide mode must be unaffected: no exclusions, no block.
    plain = guide_regions_yaml_snippet({1: [688]}, {1: ["Cerebral cortex"]}, "/tmp/g.nii.gz")
    assert "atlas_exclude_ids" not in plain, plain
    print("   ok")


def selftest_labels_voxel_size():
    print("18. mode labels: the labels' voxel size, um vs mm vs neither")
    # This codebase's own outputs carry microns directly...
    assert labels_voxel_size_um((25.0, 25.0, 25.0)) == [25.0, 25.0, 25.0]
    # ...files from elsewhere (the DevCCF downloads) are in millimetres.
    assert labels_voxel_size_um((0.02, 0.02, 0.02)) == [20.0, 20.0, 20.0]
    assert labels_voxel_size_um((0.02, 0.02, 0.02), override=[25, 25, 25]) == [25.0, 25.0, 25.0]
    try:
        labels_voxel_size_um((1.0, 1.0, 1.0))
    except ValueError as exc:
        assert "labels_voxel_size_um" in str(exc), exc
    else:
        raise AssertionError("a (1,1,1) header must refuse to be guessed, not be read as 1 um")
    print("   ok")


def selftest_regrid_nearest():
    print("19. mode labels: regridding the registration onto the raw stack")
    # A fine grid at 25 um iso vs the raw stack's 32 um along z / 2.6 um in
    # plane -- i.e. z is COARSER on the raw grid and x/y much finer, which is
    # the real s12t geometry and the direction that catches an axis mix-up.
    fine_spacing = [25.0, 25.0, 25.0]          # (z, y, x)
    raw_spacing = [32.0, 2.6, 2.6]             # (z, y, x)

    # label == the fine voxel's own index along each axis, so a regridded
    # voxel can be checked against the index its physical position implies.
    fine = np.zeros((8, 6, 5), dtype=np.uint16)
    for z in range(8):
        for y in range(6):
            for x in range(5):
                fine[z, y, x] = z * 100 + y * 10 + x

    raw_shape = (6, 50, 40)
    out = regrid_nearest(fine, fine_spacing, raw_shape, raw_spacing)
    assert out.shape == raw_shape, out.shape
    assert out.dtype == fine.dtype, "a regrid must not change the label dtype"

    for z, y, x in ((0, 0, 0), (3, 20, 17), (5, 49, 39)):
        want = (min(7, int(round(z * 32.0 / 25.0))) * 100
                + min(5, int(round(y * 2.6 / 25.0))) * 10
                + min(4, int(round(x * 2.6 / 25.0))))
        assert out[z, y, x] == want, f"at {(z, y, x)}: {out[z, y, x]} != {want}"

    # Out-of-range destination voxels clamp to the edge rather than wrapping:
    # the raw stack legitimately extends past what was registered (crop_for_
    # registration), and a wrap would paste the far side of the brain there.
    tall = regrid_nearest(fine, fine_spacing, (40, 6, 5), fine_spacing)
    assert np.array_equal(tall[-1], fine[-1]), "past the end must clamp, not wrap"

    # Identity when the grids match, whatever the spacing.
    same = regrid_nearest(fine, fine_spacing, fine.shape, fine_spacing)
    assert np.array_equal(same, fine)
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
    selftest_seed_assignment()
    selftest_assignment_rows()

    # mode: labels -- painting on a registration result (see that section above)
    selftest_plane_keyframes_are_whole_planes()
    selftest_labels_export_sparse_vs_dense()
    selftest_recollapse_keeps_edits()
    selftest_yaml_snippet_carries_exclusions()
    selftest_labels_voxel_size()
    selftest_regrid_nearest()
    with tempfile.TemporaryDirectory() as tmp:
        selftest_labels_sidecar_roundtrip(tmp)
    print("=== all selftests passed ===")
    print("(shared/atlas_reference.py --selftest / tools/atlas_view.py --selftest cover atlas loading, "
          "ontology maths and the ortho-view geometry)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Paint a guide outline on a sample volume")
    local_config.add_config_arg(parser, "paint_mask")
    parser.add_argument("--selftest", action="store_true",
                        help="run the built-in synthetic tests (no GUI, no config) and exit")
    args_cli = parser.parse_args()

    if args_cli.selftest:
        return run_selftests()

    cfg = _load_local_config(args_cli.config)
    if cfg.mode == "labels":
        _run_labels(cfg)
    else:
        _run_guide(SimpleNamespace(
            image_path=cfg.image_path, output_path=cfg.output_path,
            existing_mask=cfg.existing_mask_path,
            region_labels=cfg.region_labels, region_ids=cfg.region_ids,
            voxel_size_um=cfg.voxel_size_um, atlas=cfg.atlas))

    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
