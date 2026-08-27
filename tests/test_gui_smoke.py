"""Smoke tests for the Qt/napari wiring the other tests cannot reach.

Everything else in tests/ (and every `--selftest` in this repo) is deliberately
headless: pure numpy in, pure numpy out. That covers the export maths well and
the panels not at all -- whether clicking Expand actually recollapses the paint
layer, whether the Export button writes its five files, whether the fill/outline
checkbox reaches the layers it is supposed to. Those are exactly the parts that
break when napari changes, so they get a test that really builds the windows.

No display needed to RUN it: with no $DISPLAY this re-execs itself under
xvfb-run (installed on the Linux box the tools are used from). Without either,
it skips with a message rather than failing -- a checkout on a machine with no
GUI stack should not report a red test for something it cannot run.

    conda activate antsreg
    python tests/test_gui_smoke.py

All-synthetic, like the other tests here: the ontology, the atlas annotation,
the raw stack and the registration output are all built below, so nothing
depends on ../Registration_ants/atlas/ being present.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ensure_display():
    """True if a GUI can be opened, re-execing under xvfb-run if that is what
    it takes. False means "skip", not "fail".

    On Linux a local Xvfb is preferred even when $DISPLAY is already set,
    which is the opposite of what it looks like it should do. The reason is
    the normal way these tools get used: over ssh with X11 forwarding, where
    $DISPLAY is something like localhost:10.0 and OpenGL is tunnelled back to
    the client. That display advertises GL 1.4 and napari's shaders do not
    compile against it -- the process dies in vispy with "Shader compilation
    error", nowhere near anything this test is checking. Xvfb + llvmpipe
    gives GL 2.1+ locally, and since nothing here ever LOOKS at the window,
    an invisible working context beats a visible broken one.

    Set GUI_SMOKE_USE_DISPLAY=1 to force the current $DISPLAY instead (for
    watching the windows, or on a machine whose GL is fine).
    """
    if os.environ.get("_GUI_SMOKE_UNDER_XVFB") or os.environ.get("GUI_SMOKE_USE_DISPLAY"):
        return bool(os.environ.get("DISPLAY")) or sys.platform in ("win32", "darwin")
    if sys.platform.startswith("linux"):
        xvfb = shutil.which("xvfb-run")
        if xvfb:
            # -a picks a free server number, so parallel runs don't collide.
            os.execve(xvfb, [xvfb, "-a", "-s", "-screen 0 1280x1024x24", sys.executable,
                             str(Path(__file__).resolve())],
                      dict(os.environ, _GUI_SMOKE_UNDER_XVFB="1"))
        return False
    return bool(os.environ.get("DISPLAY")) or sys.platform in ("win32", "darwin")


# =====================================================================================
# synthetic inputs
# =====================================================================================
# The shape that matters is CCFv3's: several levels deep, and only the LEAVES
# carry voxels in the annotation. That is what makes a fully-expanded parent's
# atlas side come out empty, which Partition.empty_atlas_side exists to catch.
_ONTOLOGY = {"msg": [{
    "id": 997, "name": "root", "acronym": "root", "children": [{
        "id": 8, "name": "Basic cell groups and regions", "acronym": "grey", "children": [
            {"id": 567, "name": "Cerebrum", "acronym": "CH", "children": [
                {"id": 688, "name": "Cerebral cortex", "acronym": "CTX", "children": [
                    {"id": 695, "name": "Cortical plate", "acronym": "CTXpl", "children": [
                        {"id": 315, "name": "Isocortex", "acronym": "Isocortex", "children": []},
                        {"id": 1089, "name": "Hippocampal formation", "acronym": "HPF",
                         "children": [
                             {"id": 1080, "name": "Hippocampal region", "acronym": "HIP",
                              "children": []},
                             {"id": 822, "name": "Retrohippocampal region", "acronym": "RHP",
                              "children": []}]}]},
                    {"id": 703, "name": "Cortical subplate", "acronym": "CTXsp",
                     "children": []}]}]},
            {"id": 1129, "name": "Interbrain", "acronym": "IB", "children": []}]}]}]}

# Grid geometry copied from the real s12t case, just small: the registration
# runs on an isotropic 25 um grid, the raw stack is 2.6 um in plane and 32 um
# along z. z COARSER and x/y much finer is the direction that catches an axis
# mix-up in the regrid, so it is worth keeping in a toy.
# Only leaves carry voxels, as in a real annotation.
LEAF_IDS = (315, 1080, 822, 703, 1129)

RAW_SHAPE = (8, 200, 220)          # (z, y, x)
RAW_VOXEL_UM = [2.6, 2.6, 32.0]    # (x, y, z), as the pipeline config states it
LABELS_SHAPE = (11, 22, 24)
LABELS_VOXEL_UM = [25.0, 25.0, 25.0]


def _write_inputs(tmp):
    import SimpleITK as sitk
    import tifffile

    ontology = tmp / "ontology.json"
    ontology.write_text(json.dumps(_ONTOLOGY), encoding="utf-8")

    # Annotation: leaves only, in slabs, so every group has voxels to find.
    # Written as NIfTI, not TIFF: shared.atlas_reference reads through
    # SimpleITK, and a 3D uint32 TIFF written by tifffile does NOT survive
    # that round trip (verified -- it comes back as thousands of garbage
    # ids). Real atlases are fine either way; a toy one has to be checked,
    # which is what _assert_annotation_loaded below does.
    annotation = np.zeros((40, 40, 40), dtype=np.uint32)
    for i, leaf in enumerate(LEAF_IDS):
        annotation[i * 8:(i + 1) * 8] = leaf
    annotation_path = tmp / "annotation.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(annotation), str(annotation_path))

    raw_path = tmp / "sample_registration.tif"
    rng = np.random.default_rng(0)
    tifffile.imwrite(str(raw_path), (rng.random(RAW_SHAPE) * 1000).astype(np.uint16))

    # The "registration output": the same leaves, in slabs along z, so the
    # collapsed canvas has several groups on most planes.
    labels = np.zeros(LABELS_SHAPE, dtype=np.uint32)
    for i, leaf in enumerate(LEAF_IDS):
        labels[:, :, i * 4:(i + 1) * 4] = leaf
    labels_sitk = sitk.GetImageFromArray(labels)
    # Microns directly, which is what this codebase's own outputs carry (see
    # paint_mask.labels_voxel_size_um) -- NOT the NIfTI millimetre convention.
    labels_sitk.SetSpacing(tuple(LABELS_VOXEL_UM))
    labels_path = tmp / "sample_labels_in_sample.nii.gz"
    sitk.WriteImage(labels_sitk, str(labels_path))

    partition_path = tmp / "seed.regions.json"
    partition_path.write_text(json.dumps({
        "region_ids": {"1": [688], "2": [1129]},
        "regions": {"1": ["Cerebral cortex"], "2": ["Interbrain"]},
    }), encoding="utf-8")

    return SimpleNamespace(ontology=ontology, annotation=annotation_path, raw=raw_path,
                           labels=labels_path, partition=partition_path)


def _assert_annotation_loaded(inputs):
    """The toy atlas must come back with exactly the ids it was written with.

    Without this the suite can pass on a misread annotation: a garbled one
    still happens to contain a few of the right ids, so the partition still
    expands, and the only visible sign is a nonsense count in the [atlas]
    line nobody reads.
    """
    from shared import atlas_reference
    atlas = atlas_reference.load_atlas_reference(
        atlas_reference.atlas_reference_config({
            "atlas_annotation_path": str(inputs.annotation),
            "ontology_path": str(inputs.ontology),
            "atlas_resolution_um": 25}), include_template=False)
    got = {int(i) for i in atlas.present_ids} - {0}
    assert got == set(LEAF_IDS), (
        f"the toy annotation did not survive the SimpleITK round trip: wrote {LEAF_IDS}, "
        f"read back {sorted(got)[:12]}{'...' if len(got) > 12 else ''}")
    assert len(atlas.structures) == 11, len(atlas.structures)


def _widget(viewer, name_fragment):
    """The inner widget of the dock whose name contains `name_fragment`."""
    docks = viewer.window._dock_widgets
    matches = [k for k in docks if name_fragment in k]
    assert matches, f"no dock matching {name_fragment!r}; have {sorted(docks)}"
    return docks[matches[0]].widget()


def _button(widget, text_fragment):
    import paint_mask as pm
    matches = [b for b in widget.findChildren(pm.QPushButton) if text_fragment in b.text()]
    assert matches, f"no button matching {text_fragment!r}"
    return matches[0]


# =====================================================================================
# paint_mask.py -- mode: guide
# =====================================================================================
def test_guide_mode_window(tmp, inputs):
    print("1. paint_mask mode: guide -- window, fill/outline switch, export...")
    import napari
    import paint_mask as pm

    from shared import atlas_reference

    out = tmp / "guide.nii.gz"
    # With an atlas, so _add_ontology_picker really builds the "Atlas /
    # Ontology" dock -- the panel this covers is the one that gets used, and
    # atlas=None quietly skips it.
    pm._run_guide(SimpleNamespace(
        image_path=str(inputs.raw), output_path=str(out), existing_mask=None,
        region_labels={1: ["Isocortex"]}, region_ids={}, voxel_size_um=None,
        atlas=atlas_reference.atlas_reference_config({
            "atlas_annotation_path": str(inputs.annotation),
            "ontology_path": str(inputs.ontology),
            "atlas_resolution_um": 25})))
    viewer = napari.current_viewer()
    try:
        paint = viewer.layers["guide outline (paint here)"]
        assert [l.name for l in viewer.layers] == ["sample", "guide outline (paint here)"]
        assert any("Ontology" in k for k in viewer.window._dock_widgets), \
            "the ontology picker did not build its panel"

        # Regions are FILLED by default in both modes -- napari's own default,
        # never overridden here, and the thing single_sample.py was changed to
        # agree with.
        checkbox = _widget(viewer, "Display").findChildren(pm.QCheckBox)[0]
        assert paint.contour == 0, "a region layer must start filled, not as an outline"
        checkbox.setChecked(True)
        assert paint.contour == 1, "the outline switch did not reach the paint layer"
        checkbox.setChecked(False)
        assert paint.contour == 0

        paint.data[2, 40:120, 40:120] = 1
        paint.data[6, 60:140, 60:140] = 1
        paint.refresh()
        _button(_widget(viewer, "Guide Outline Export"), "Export").click()

        assert out.exists(), "Export Outline wrote no volume"
        sidecar = json.loads((tmp / "guide.regions.json").read_text())
        assert sidecar["annotated_slices"] == {"1": [2, 6]}, sidecar["annotated_slices"]
    finally:
        viewer.close()
    print("   OK")


# =====================================================================================
# paint_mask.py -- mode: labels
# =====================================================================================
def test_labels_mode_window(tmp, inputs):
    print("2. paint_mask mode: labels -- raw-grid canvas, partition panel, export...")
    import napari
    import SimpleITK as sitk
    import paint_mask as pm
    from shared import atlas_reference

    out = tmp / "corrected_guide.nii.gz"
    pm._run_labels(SimpleNamespace(
        mode="labels", image_path=str(inputs.raw), output_path=str(out),
        labels_path=str(inputs.labels), atlas_output_path=None,
        partition_path=str(inputs.partition), min_region_mm3=0.0,
        voxel_size_um=RAW_VOXEL_UM, labels_voxel_size_um=None,
        region_labels={}, region_ids={},
        atlas=atlas_reference.atlas_reference_config({
            "atlas_annotation_path": str(inputs.annotation),
            "ontology_path": str(inputs.ontology),
            "atlas_resolution_um": 25})))
    viewer = napari.current_viewer()
    try:
        paint = viewer.layers["regions (paint here)"]
        # THE point of the mode: the canvas is the raw stack, with the
        # registration regridded up onto it -- not the other way round.
        assert paint.data.shape == RAW_SHAPE, (
            f"the canvas is {paint.data.shape}, but painting must happen on the raw grid "
            f"{RAW_SHAPE} -- the isotropic grid's planes were never imaged")
        assert paint.data.any(), "the collapsed registration came out empty"
        assert "registration as-is (read-only)" in viewer.layers

        panel = _widget(viewer, "Partition")
        listing = panel.findChildren(pm.QListWidget)[0]
        assert listing.count() == 2, "the seed partition has two groups"

        listing.setCurrentRow(0)
        assert paint.selected_label == 1, "selecting a group must set the brush to its label"

        # Expand Cerebral cortex, then Cortical plate: per-region refinement,
        # with the parent kept as the residual each time.
        _button(panel, "Expand").click()
        rows = [listing.item(i).text() for i in range(listing.count())]
        assert any("Cortical plate" in r for r in rows), rows
        assert any("residual" in r for r in rows), "the expanded parent must stay as residual"
        listing.setCurrentRow([i for i, r in enumerate(rows) if "Cortical plate" in r][0])
        _button(panel, "Expand").click()
        assert any("Isocortex" in listing.item(i).text() for i in range(listing.count()))
        expanded = listing.count()

        # ...and merging puts it back, recursively.
        listing.setCurrentRow(0)
        _button(panel, "Merge").click()
        assert listing.count() == 2, "Merge must drop the whole subtree, not one level"
        _button(panel, "Expand").click()
        listing.setCurrentRow([i for i in range(listing.count())
                               if "Cortical plate" in listing.item(i).text()][0])
        _button(panel, "Expand").click()
        assert listing.count() == expanded

        iso_label = [int(listing.item(i).text().split()[0]) for i in range(listing.count())
                     if "Isocortex" in listing.item(i).text()][0]
        for z in (2, 6):
            plane = paint.data[z]
            ys, xs = np.nonzero(plane != 0)
            plane[ys[:len(ys) // 8], xs[:len(ys) // 8]] = iso_label
        paint.refresh()

        _button(_widget(viewer, "Correction Export"), "Export").click()

        atlas_out = tmp / "corrected_guide_atlas.nii.gz"
        for path in (out, atlas_out, tmp / "corrected_guide.regions.json",
                     tmp / "corrected_guide.annotated_slices.json",
                     tmp / "corrected_guide_atlas.keyframes.json"):
            assert path.exists(), f"Export wrote no {path.name}"

        guide = sitk.GetArrayFromImage(sitk.ReadImage(str(out)))
        dense = sitk.GetArrayFromImage(sitk.ReadImage(str(atlas_out)))
        assert guide.shape == RAW_SHAPE and dense.shape == RAW_SHAPE
        assert not guide[0].any() and not guide[-1].any(), \
            "the sparse guide must stay empty outside the keyframe span"
        assert dense[0].any() and dense[-1].any(), \
            "the dense volume must carry the registration on every plane"

        meta = json.loads((tmp / "corrected_guide_atlas.keyframes.json").read_text())
        assert meta["hand_drawn_slices"] == [2, 6], meta["hand_drawn_slices"]
        assert meta["baseline_labels_path"].endswith("sample_labels_in_sample.nii.gz")
        assert meta["grids"]["raw_shape_zyx"] == list(RAW_SHAPE)
        assert meta["grids"]["labels_shape_zyx"] == list(LABELS_SHAPE)
        # Nesting reached the sidecar, so the atlas-side subtraction survives a resume.
        assert meta["atlas_exclude_ids"], "a nested partition must record atlas_exclude_ids"
    finally:
        viewer.close()
    print("   OK")


def test_labels_mode_resume(tmp, inputs):
    print("3. paint_mask mode: labels -- reopening restores the keyframes and the partition...")
    import napari
    import paint_mask as pm
    from shared import atlas_reference

    out = tmp / "corrected_guide.nii.gz"
    pm._run_labels(SimpleNamespace(
        mode="labels", image_path=str(inputs.raw), output_path=str(out),
        labels_path=str(inputs.labels), atlas_output_path=None,
        partition_path=str(inputs.partition), min_region_mm3=0.0,
        voxel_size_um=RAW_VOXEL_UM, labels_voxel_size_um=None,
        region_labels={}, region_ids={},
        atlas=atlas_reference.atlas_reference_config({
            "atlas_annotation_path": str(inputs.annotation),
            "ontology_path": str(inputs.ontology),
            "atlas_resolution_um": 25})))
    viewer = napari.current_viewer()
    try:
        paint = viewer.layers["regions (paint here)"]
        listing = _widget(viewer, "Partition").findChildren(pm.QListWidget)[0]
        # The expanded partition came back from the sidecar, not the seed file.
        assert listing.count() > 2, "a resume must restore the partition it was saved with"

        # Exactly the hand-drawn planes differ from the baseline again -- the
        # interpolated ones in between were re-derived, not trusted.
        baseline = viewer.layers["registration as-is (read-only)"].data
        assert sorted(pm.plane_keyframes(paint.data, baseline)) == [2, 6], \
            "resume must restore the keyframes, and ONLY the keyframes"
    finally:
        viewer.close()
    print("   OK")


# =====================================================================================
# single_sample.py
# =====================================================================================
def test_single_sample_fill_switch():
    print("4. single_sample: regions filled by default, outline one click away...")
    import napari
    import single_sample

    viewer = napari.Viewer(show=False)
    try:
        viewer.add_labels(np.ones((4, 10, 10), dtype=np.uint32), name="Atlas Regions")
        viewer.add_labels(np.ones((4, 10, 10), dtype=np.uint32), name=">> Highlight Atlas <<")

        class Stub:
            REGION_LAYER_NAMES = single_sample.MainController.REGION_LAYER_NAMES
            _apply_region_contour = single_sample.MainController._apply_region_contour

        stub = Stub()
        stub.viewer = viewer
        stub.cb_outline = SimpleNamespace(isChecked=lambda: False)
        stub._apply_region_contour()
        assert viewer.layers["Atlas Regions"].contour == 0, \
            "brain regions must be FILLED by default -- outline-only hides which region is which"

        stub.cb_outline = SimpleNamespace(isChecked=lambda: True)
        stub._apply_region_contour()
        assert viewer.layers["Atlas Regions"].contour == 1
        # The search-result highlight deliberately does not follow: contouring
        # it would hide the thing the user just searched for.
        assert viewer.layers[">> Highlight Atlas <<"].contour == 0
    finally:
        viewer.close()
    print("   OK")


# =====================================================================================
# panel widths
# =====================================================================================
QT_DEFAULT_MAX = 16777215      # QWIDGETSIZE_MAX -- Qt's "no maximum"


def _code_only(path):
    """A file's executable text, with comments and strings dropped.

    A plain substring scan is not usable here: the very line that explains
    why setMaximumWidth must not come back contains the word
    setMaximumWidth, so the check would fail on its own documentation.
    """
    import tokenize

    with open(path, "rb") as handle:
        tokens = tokenize.tokenize(handle.readline)
        return "".join(tok.string for tok in tokens
                       if tok.type not in (tokenize.COMMENT, tokenize.STRING))


def _assert_resizable(widget, what):
    assert widget.minimumWidth() <= 1, (
        f"{what} has a minimum width of {widget.minimumWidth()} px, so the splitter cannot "
        f"be dragged narrower than that -- wrap it in ontology_tree_ui.shrinkable")
    assert widget.maximumWidth() == QT_DEFAULT_MAX, (
        f"{what} is capped at {widget.maximumWidth()} px and can never be widened past it -- "
        f"use ontology_tree_ui.set_dock_width for a STARTING width instead of setMaximumWidth")


def test_panels_are_resizable(tmp, inputs):
    print("5. both tools: every side panel can be dragged to any width...")
    import napari
    import paint_mask as pm
    from shared import atlas_reference, ontology_tree_ui

    # The helper itself, against a real napari dock.
    viewer = napari.Viewer(show=False)
    try:
        from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
        panel = QWidget()
        QVBoxLayout(panel).addWidget(QLabel("a caption long enough to pin a panel's minimum "
                                            "width if nothing stops it from doing so"))
        pinned = panel.minimumSizeHint().width()
        assert pinned > 100, f"the caption should have been wide ({pinned} px) before shrinking"
        ontology_tree_ui.shrinkable(panel)
        dock = viewer.window.add_dock_widget(panel, area="left", name="width probe")
        ontology_tree_ui.set_dock_width(dock, 400)
        _assert_resizable(panel, "a shrinkable panel")
    finally:
        viewer.close()

    # ...and every panel BOTH paint_mask modes actually build, the ontology
    # tree included -- it is the deepest widget here and the easiest to pin.
    pm._run_guide(SimpleNamespace(
        image_path=str(inputs.raw), output_path=str(tmp / "widths_guide.nii.gz"),
        existing_mask=None, region_labels={}, region_ids={}, voxel_size_um=None,
        atlas=atlas_reference.atlas_reference_config({
            "atlas_annotation_path": str(inputs.annotation),
            "ontology_path": str(inputs.ontology),
            "atlas_resolution_um": 25})))
    viewer = napari.current_viewer()
    try:
        docks = viewer.window._dock_widgets
        assert any("Ontology" in k for k in docks), sorted(docks)
        for name in docks:
            _assert_resizable(docks[name].widget(), f"paint_mask guide mode's {name!r} panel")
    finally:
        viewer.close()

    pm._run_labels(SimpleNamespace(
        mode="labels", image_path=str(inputs.raw), output_path=str(tmp / "widths.nii.gz"),
        labels_path=str(inputs.labels), atlas_output_path=None,
        partition_path=str(inputs.partition), min_region_mm3=0.0,
        voxel_size_um=RAW_VOXEL_UM, labels_voxel_size_um=None,
        region_labels={}, region_ids={},
        atlas=atlas_reference.atlas_reference_config({
            "atlas_annotation_path": str(inputs.annotation),
            "ontology_path": str(inputs.ontology),
            "atlas_resolution_um": 25})))
    viewer = napari.current_viewer()
    try:
        docks = viewer.window._dock_widgets
        assert "Partition" in docks, sorted(docks)
        for name in docks:
            _assert_resizable(docks[name].widget(), f"paint_mask labels mode's {name!r} panel")
    finally:
        viewer.close()

    # single_sample builds its docks inside MainController, which needs a real
    # sample directory to construct -- so its two panels are checked at the
    # source level instead. Narrow on purpose: the bug being guarded against is
    # specifically a width CAP creeping back in, which is what made the region
    # panel un-widenable and is a one-liner to reintroduce.
    code = _code_only(ROOT / "single_sample.py")
    assert "setMaximumWidth" not in code, (
        "single_sample.py caps a panel's width again -- that is exactly what stops the "
        "region/ontology panel being dragged wider. Use ontology_tree_ui.set_dock_width.")
    assert "ontology_tree_ui.shrinkable" in code
    print("   OK")


# =====================================================================================
# the assignment panel, and how the panels are laid out
# =====================================================================================
def _open_guide(pm, tmp, inputs, atlas_reference, name, **overrides):
    """Guide mode with an atlas, opened on the synthetic inputs."""
    args = dict(
        image_path=str(inputs.raw), output_path=str(tmp / name), existing_mask=None,
        region_labels={}, region_ids={}, voxel_size_um=None,
        atlas=atlas_reference.atlas_reference_config({
            "atlas_annotation_path": str(inputs.annotation),
            "ontology_path": str(inputs.ontology),
            "atlas_resolution_um": 25}))
    args.update(overrides)
    pm._run_guide(SimpleNamespace(**args))
    import napari
    return napari.current_viewer()


def test_assignment_panel_drops_one_region(tmp, inputs):
    print("6. paint_mask guide mode: drop ONE region off a label, and hear about an empty one...")
    import paint_mask as pm
    from shared import atlas_reference

    # Three regions on one brush label -- the case the panel exists for: a
    # label routinely carries a dozen, and taking one back out used to mean
    # finding it again in the 12-deep ontology tree above.
    viewer = _open_guide(pm, tmp, inputs, atlas_reference, "assign.nii.gz",
                         region_ids={1: [315, 1080, 822]})
    try:
        panel = _widget(viewer, "Ontology")
        tree = panel.findChild(pm.QTreeWidget, "assignment_tree")
        note = panel.findChild(pm.QLabel, "assignment_empty_note")
        assert tree is not None and note is not None, "the assignment panel is not a tree"

        head = tree.topLevelItem(0)
        assert tree.topLevelItemCount() == 1 and head.childCount() == 3, \
            "one brush label with its three regions listed under it"
        assert not note.isVisible(), "nothing is empty yet"

        # One region, straight off the list.
        [child] = [head.child(i) for i in range(head.childCount())
                   if head.child(i).data(1, pm.Qt.UserRole) == 1080]
        child.setSelected(True)
        _button(panel, "Remove selected").click()
        head = tree.topLevelItem(0)
        left = {head.child(i).data(1, pm.Qt.UserRole) for i in range(head.childCount())}
        assert left == {315, 822}, f"only the selected region should have gone, left {left}"

        # ...and the rest, which must NOT make the label disappear silently:
        # something is probably still painted with it, and an outline with no
        # region cannot be paired with an atlas structure downstream.
        head.setSelected(True)
        _button(panel, "Remove selected").click()
        head = tree.topLevelItem(0)
        assert head.childCount() == 0 and "NO REGION" in head.text(0), head.text(0)
        assert note.isVisible(), "an emptied brush label must say so, not vanish"
        assert "1" in note.text()

        # Removing an already-empty label is how the reminder is dismissed.
        head.setSelected(True)
        _button(panel, "Remove selected").click()
        assert not note.isVisible(), "removing an empty label again forgets it"
    finally:
        viewer.close()
    print("   OK")


def test_panels_are_tabbed_and_short(tmp, inputs):
    print("7. paint_mask guide mode: one tab bar per side, layer controls free to shrink...")
    import paint_mask as pm
    from PyQt5.QtWidgets import QScrollArea
    from shared import atlas_reference

    viewer = _open_guide(pm, tmp, inputs, atlas_reference, "tabs.nii.gz")
    try:
        window = viewer.window._qt_window
        qt_viewer = viewer.window._qt_viewer
        docks = viewer.window._dock_widgets

        # Left: napari's own two docks and the ontology panel share one tab bar.
        ontology = docks[[k for k in docks if "Ontology" in k][0]]
        left = set(window.tabifiedDockWidgets(qt_viewer.dockLayerControls))
        assert ontology in left and qt_viewer.dockLayerList in left, (
            "the left panels are stacked, not tabbed -- three docks down one column arrive "
            "as slivers")

        # Right: every panel this tool adds there, in one more tab bar.
        export = docks[[k for k in docks if "Export" in k][0]]
        right = set(window.tabifiedDockWidgets(export))
        for name in ("Relabel", "Erase", "Display"):
            assert docks[name] in right, f"{name} is not tabbed with the export panel"

        # And the complaint that made a stacked column unusable in the first
        # place: the layer controls stop shrinking while they still own half
        # the column. Wrapped in a scroll area, the floor is the explicit
        # minimum rather than a dozen rows of controls.
        controls = qt_viewer.dockLayerControls
        assert isinstance(controls.widget(), QScrollArea), \
            "layer controls must scroll, or shrinking them just clips the rows off"
        assert controls.widget().minimumSizeHint().height() <= 80, \
            f"the layer controls still demand {controls.widget().minimumSizeHint().height()} px"
        assert controls.minimumHeight() <= 48, controls.minimumHeight()
    finally:
        viewer.close()
    print("   OK")


def main():
    print("=== tests/test_gui_smoke.py ===")
    if not _ensure_display():
        print("SKIPPED: no way to open a working GL context -- this test builds real Qt "
              "windows.\n"
              "         Linux: sudo apt install xvfb (preferred even over a forwarded\n"
              "         $DISPLAY, whose GL is usually too old for napari's shaders).\n"
              "         To force the current display anyway: GUI_SMOKE_USE_DISPLAY=1")
        return 0

    import paint_mask  # noqa: F401  -- fail loudly here if the env is wrong
    paint_mask._import_gui()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inputs = _write_inputs(tmp)
        _assert_annotation_loaded(inputs)
        test_guide_mode_window(tmp, inputs)
        test_labels_mode_window(tmp, inputs)
        test_labels_mode_resume(tmp, inputs)
        test_single_sample_fill_switch()
        test_panels_are_resizable(tmp, inputs)
        test_assignment_panel_drops_one_region(tmp, inputs)
        test_panels_are_tabbed_and_short(tmp, inputs)
    print("\nALL GUI SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
