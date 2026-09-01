"""Smoke tests for the Qt/napari wiring the other tests cannot reach.

Everything else in tests/ (and every `--selftest` in this repo) is deliberately
headless: pure numpy in, pure numpy out. That covers the export maths well and
the panels not at all -- whether picking a region in the ontology tree actually
highlights it and recollapses the paint layer, whether the Export button writes
its five files, whether the fill/outline
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
RAW_UM = [2.6, 2.6, 32.0]          # (x, y, z) microns, as a real raw stack has
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


def _tree_item(tree, structure_id):
    """The ontology tree's row for one structure id.

    By id, not by name: the row text carries the acronym and the voxel count
    too, and matching on a name substring is exactly the ambiguity the whole
    ids-not-names convention exists to avoid.
    """
    import paint_mask as pm
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        item = stack.pop()
        if item.data(0, pm.Qt.UserRole) == structure_id:
            return item
        stack += [item.child(i) for i in range(item.childCount())]
    raise AssertionError(f"no tree row for structure {structure_id}")


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
        image_path=str(inputs.raw), output_path=str(out), existing_mask_path=None, damage_labels=[],
        region_labels={1: ["Isocortex"]}, region_ids={}, voxel_size_um=None,
        atlas=atlas_reference.atlas_reference_config({
            "atlas_annotation_path": str(inputs.annotation),
            "ontology_path": str(inputs.ontology),
            "atlas_resolution_um": 25})))
    viewer = napari.current_viewer()
    try:
        paint = viewer.layers["guide outline (paint here)"]
        # The two painting layers come first and in this order; the Reposition
        # section's three follow, hidden, and are asserted by name rather than
        # by an exact list so adding a panel layer is not a test edit.
        names = [l.name for l in viewer.layers]
        assert names[:2] == ["sample", "guide outline (paint here)"], names
        assert set(names[2:]) == {pm._REPOSITION_FRAGMENTS_LAYER,
                                  pm._REPOSITION_GHOST_LAYER}, names
        assert not any(viewer.layers[n].visible for n in names[2:]), \
            "reposition layers must start hidden -- most samples never cracked"
        assert viewer.layers.selection == {viewer.layers["guide outline (paint here)"]}, \
            "the region brush stays selected, not the fragments layer added after it"
        assert any("Ontology" in k for k in viewer.window._dock_widgets), \
            "the ontology picker did not build its panel"

        tools = _widget(viewer, "Export & tools")
        # WHICH sample is open, on screen rather than only in the config: the
        # window title carries its name, and the banner pinned above the tools
        # sections carries the paths behind it (with the full path as the
        # tooltip, since the dock is narrower than they are).
        assert "sample_registration" in viewer.title, viewer.title
        banner = [(l.text(), l.toolTip()) for l in tools.findChildren(pm.QLabel)]
        assert ("sample_registration", str(inputs.raw)) in banner, banner
        assert any(text.startswith("image:") and tip == str(inputs.raw)
                   for text, tip in banner), banner
        assert any(text.startswith("export:") and tip == str(out)
                   for text, tip in banner), banner
        assert not any(text.startswith("resume:") for text, _ in banner), \
            "no existing_mask_path was given, so that row must be dropped, not shown empty"

        # Regions are FILLED by default in both modes -- napari's own default,
        # never overridden here, and the thing single_sample.py was changed to
        # agree with.
        checkbox = [b for b in tools.findChildren(pm.QCheckBox)
                    if "outline only" in b.text()][0]
        assert paint.contour == 0, "a region layer must start filled, not as an outline"
        checkbox.setChecked(True)
        assert paint.contour == 1, "the outline switch did not reach the paint layer"
        checkbox.setChecked(False)
        assert paint.contour == 0

        paint.data[2, 40:120, 40:120] = 1
        paint.data[6, 60:140, 60:140] = 1
        paint.refresh()
        _button(_widget(viewer, "Export & tools"), "Export Outline").click()

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
    from shared import atlas_reference, hover_bar

    out = tmp / "corrected_guide.nii.gz"
    pm._run_labels(SimpleNamespace(
        mode="labels", image_path=str(inputs.raw), output_path=str(out),
        labels_path=str(inputs.labels), dense_output_path=None, resume_from=None,
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

        # Same banner as guide mode, plus the registration being corrected --
        # "which file of which sample" is the question this mode most needs
        # answered before an hour of painting (see pm._sample_banner).
        assert "sample_registration" in viewer.title, viewer.title
        banner = [(l.text(), l.toolTip())
                  for l in _widget(viewer, "Export & tools").findChildren(pm.QLabel)]
        assert ("sample_registration", str(inputs.raw)) in banner, banner
        assert any(text.startswith("registration:") and tip == str(inputs.labels)
                   for text, tip in banner), banner
        assert any(text.startswith("dense out:") and tip.endswith("corrected_guide_atlas.nii.gz")
                   for text, tip in banner), banner

        panel = _widget(viewer, "Partition")
        listing = panel.findChildren(pm.QListWidget)[0]
        assert listing.count() == 2, "the seed partition has two groups"

        tree = panel.findChild(pm.QTreeWidget, "partition_tree")
        listing.setCurrentRow(0)
        assert paint.selected_label == 1, "selecting a group must set the brush to its label"
        assert tree.currentItem().data(0, pm.Qt.UserRole) == 688, \
            "selecting a group must take the tree to the region it stands for"

        def rows():
            return [listing.item(i).text() for i in range(listing.count())]

        def label_of(name):
            return [int(r.split()[0]) for r in rows() if name in r][0]

        def pick(structure_id):
            tree.setCurrentItem(_tree_item(tree, structure_id))

        # Picking a region: three levels below the seed partition's Cerebral
        # cortex, i.e. where expanding could only get in three rounds. It
        # highlights on the sample and says which label owns it today,
        # WITHOUT touching the partition.
        highlight = viewer.layers["selected region (atlas pick)"]
        pick(315)                                        # Isocortex
        assert highlight.visible and int(highlight.data.sum()) > 0, \
            "picking a region must light it up where the registration put it"
        assert highlight.data.shape == LABELS_SHAPE, \
            "the highlight belongs on the registration's own grid, not the raw stack's"
        assert listing.count() == 2, "picking a region must not change the partition"
        assert paint.selected_label == 1, "...but it does set the brush to the label that owns it"

        # Splitting it out is the one click that does: its own label, the
        # parent kept as the residual, the paint layer recollapsed under it.
        _button(panel, "own brush label").click()
        assert any("Isocortex" in r for r in rows()), rows()
        assert any("residual" in r for r in rows()), "the parent must stay as the residual"
        iso_label = label_of("Isocortex")
        assert (paint.data == iso_label).any(), "the split must recollapse the paint layer"

        # A second, deeper pick under the same parent -- then Merge drops the
        # whole subtree that came out of label 1, not one level of it.
        pick(1080)                                       # Hippocampal region
        _button(panel, "own brush label").click()
        assert listing.count() == 4, rows()
        listing.setCurrentRow(0)
        _button(panel, "Merge").click()
        assert listing.count() == 2, "Merge must drop everything split out of the label"
        assert not (paint.data == iso_label).any(), "merging must recollapse the paint layer too"

        # Remove: split_out's own undo, for a group with nothing under it.
        pick(315)
        _button(panel, "own brush label").click()
        dropped = label_of("Isocortex")
        listing.setCurrentRow([i for i, r in enumerate(rows()) if "Isocortex" in r][0])
        _button(panel, "Remove this label").click()
        assert listing.count() == 2, rows()
        assert not (paint.data == dropped).any(), \
            "removing a label must give its voxels back to the group above it"

        pick(315)
        _button(panel, "own brush label").click()
        iso_label = label_of("Isocortex")

        # The whole annotation, not just the partition's handful of groups,
        # and in atlas_view's OWN colour indices so the two tools agree.
        regions = viewer.layers["atlas regions (all, read-only)"]
        assert regions.data.shape == LABELS_SHAPE, \
            "the region reference belongs on the registration's grid"
        assert viewer.layers.index(regions) < viewer.layers.index(paint), \
            "the reference must sit UNDER the paint layer, or it hides what you draw"
        atlas_ref = atlas_reference.load_atlas_reference(
            atlas_reference.atlas_reference_config({
                "atlas_annotation_path": str(inputs.annotation),
                "ontology_path": str(inputs.ontology),
                "atlas_resolution_um": 25}), include_template=False)
        # (z, y, x) = (5, 10, 1) is in the first slab, i.e. Isocortex (315).
        assert int(regions.data[5, 10, 1]) == atlas_ref.index_of_id[315], \
            "the layer must hold compact present_ids indices, which is what makes the "\
            "colours identical to tools/atlas_view.py's"

        # ...and the bottom bar reads that region's whole ancestor chain off
        # it, in the colour the layer is drawing it in. Driven through the
        # viewer's own callback, world coordinates and all, because a
        # LAYER-level callback is delivered to the active layer only -- which
        # is exactly why this panel used to sit there empty.
        bar = _widget(viewer, "Region under cursor")
        # Width decides how many levels fit, and an off-screen window's docks are
        # narrow -- so give the bar the room a real window would.
        bar.resize(1400, 64)
        assert viewer.mouse_move_callbacks, "nothing is watching the cursor"
        on_move = viewer.mouse_move_callbacks[0]
        on_move(viewer, SimpleNamespace(position=(5 * 25.0, 10 * 25.0, 1 * 25.0)))
        text = bar.text()
        assert "Isocortex" in text and "Cerebral cortex" in text, text
        assert text.index("Cerebral cortex") < text.index("Isocortex"), \
            "the chain runs root -> leaf, shallowest first"
        rgba = regions.colormap.map(np.array([atlas_ref.index_of_id[315]]))[0]
        assert hover_bar.colours(rgba)[0] in bar.styleSheet(), \
            "the strip must be painted the region's own colour"
        # The id rides along so it can be pasted straight back into the
        # search box below -- which is the other half of this:
        assert "[315]" in text, text
        search = panel.findChild(pm.QLineEdit, "partition_search")
        search.setText("315")
        assert not _tree_item(tree, 315).isHidden(), "searching an id must find its region"
        assert _tree_item(tree, 1080).isHidden(), "...and hide the ones it does not match"
        search.setText("isocortex 315")      # terms are ANDed, in any order
        assert not _tree_item(tree, 315).isHidden(), "name + id together must still match"
        search.setText("315 nonesuch")
        assert _tree_item(tree, 315).isHidden(), "every term has to match, not just one"
        search.setText("")

        # Off the volume: the bar says so instead of holding the last region.
        on_move(viewer, SimpleNamespace(position=(1e6, 1e6, 1e6)))
        assert "Isocortex" not in bar.text(), bar.text()
        for z in (2, 6):
            plane = paint.data[z]
            ys, xs = np.nonzero(plane != 0)
            plane[ys[:len(ys) // 8], xs[:len(ys) // 8]] = iso_label
        paint.refresh()

        # A voxel that was actually corrected says so in the bar, on top of
        # the atlas chain -- that is what tells a correction apart from a
        # region left alone, once the whole plane counts as a keyframe.
        zyx = np.argwhere(paint.data != viewer.layers["registration as-is (read-only)"].data)[0]
        scale = RAW_VOXEL_UM[::-1]
        on_move(viewer, SimpleNamespace(position=tuple(c * s for c, s in zip(zyx, scale))))
        assert "REPAINTED" in bar.text(), bar.text()

        # A cracked piece, grabbed and left unposed -- test 4 reopens this
        # export under new names and has to find it again.
        frag = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER]
        frag_data = frag.data.copy()
        frag_data[2, 20:40, 20:40] = 2
        frag.data = frag_data

        _button(_widget(viewer, "Export & tools"), "Export Guide + Atlas").click()

        atlas_out = tmp / "corrected_guide_atlas.nii.gz"
        for path in (out, atlas_out, tmp / "corrected_guide.regions.json",
                     tmp / "corrected_guide.annotated_slices.json",
                     tmp / "corrected_guide_atlas.keyframes.json",
                     tmp / "corrected_guide.reposition.json",
                     tmp / "corrected_guide_fragments.nii.gz"):
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
        labels_path=str(inputs.labels), dense_output_path=None, resume_from=None,
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
def test_labels_mode_resume_from(tmp, inputs):
    print("4. paint_mask mode: labels -- resume_from reads one archive, writes another...")
    import napari
    import paint_mask as pm
    from shared import atlas_reference

    # Round 1's archive, written by tests 2/3. Resuming from it while exporting
    # under new names must leave it exactly as it was -- that is the whole
    # point of splitting the read path off the write path.
    round1 = tmp / "corrected_guide_atlas.nii.gz"
    round1_sidecar = tmp / "corrected_guide_atlas.keyframes.json"
    before = (round1.read_bytes(), round1_sidecar.read_bytes())

    out2 = tmp / "round2.nii.gz"
    dense2 = tmp / "round2_atlas.nii.gz"
    pm._run_labels(SimpleNamespace(
        mode="labels", image_path=str(inputs.raw), output_path=str(out2),
        labels_path=str(inputs.labels), dense_output_path=str(dense2),
        resume_from=str(round1),
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
        baseline = viewer.layers["registration as-is (read-only)"].data
        assert sorted(pm.plane_keyframes(paint.data, baseline)) == [2, 6], \
            "resume_from must restore round 1's keyframes"
        assert _widget(viewer, "Partition").findChildren(pm.QListWidget)[0].count() > 2, \
            "resume_from must restore round 1's partition too"
        # ...and the fragments, which are the case a new output name used to
        # lose: round 1's plan sits next to round 1's GUIDE stem, and nothing
        # in this config names it -- the dense file's sidecar does, via the
        # guide_path it recorded when it was written.
        assert (viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER].data == 2).any(), \
            "the fragment grabbed in round 1 did not survive the new output names"

        _button(_widget(viewer, "Export & tools"), "Export Guide + Atlas").click()
        for path in (out2, dense2, tmp / "round2_atlas.keyframes.json",
                     tmp / "round2.regions.json"):
            assert path.exists(), f"Export wrote no {path.name}"
        # The archive that was READ is untouched, byte for byte.
        assert (round1.read_bytes(), round1_sidecar.read_bytes()) == before, \
            "resume_from must not write back to the file it resumed from"
        # ...and the new one still points at the registration as its baseline,
        # never at the dense file it was resumed from.
        meta = json.loads((tmp / "round2_atlas.keyframes.json").read_text())
        assert meta["baseline_labels_path"].endswith("sample_labels_in_sample.nii.gz"), meta
        assert meta["hand_drawn_slices"] == [2, 6], meta["hand_drawn_slices"]
    finally:
        viewer.close()
    print("   OK")


def test_single_sample_fill_switch():
    print("5. single_sample: regions filled by default, outline one click away...")
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
    print("6. both tools: every side panel can be dragged to any width...")
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
        existing_mask_path=None, damage_labels=[], region_labels={}, region_ids={}, voxel_size_um=None,
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
        labels_path=str(inputs.labels), dense_output_path=None, resume_from=None,
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
        image_path=str(inputs.raw), output_path=str(tmp / name), existing_mask_path=None, damage_labels=[],
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
    print("7. paint_mask guide mode: drop ONE region off a label, and hear about an empty one...")
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
    print("20. paint_mask guide mode: one left tab bar, one right panel, controls free to shrink...")
    import paint_mask as pm
    from PyQt5.QtWidgets import QScrollArea
    from shared import atlas_reference

    viewer = _open_guide(pm, tmp, inputs, atlas_reference, "tabs.nii.gz")
    try:
        window = viewer.window._qt_window
        qt_viewer = viewer.window._qt_viewer
        docks = viewer.window._dock_widgets

        # Left: the layer controls and this tool's tools panel share one tab
        # bar. The LAYER LIST is deliberately not in it -- _tab_the_panels
        # keeps it stacked below and always visible, since it is what you
        # check after painting to see which layers exist. (This assertion
        # used to demand the opposite and had gone stale against that change.)
        ontology = docks[[k for k in docks if "Ontology" in k][0]]
        left = set(window.tabifiedDockWidgets(qt_viewer.dockLayerControls))
        assert docks["Export & tools"] in left, (
            "the left panels are stacked, not tabbed -- three docks down one column arrive "
            "as slivers")
        assert qt_viewer.dockLayerList not in left, (
            "the layer list must stay stacked and visible, not hidden behind a tab")
        assert qt_viewer.dockLayerList.isVisible(), "the layer list should be shown"

        # ...and the region panel is NOT one of them: it owns the right column
        # on its own, which is what the left/right swap was for.
        assert ontology not in left, "the region panel must have a column to itself"
        assert not window.tabifiedDockWidgets(ontology), \
            "nothing should share the region panel's column"

        # The tools are ONE dock of foldable sections -- tabs hid three panels
        # behind labels that had to be remembered, and each is only a handful
        # of controls. The first section opens, the rest start folded.
        tools = docks["Export & tools"]
        headers = {b.text().strip("\u25be\u25b8 "): b
                   for b in tools.widget().findChildren(pm.QPushButton) if b.isCheckable()}
        for title in ("EXPORT", "RELABEL", "ERASE", "DISPLAY"):
            assert title in headers, f"{title} is not a section of the tools panel"
        assert headers["EXPORT"].isChecked(), "the export section should start open"
        assert not headers["ERASE"].isChecked(), "the rest should start folded"
        erase_btn = _button(tools.widget(), "Polygon erase")
        assert not erase_btn.isVisible(), "a folded section must not show its controls"
        headers["ERASE"].setChecked(True)
        assert erase_btn.isVisible(), "clicking a header must open its section"

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



def _grab_at(pm, viewer, tools, zyx):
    """Press the grab button, then click the image -- the real two-step path.

    The button only ARMS the grab; the click that decides what to take lands on
    the canvas afterwards. Driving it any other way would test a code path the
    hand never uses.
    """
    from types import SimpleNamespace
    _button(tools, "Grab this plane").click()
    scale = viewer.layers["sample"].scale
    armed = viewer.mouse_drag_callbacks[-1]
    armed(viewer, SimpleNamespace(position=tuple(c * s for c, s in zip(zyx, scale))))


def _viewer_plan(viewer, pm):
    """The reposition plan the panel currently holds, via the fragments
    layer's metadata."""
    return viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER].metadata["reposition_state"].plan()


def _pin_pose(tools):
    """Record the current pose on the current plane.

    There is no Set-keyframe button any more: the pose controls record as they
    are used. Pressing Enter in a box is the gesture that records a plane whose
    numbers did NOT change -- i.e. how an identity keyframe is pinned -- and it
    is also the deterministic one to drive from a test, since setValue with an
    unchanged value emits nothing at all.
    """
    import paint_mask as pm
    tools.findChild(pm.QDoubleSpinBox, "reposition_tx").editingFinished.emit()


def _reposition_section(pm, viewer):
    """The Reposition section's widget, and its four named pose controls."""
    tools = viewer.window._dock_widgets["Export & tools"].widget()
    headers = {b.text().strip("\u25be\u25b8 "): b
               for b in tools.findChildren(pm.QPushButton) if b.isCheckable()}
    assert "REPOSITION" in headers, f"no Reposition section; have {sorted(headers)}"
    headers["REPOSITION"].setChecked(True)          # a folded section hides its controls
    boxes = {key: tools.findChild(pm.QDoubleSpinBox, f"reposition_{key}")
             for key in ("tx", "ty", "rot", "dz", "feather")}
    boxes["fragment"] = tools.findChild(pm.QSpinBox, "reposition_fragment")
    missing = [k for k, v in boxes.items() if v is None]
    assert not missing, f"reposition controls not found: {missing}"
    return tools, boxes


def test_reposition_panel(tmp, inputs):
    print("8. paint_mask: the Reposition section in guide mode -- pose, keyframe, export, resume...")
    import paint_mask as pm
    from shared import atlas_reference
    from registration_ants import reposition as rp

    out = tmp / "repo.nii.gz"
    viewer = _open_guide(pm, tmp, inputs, atlas_reference, out.name, voxel_size_um=RAW_UM)
    try:
        tools, boxes = _reposition_section(pm, viewer)
        frag = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER]

        # Paint fragment 1 on two planes only, then let the panel fill the gap
        # -- the whole point of keyframed outlines is not tracing every plane.
        data = frag.data.copy()
        data[1, 12:22, 12:22] = 1
        data[5, 12:22, 12:22] = 1
        frag.data = data
        # NOT filled in the layer: the grabbed planes stay visible as the
        # keyframes they are, and the filling happens where the plan is used.
        assert not (frag.data[3] == 1).any(), \
            "the fragments layer densified itself; the keyframes would be unrecoverable"
        assert (rp.densify_plane(frag.data, 3) == 1).any(), "the fill is not reachable on demand"

        # THE POSE RECORDS ITSELF. Plane 1 is pinned unmoved by touching a
        # control and putting it back (nothing to change: it is already 0), and
        # plane 5 by simply moving the sliders -- no Set-keyframe button, which
        # was a step that could only be forgotten.
        assert not [b for b in tools.findChildren(pm.QPushButton)
                    if "Set keyframe" in b.text()], "the Set keyframe button is still there"
        def posed():
            return sorted(k["z"] for f in _viewer_plan(viewer, pm)["fragments"]
                          for k in f["keyframes"])

        viewer.dims.set_current_step(0, 1)
        boxes["tx"].setValue(0.0)
        boxes["rot"].setValue(0.0)
        assert posed() == [], "setValue with an unchanged value is not a pose being set"
        _pin_pose(tools)
        assert posed() == [1], "touching a control must pin this plane, identity included"

        viewer.dims.set_current_step(0, 5)
        boxes["tx"].setValue(50.0)
        boxes["rot"].setValue(6.0)
        assert posed() == [1, 5], "moving a slider must record the plane it was moved on"

        # ...and scrolling does NOT: the panel writes the interpolated pose into
        # the sliders on every plane change, and reading that back as an edit
        # would turn every plane looked at into a keyframe.
        viewer.dims.set_current_step(0, 3)
        viewer.dims.set_current_step(0, 5)
        assert posed() == [1, 5], "scrolling through the planes recorded one"

        plan = _viewer_plan(viewer, pm)
        assert len(plan["fragments"]) == 1, plan["fragments"]
        kfs = plan["fragments"][0]["keyframes"]
        assert [k["z"] for k in kfs] == [1, 5], kfs
        assert kfs[1]["tx_um"] == 50.0 and kfs[1]["theta_deg"] == 6.0, kfs[1]

        # Scrolling to an in-between plane must show what interpolation will
        # actually do there, not leave the last edited pose on screen.
        viewer.dims.set_current_step(0, 3)
        assert abs(boxes["tx"].value() - 25.0) < 0.05, boxes["tx"].value()

        _button(tools, "Boundary check").click()

        # A couple of painted planes as well, so this export writes a guide
        # volume too -- the next round below is opened ON it (existing_mask_
        # path), which is how a real second round starts.
        paint = viewer.layers["guide outline (paint here)"]
        paint.data[1, 30:60, 30:60] = 1
        paint.data[5, 30:60, 30:60] = 1
        paint.refresh()
        _button(tools, "Export Outline").click()
    finally:
        viewer.close()

    stem = pm._output_stem(str(out))
    plan_path = Path(f"{stem}.reposition.json")
    fragments_path = Path(f"{stem}_fragments.nii.gz")
    assert plan_path.exists() and fragments_path.exists(), "reposition export wrote nothing"
    written = rp.read_plan(plan_path)
    assert [k["z"] for k in written["fragments"][0]["keyframes"]] == [1, 5], written
    assert written["labels_path"] == str(fragments_path)

    # Reopening the same output path picks the work back up.
    viewer = _open_guide(pm, tmp, inputs, atlas_reference, out.name)
    try:
        tools, boxes = _reposition_section(pm, viewer)
        resumed = _viewer_plan(viewer, pm)
        assert [k["z"] for k in resumed["fragments"][0]["keyframes"]] == [1, 5], resumed
        assert (viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER].data == 1).any(), \
            "the fragment outlines did not come back"
    finally:
        viewer.close()

    # NEXT ROUND, NEW OUTPUT NAME -- how rounds are actually done (the config
    # template says to keep each round's export as a snapshot). The paint layer
    # is resumed from existing_mask_path, and the fragments have to follow it:
    # keyed on output_path alone they were left behind at the old stem, so a
    # second round opened with the pieces gone and a plan that moved nothing.
    round2 = tmp / "repo_round2.nii.gz"
    viewer = _open_guide(pm, tmp, inputs, atlas_reference, round2.name,
                         existing_mask_path=str(out))
    try:
        _reposition_section(pm, viewer)
        resumed = _viewer_plan(viewer, pm)
        assert [k["z"] for k in resumed["fragments"][0]["keyframes"]] == [1, 5], resumed
        assert (viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER].data == 1).any(), \
            "the fragment outlines did not follow the guide into the next round"
    finally:
        viewer.close()

    # GRABBED BUT NOT YET POSED. Separating the pieces is the slow half of the
    # work; it used to export nothing at all until the first keyframe existed,
    # so quitting after an afternoon of grabbing lost every one of them.
    grabbed = tmp / "grabbed.nii.gz"
    viewer = _open_guide(pm, tmp, inputs, atlas_reference, grabbed.name, voxel_size_um=RAW_UM)
    try:
        frag = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER]
        data = frag.data.copy()
        data[2, 12:22, 12:22] = 3
        data[2, 30:40, 30:40] = pm._REPOSITION_CUT_LABEL
        frag.data = data
        _button(_reposition_section(pm, viewer)[0], "Export Outline").click()
    finally:
        viewer.close()
    stem = pm._output_stem(str(grabbed))
    assert Path(f"{stem}.reposition.json").exists(), \
        "grabbed fragments with no keyframe yet exported nothing"
    saved = rp.read_plan(Path(f"{stem}.reposition.json"))
    assert [f["label"] for f in saved["fragments"]] == [3], saved["fragments"]
    assert saved["fragments"][0]["keyframes"] == [], "a grab is not a move"

    viewer = _open_guide(pm, tmp, inputs, atlas_reference, grabbed.name, voxel_size_um=RAW_UM)
    try:
        back = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER].data
        assert (back == 3).any(), "the grabbed-but-unposed fragment did not come back"
        assert not (back == pm._REPOSITION_CUT_LABEL).any(), \
            "cuts steer a grab and must not be shipped or restored"
    finally:
        viewer.close()

    # A sample that never cracked must not acquire an empty plan file next to
    # every guide it exports.
    clean = tmp / "clean.nii.gz"
    viewer = _open_guide(pm, tmp, inputs, atlas_reference, clean.name)
    try:
        tools, _ = _reposition_section(pm, viewer)
        _button(tools, "Export Outline").click()
    finally:
        viewer.close()
    assert not Path(f"{pm._output_stem(str(clean))}.reposition.json").exists(), \
        "an untouched Reposition section still wrote a plan"
    print("   OK")



def test_reposition_grab_from_click(tmp, inputs):
    print("9. paint_mask: grabbing a piece from a click, cleanly across the crack...")
    import numpy as np
    import tifffile
    import paint_mask as pm
    from shared import atlas_reference

    # A slab with a flap along its top edge, separated by a two-voxel crack on
    # planes 0..5 and welded on from plane 6 -- a hinge, the geometry the whole
    # per-plane model exists for.
    stack = np.full((8, 200, 220), 50, dtype=np.uint16)
    stack[:, 80:180, 20:200] = 3000                 # the brain
    stack[:, 20:70, 60:160] = 3000                  # the flap
    stack[6:, 70:80, 60:160] = 3000                 # ...welded on from z=6
    raw = tmp / "flap_stack.tif"
    tifffile.imwrite(str(raw), stack)

    viewer = _open_guide(pm, tmp, inputs, atlas_reference, "grab.nii.gz",
                         image_path=str(raw), voxel_size_um=RAW_UM)
    try:
        tools, boxes = _reposition_section(pm, viewer)
        frag = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER]
        assert not frag.data.any(), "nothing should be outlined before the grab"

        scale = viewer.layers["sample"].scale
        _grab_at(pm, viewer, tools, (3, 45, 110))

        got = frag.data == 1
        planes = sorted(int(z) for z in np.unique(np.nonzero(got)[0]))
        assert planes == [3], f"a grab takes exactly the plane clicked, got {planes}"
        assert got[3, 20:70, 60:160].all(), "the flap itself was not taken"
        assert not got[3, 80:180, 20:200].any(), "the grab leaked across the crack into the brain"

        # Clicking off tissue is a mistake worth naming, and must not wipe the
        # outline already grabbed.
        _grab_at(pm, viewer, tools, (3, 2, 2))
        assert (frag.data == 1).any(), "a bad click destroyed the previous grab"
    finally:
        viewer.close()
    print("   OK")



def test_reposition_three_pieces_each_move_their_own_way(tmp, inputs):
    print("11. paint_mask: cortex split in three -- three fragments closing toward the middle...")
    import numpy as np
    import tifffile
    import paint_mask as pm
    from shared import atlas_reference

    # One band of "cortex" cut into three by two gaps. Each piece is a third of
    # the tissue on its plane -- fine, since a grab takes the component clicked
    # and asks no question about how big a piece is allowed to be.
    stack = np.full((8, 200, 220), 50, dtype=np.uint16)
    for x0, x1 in ((10, 70), (76, 136), (142, 202)):
        stack[:, 60:140, x0:x1] = 3000
    raw = tmp / "three_pieces.tif"
    tifffile.imwrite(str(raw), stack)

    viewer = _open_guide(pm, tmp, inputs, atlas_reference, "three.nii.gz",
                         image_path=str(raw), voxel_size_um=RAW_UM)
    try:
        tools, boxes = _reposition_section(pm, viewer)
        frag = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER]
        scale = viewer.layers["sample"].scale

        # Left and right pieces move inward; the middle one is the anchor and
        # is never grabbed -- tissue that split did not shrink, so something
        # has to hold still or closing the gaps just makes the cortex smaller.
        for label, x in ((1, 40), (2, 172)):
            boxes["fragment"].setValue(label)
            _grab_at(pm, viewer, tools, (3, 100, x))

        got = frag.data
        assert (got == 1)[3, 100, 40] and (got == 2)[3, 100, 172], "a piece was not grabbed"
        assert not (got == 1)[3, 100, 100], "the grab crossed the gap into the middle piece"
        assert not (got == 2)[3, 100, 100], "the grab crossed the gap into the middle piece"
        assert not got[3, 100, 100], "the anchor piece must stay unclaimed"

        # Each carries its own transform, in its own direction.
        viewer.dims.set_current_step(0, 3)
        for label, tx in ((1, 6.0), (2, -6.0)):
            boxes["fragment"].setValue(label)
            boxes["tx"].setValue(tx)
            _pin_pose(tools)
        plan = _viewer_plan(viewer, pm)
        moves = {f["label"]: f["keyframes"][0]["tx_um"] for f in plan["fragments"]}
        assert moves == {1: 6.0, 2: -6.0}, moves
    finally:
        viewer.close()
    print("   OK")



def test_reposition_refuses_a_plan_with_no_voxel_size(tmp, inputs):
    print("12. paint_mask: no voxel_size_um -> the panel opens but will not export a plan...")
    import numpy as np
    import paint_mask as pm
    from pathlib import Path
    from shared import atlas_reference

    # _open_guide already passes voxel_size_um=None, which is legal for painting
    # a guide (it only sets the display aspect) and NOT legal for a plan: the
    # offsets would be voxel counts wearing a micron label, self-consistent on
    # the image and the wrong scale on the cells.
    out = tmp / "novoxel.nii.gz"
    viewer = _open_guide(pm, tmp, inputs, atlas_reference, out.name)
    try:
        tools, boxes = _reposition_section(pm, viewer)
        assert any("VOXELS, not microns" in w.text()
                   for w in tools.findChildren(pm.QLabel)), "no warning about the missing scale"

        frag = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER]
        data = frag.data.copy()
        data[2, 5:25, 5:25] = 1
        frag.data = data
        viewer.dims.set_current_step(0, 2)
        boxes["tx"].setValue(5.0)
        _pin_pose(tools)

        # Exporting reports the refusal and writes the guide anyway -- the two
        # are independent, and an exception out of a Qt handler would abort the
        # process instead of saying anything.
        assert pm._export_reposition(frag.metadata["reposition_state"], str(out)) == []
        _button(tools, "Export Outline").click()
    finally:
        viewer.close()
    assert not Path(f"{pm._output_stem(str(out))}.reposition.json").exists()
    print("   OK")



def test_reposition_single_plane_grab_keeps_pieces_apart(tmp, inputs):
    print("14. paint_mask: per-plane grab for pieces that overlap in xy at different z...")
    import numpy as np
    import tifffile
    import paint_mask as pm
    from shared import atlas_reference

    # Two pieces at the SAME xy, far apart in z. Nothing infers an extent here:
    # each piece's planes are the ones clicked, so the two cannot run together.
    stack = np.full((8, 200, 220), 50, dtype=np.uint16)
    stack[:, 90:180, 20:200] = 3000            # the brain, on every plane
    stack[1:3, 20:70, 60:160] = 3000           # piece A, low z
    stack[5:7, 20:70, 60:160] = 3000           # piece B, same xy, high z
    raw = tmp / "two_pieces.tif"
    tifffile.imwrite(str(raw), stack)

    viewer = _open_guide(pm, tmp, inputs, atlas_reference, "twopieces.nii.gz",
                         image_path=str(raw), voxel_size_um=RAW_UM)
    try:
        tools, boxes = _reposition_section(pm, viewer)
        frag = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER]
        scale = viewer.layers["sample"].scale

        def grab(label, z):
            boxes["fragment"].setValue(label)
            viewer.dims.set_current_step(0, z)
            _grab_at(pm, viewer, tools, (z, 45, 110))

        # Fragment 1 is piece A: grab its first and last plane. The second grab
        # must NOT wipe the first -- per-plane grabs accumulate.
        grab(1, 1)
        grab(1, 2)
        planes1 = sorted(int(z) for z in np.unique(np.nonzero(frag.data == 1)[0]))
        assert planes1 == [1, 2], f"per-plane grabs did not accumulate: {planes1}"

        # Fragment 2 is piece B, at the same xy. Different number, no collision.
        grab(2, 5)
        grab(2, 6)
        planes2 = sorted(int(z) for z in np.unique(np.nonzero(frag.data == 2)[0]))
        assert planes2 == [5, 6], planes2
        assert not (frag.data == 1)[5].any(), "piece B was claimed by fragment 1"
        assert not (frag.data == 2)[1].any(), "piece A was claimed by fragment 2"

        # A third fragment seeded on piece A's own plane: same xy as the other
        # two, and still nobody else's. A grab's extent is exactly the plane
        # clicked -- nothing here infers how far a piece goes.
        boxes["fragment"].setValue(3)
        viewer.dims.set_current_step(0, 1)
        _grab_at(pm, viewer, tools, (1, 45, 110))
        assert sorted(int(z) for z in np.unique(np.nonzero(frag.data == 3)[0])) == [1]
    finally:
        viewer.close()
    print("   OK")



def test_old_plans_drop_their_line_pairs(tmp, inputs):
    print("22. paint_mask: a plan written when lines existed loads and re-exports without them...")
    import numpy as np
    import SimpleITK as sitk
    import paint_mask as pm
    from shared import atlas_reference
    from registration_ants import reposition as rp

    out = tmp / "oldplan.nii.gz"
    stem = pm._output_stem(str(out))
    fragments = np.zeros(RAW_SHAPE, dtype=np.uint8)
    fragments[2, 10:40, 10:60] = 1
    fragments[5, 10:40, 10:60] = 1
    sitk.WriteImage(sitk.GetImageFromArray(fragments), f"{stem}_fragments.nii.gz")

    # The shape the two-line fit used to leave behind: provenance hung off each
    # keyframe, recording the pair it was fitted from.
    keyframes = []
    for z, tx in ((2, 0.0), (5, 40.0)):
        keyframe = rp.make_keyframe(z, tx_um=tx, center_um=(100.0, 100.0))
        keyframe["segments"] = {"source": [[10.0, 10.0], [90.0, 10.0]],
                                "target": [[50.0, 10.0], [130.0, 10.0]],
                                "source_z": z, "target_z": z}
        keyframes.append(keyframe)
    plan = rp.make_plan(RAW_SHAPE, RAW_UM, [rp.make_fragment(1, keyframes, "old flap")],
                        labels_path=f"{stem}_fragments.nii.gz")
    rp.write_plan(f"{stem}.reposition.json", plan)

    viewer = _open_guide(pm, tmp, inputs, atlas_reference, out.name, voxel_size_um=RAW_UM)
    try:
        assert not [l for l in viewer.layers if "segment" in l.name], \
            f"a segments layer came back: {[l.name for l in viewer.layers]}"
        resumed = _viewer_plan(viewer, pm)["fragments"][0]
        assert [k["z"] for k in resumed["keyframes"]] == [2, 5], resumed
        assert resumed["name"] == "old flap", "the fragment's name did not survive the resume"
        assert not any("segments" in k for k in resumed["keyframes"]), \
            "the line pair was carried back into the plan this session holds"
        _button(_widget(viewer, "Export & tools"), "Export Outline").click()
    finally:
        viewer.close()

    written = rp.read_plan(f"{stem}.reposition.json")
    assert [k["z"] for k in written["fragments"][0]["keyframes"]] == [2, 5], written
    assert not any("segments" in k for k in written["fragments"][0]["keyframes"]), \
        "the re-exported plan still carries the line pairs"
    print("   OK")


def test_reposition_drag_the_outline(tmp, inputs):
    print("21. paint_mask: copy the fragment outline, drag it, read the pose back...")
    import numpy as np
    import paint_mask as pm
    from shared import atlas_reference

    viewer = _open_guide(pm, tmp, inputs, atlas_reference, "ghost.nii.gz", voxel_size_um=RAW_UM)
    try:
        tools, boxes = _reposition_section(pm, viewer)
        frag = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER]
        ghost = viewer.layers[pm._REPOSITION_GHOST_LAYER]
        boxes["fragment"].setValue(2)

        # Nothing grabbed on this plane: it says so rather than copying air.
        viewer.dims.set_current_step(0, 2)
        _button(tools, "Copy this outline").click()
        assert len(ghost.data) == 0, "a ghost was made from an empty plane"

        data = frag.data.copy()
        data[2, 10:40, 10:60] = 2
        data[2, 10:20, 60:75] = 2          # asymmetric, so a rotation is identifiable
        frag.data = data

        _button(tools, "Copy this outline").click()
        assert len(ghost.data) == 1 and ghost.visible, "the outline was not copied"
        assert ghost.mode == "select" and ghost.selected_data == {0}, \
            "the ghost should come back selected, ready to drag"
        assert int(ghost.features["fragment"].iloc[0]) == 2, ghost.features
        before = np.asarray(ghost.data[0], dtype=float)
        assert len(before) >= 3, f"a polygon needs 3+ vertices, got {len(before)}"

        # Move it exactly as a drag+rotate would, and read the pose back. The
        # numbers have to come out as the ones that were applied -- this is the
        # whole contract of the outline path.
        tx_um, ty_um, deg = 130.0, -52.0, 11.0
        rot = np.array([[np.cos(np.radians(deg)), -np.sin(np.radians(deg))],
                        [np.sin(np.radians(deg)), np.cos(np.radians(deg))]])
        xy = np.column_stack([before[:, 2] * RAW_UM[0], before[:, 1] * RAW_UM[1]])
        pivot = xy.mean(axis=0)
        moved = (xy - pivot) @ rot.T + pivot + [tx_um, ty_um]
        ghost.data = [np.column_stack([before[:, 0],
                                       moved[:, 1] / RAW_UM[1], moved[:, 0] / RAW_UM[0]])]

        # No button: moving the ghost is what sets the pose.
        assert not [b for b in tools.findChildren(pm.QPushButton)
                    if "Fit from the dragged" in b.text()], "the Fit button is still there"
        assert abs(boxes["rot"].value() - deg) < 0.2, boxes["rot"].value()
        # tx/ty are written down about the fitted centre, so check the MOTION
        # rather than the two numbers it is expressed as: the drag has already
        # recorded the keyframe, so ask the plan itself where the outline lands.
        from registration_ants import reposition as rp
        keyframe = _viewer_plan(viewer, pm)["fragments"][0]["keyframes"][0]
        assert np.abs(rp.transform_points_um(xy, keyframe) - moved).max() < 0.5, \
            "the fitted transform does not carry the outline where it was dragged"

        # A resize is not a rigid move: reported, and NOT folded into the pose.
        stretched = (moved - moved.mean(axis=0)) * 1.2 + moved.mean(axis=0)
        ghost.data = [np.column_stack([before[:, 0],
                                       stretched[:, 1] / RAW_UM[1], stretched[:, 0] / RAW_UM[0]])]
        said = [w.text() for w in tools.findChildren(pm.QLabel) if "resized" in w.text()]
        assert said, "a 20% resize of the ghost went unreported"
        assert "+20" in said[0], said[0]

        # Copying again replaces the ghost rather than piling them up.
        _button(tools, "Copy this outline").click()
        assert len(ghost.data) == 1, f"{len(ghost.data)} ghosts for one fragment"

        # MOVING THE PIVOT MUST NOT MOVE THE TISSUE. Pinning the hinge on a
        # piece already posed used to swing it by (R - I)(c_new - c_old), which
        # is the whole width of the sample for a far-away pivot.
        ghost.data = [np.column_stack([before[:, 0],
                                       moved[:, 1] / RAW_UM[1], moved[:, 0] / RAW_UM[0]])]
        landed = rp.transform_points_um(
            xy, _viewer_plan(viewer, pm)["fragments"][0]["keyframes"][0])
        viewer.cursor.position = tuple(np.array([2.0, 400.0, 900.0]) * np.array(RAW_UM[::-1]))
        _button(tools, "Rotation centre <- cursor").click()
        keyframe = _viewer_plan(viewer, pm)["fragments"][0]["keyframes"][0]
        assert abs(keyframe["center_um"][0] - 900.0 * RAW_UM[0]) < 1.0, keyframe["center_um"]
        assert np.abs(rp.transform_points_um(xy, keyframe) - landed).max() < 0.5, \
            "re-pivoting moved the tissue instead of only renaming the numbers"
    finally:
        viewer.close()
    print("   OK")


def test_reposition_export_stays_sparse(tmp, inputs):
    print("16. paint_mask: the exported outline keeps the grabbed planes, and reopens as them...")
    import numpy as np
    import SimpleITK as sitk
    from pathlib import Path
    import paint_mask as pm
    from shared import atlas_reference
    from registration_ants import reposition as rp

    out = tmp / "sparse.nii.gz"
    viewer = _open_guide(pm, tmp, inputs, atlas_reference, out.name, voxel_size_um=RAW_UM)
    try:
        tools, boxes = _reposition_section(pm, viewer)
        frag = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER]
        data = frag.data.copy()
        data[1, 5:25, 5:25] = 1
        data[5, 5:25, 5:25] = 1
        frag.data = data
        for z in (1, 5):
            viewer.dims.set_current_step(0, z)
            boxes["tx"].setValue(0.0 if z == 1 else 26.0)
            _pin_pose(tools)
        _button(tools, "Export Outline").click()
    finally:
        viewer.close()

    stem = pm._output_stem(str(out))
    written = sitk.GetArrayFromImage(sitk.ReadImage(f"{stem}_fragments.nii.gz"))
    planes = sorted(int(z) for z in np.unique(np.nonzero(written == 1)[0]))
    assert planes == [1, 5], f"the export was densified; keyframes are gone: {planes}"
    # ...and filling it on use covers the span, which is what the pipeline does.
    assert sorted(int(z) for z in np.unique(np.nonzero(
        rp.densify_fragments(written) == 1)[0])) == [1, 2, 3, 4, 5]

    # Reopening gives the grabbed planes back, not a solid block.
    viewer = _open_guide(pm, tmp, inputs, atlas_reference, out.name, voxel_size_um=RAW_UM)
    try:
        _reposition_section(pm, viewer)
        back = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER].data
        assert sorted(int(z) for z in np.unique(np.nonzero(back == 1)[0])) == [1, 5], \
            "a reopened session cannot tell a grabbed plane from a filled one"
    finally:
        viewer.close()
    print("   OK")



def test_reposition_grab_is_armed_then_clicked(tmp, inputs):
    print("17. paint_mask: the grab button arms a click, and the result becomes visible...")
    import numpy as np
    import tifffile
    from types import SimpleNamespace
    import paint_mask as pm
    from shared import atlas_reference

    stack = np.full((8, 200, 220), 50, dtype=np.uint16)
    stack[:, 90:180, 20:200] = 3000
    stack[2:6, 20:70, 60:160] = 3000
    raw = tmp / "armed.tif"
    tifffile.imwrite(str(raw), stack)

    viewer = _open_guide(pm, tmp, inputs, atlas_reference, "armed.nii.gz",
                         image_path=str(raw), voxel_size_um=RAW_UM)
    try:
        tools, boxes = _reposition_section(pm, viewer)
        frag = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER]
        assert not frag.visible, "the fragments layer should start hidden"

        before = len(viewer.mouse_drag_callbacks)
        _button(tools, "Grab this plane").click()
        # Pressing the button must NOT grab anything yet -- it arms a click.
        assert not frag.data.any(), "the button grabbed before anything was pointed at"
        assert len(viewer.mouse_drag_callbacks) == before + 1, "no click was armed"
        assert frag.visible and frag.mode == "pan_zoom", \
            "while armed the click must land as a click, not a brush stroke"

        viewer.dims.set_current_step(0, 3)
        scale = viewer.layers["sample"].scale
        armed = viewer.mouse_drag_callbacks[-1]
        armed(viewer, SimpleNamespace(position=tuple(c * s for c, s in zip((3, 45, 110), scale))))

        assert (frag.data == 1)[3, 45, 110], "the click did not grab"
        assert frag.visible, "a grab whose result cannot be seen reads as a dead button"
        assert len(viewer.mouse_drag_callbacks) == before, "the armed click was not one-shot"

        # A click on background says so, and leaves the previous grab alone.
        _button(tools, "Grab this plane").click()
        viewer.mouse_drag_callbacks[-1](
            viewer, SimpleNamespace(position=tuple(c * s for c, s in zip((3, 5, 5), scale))))
        assert (frag.data == 1).any(), "a bad click destroyed the previous grab"
        assert any("not on tissue" in w.text() for w in tools.findChildren(pm.QLabel)), \
            "a click on background said nothing"
    finally:
        viewer.close()
    print("   OK")



def test_reposition_grab_prefers_the_painted_mask(tmp, inputs):
    print("18. paint_mask: a grab takes the painted outline, not a threshold of the image...")
    import numpy as np
    import tifffile
    import paint_mask as pm
    from shared import atlas_reference

    # The image and the painted outline DISAGREE on purpose: the bright region
    # is wider than what was traced. A grab that reads the image comes back
    # with the wide shape, one that reads the paint comes back with the traced
    # one, and only the second is what someone who painted it expects.
    stack = np.full((8, 200, 220), 50, dtype=np.uint16)
    stack[:, 90:180, 20:200] = 3000
    stack[2:6, 20:70, 40:180] = 3000              # bright: x 40..180
    raw = tmp / "disagree.tif"
    tifffile.imwrite(str(raw), stack)

    viewer = _open_guide(pm, tmp, inputs, atlas_reference, "painted.nii.gz",
                         image_path=str(raw), voxel_size_um=RAW_UM)
    try:
        tools, boxes = _reposition_section(pm, viewer)
        paint = viewer.layers["guide outline (paint here)"]
        frag = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER]
        painted = paint.data.copy()
        painted[3, 20:70, 60:160] = 2             # traced narrower: x 60..160
        paint.data = painted

        switch = [c for c in tools.findChildren(pm.QCheckBox)
                  if "painted mask" in c.text()][0]
        assert switch.isChecked(), "the painted outline should be the default source"

        viewer.dims.set_current_step(0, 3)
        _grab_at(pm, viewer, tools, (3, 45, 110))
        got = frag.data[3] == 1
        assert got[45, 70] and not got[45, 45], \
            "the grab took the image's shape, not the outline that was painted"
        assert int(got.sum()) == int((painted[3] == 2).sum()), "the traced shape was not taken"

        # Untick and it reads the image instead -- the fallback for a plane
        # that was never painted, or an outline drawn across the crack.
        switch.setChecked(False)
        boxes["fragment"].setValue(2)
        _grab_at(pm, viewer, tools, (3, 45, 110))
        wide = frag.data[3] == 2
        assert wide[45, 45], "unticking did not fall back to the image"

        # And on a plane with no paint it falls back on its own, and says so.
        switch.setChecked(True)
        boxes["fragment"].setValue(3)
        viewer.dims.set_current_step(0, 4)
        _grab_at(pm, viewer, tools, (4, 45, 110))
        assert (frag.data[4] == 3).any(), "an unpainted plane grabbed nothing"
        assert any("nothing painted on plane 4" in w.text()
                   for w in tools.findChildren(pm.QLabel)), "the fallback was silent"
    finally:
        viewer.close()
    print("   OK")



def test_reposition_names_follow_their_fragment(tmp, inputs):
    print("19. paint_mask: a fragment's name stays with it when the number changes...")
    import numpy as np
    import paint_mask as pm
    from shared import atlas_reference

    viewer = _open_guide(pm, tmp, inputs, atlas_reference, "names.nii.gz", voxel_size_um=RAW_UM)
    try:
        tools, boxes = _reposition_section(pm, viewer)
        frag = viewer.layers[pm._REPOSITION_FRAGMENTS_LAYER]
        data = frag.data.copy()
        data[2, 5:25, 5:25] = 1
        data[2, 40:60, 5:25] = 2
        frag.data = data
        name_edit = tools.findChild(pm.QLineEdit, "reposition_name")
        assert name_edit is not None, "the fragment name field was not found"
        viewer.dims.set_current_step(0, 2)

        boxes["fragment"].setValue(1)
        name_edit.setText("flap A")
        _pin_pose(tools)

        # Switching must reload the box, or fragment 2 inherits "flap A" the
        # moment its keyframe is set -- two pieces with one name, in the plan
        # and in every QC line that quotes it.
        boxes["fragment"].setValue(2)
        assert name_edit.text() == "", f"the previous fragment's name stayed: {name_edit.text()!r}"
        name_edit.setText("flap B")
        _pin_pose(tools)

        boxes["fragment"].setValue(1)
        assert name_edit.text() == "flap A", name_edit.text()

        names = {f["label"]: f["name"] for f in _viewer_plan(viewer, pm)["fragments"]}
        assert names == {1: "flap A", 2: "flap B"}, names
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
        test_labels_mode_resume_from(tmp, inputs)
        test_single_sample_fill_switch()
        test_panels_are_resizable(tmp, inputs)
        test_assignment_panel_drops_one_region(tmp, inputs)
        test_reposition_panel(tmp, inputs)
        test_reposition_grab_from_click(tmp, inputs)
        test_reposition_three_pieces_each_move_their_own_way(tmp, inputs)
        test_reposition_refuses_a_plan_with_no_voxel_size(tmp, inputs)
        test_reposition_single_plane_grab_keeps_pieces_apart(tmp, inputs)
        test_reposition_export_stays_sparse(tmp, inputs)
        test_reposition_grab_is_armed_then_clicked(tmp, inputs)
        test_reposition_grab_prefers_the_painted_mask(tmp, inputs)
        test_reposition_names_follow_their_fragment(tmp, inputs)
        test_reposition_drag_the_outline(tmp, inputs)
        test_old_plans_drop_their_line_pairs(tmp, inputs)
        test_panels_are_tabbed_and_short(tmp, inputs)
    print("\nALL GUI SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
