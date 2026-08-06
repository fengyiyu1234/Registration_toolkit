"""Interactive tool: directly hand-correct the Allen/CCF region boundaries
that ANTs already warped into sample space (`<name>_labels_in_sample.nii.gz`),
on a sparse set of z-planes, then auto-interpolate the corrections into a
dense corrected label volume -- WITHOUT re-running registration.

This is a different tool from paint_mask.py's `mask`/`guide` kinds:
those two operate *before* registration (paint a guide, then re-run SyN so
the deformation goes and matches it). This one operates *after* -- you
already have labels_in_sample.nii.gz from a completed registration, you
directly repaint whichever region(s) came out wrong on a handful of
representative slices (using real CCF region ids, picked from the ontology),
and the correction is interpolated across the volume in sample space. No
re-registration involved -- see ../Registration_ants/scripts/relabel_cells.py
for the next step, which uses the corrected volume to fix cell-to-region
assignments in cell_registration.csv.

Usage (needs a display; the antsreg env has napari+PyQt5+SimpleITK alongside
antspyx -- one env for the whole pipeline):
    conda activate antsreg
    python edit_sample_labels.py
    # no CLI args -- a form window opens for the sample/labels/output paths
    # and the ontology options, pre-filled with whatever you used last time
    # (kept in .dialog_state/, gitignored).

Workflow:
    1. A napari window opens with the sample image and a Labels layer
       PRE-FILLED with the current labels_in_sample -- you're editing the
       existing (imperfect) registration result in place, not painting from
       scratch. Sliced along axis 0, the actual imaging z-planes (same
       SimpleITK convention as the other two paint scripts -- see their
       docstrings for why this matters).
    2. Pick a CCF ontology tree depth from the "Region picker level" dropdown
       (root = level 1). This re-collapses the labels layer ITSELF to that
       level (see atlas_utils.collapse_labels_to_level) -- the whole volume
       declutters to that level's boundaries, not just one region, and
       painting assigns that level's own ids. Switch levels anytime; voxels
       you've actually painted are preserved exactly, everything else is
       re-derived fresh from the original file at the new level. The region
       list below refreshes to just that level's structures.
    3. Pick a region from the list -- this sets the paint brush to that
       region's real CCF id. Check "Show only selected region" to hide every
       other region and isolate just this one for painting. Moving the
       mouse over the image shows which region is currently under the
       cursor, so you can tell what you're about to overwrite.
    4. Repaint only the area that's wrong, on a handful of representative
       z-planes (where the mismatch starts, ends, and where its shape
       changes a lot) -- leave everything else on those planes alone, and
       leave every other plane untouched entirely. This is a delta edit on
       top of the existing labels, not a full resegmentation. Cmd/Ctrl+Z
       (and Cmd/Ctrl+Shift+Z to redo) undoes/redoes paint strokes at any
       time -- note undo doesn't retroactively un-mark a voxel as edited for
       export purposes below, it only restores its value. Erasing to 0 and
       then picking a different region + using napari's own fill/bucket
       tool re-assigns that patch in one click, without repainting by hand.
    5. Click "Export Corrected Labels". This auto-detects which pixels you
       actually painted (tracked precisely, not by diffing against the
       original -- collapsing to a level already changes most values),
       interpolates the correction (not the whole plane) between edited
       planes using shape-aware per-region signed-distance blending, and
       writes a full corrected label volume on the exact same grid as the
       input -- ready for ../Registration_ants/scripts/relabel_cells.py.

Saving one region on its own (for Dice/HD95 ground truth): picking a region
in step 3 also isolates it (id + all its ontology descendants) into its own
binary `[isolate] <name>` layer, built fresh from whatever's currently in
labels_layer. Paint/erase on THAT layer, then click "Save This Region" to
write just `<sample>_<region_slug>_corrected_mask.nii.gz`, independent of
everything else -- this is how registration_eval.py's
`dice_region_masks` ground truth is meant to be built. Because it's keyed
by the exact structure id (not a name match), parent/child regions that
nest -- e.g. Cerebellum and Cerebellar cortex -- can each be corrected and
saved on their own without interfering with each other. Switching to a
different region resets this isolate layer, so save before you switch if
you want to keep it; painting on labels_layer itself (step 4) is unaffected
either way.

If you only hand-drew a handful of representative z-planes for a region
(the usual case -- see mask_utils.interpolate_sparse_label_correction's
docstring on why the planes in between are an interpolated guess, not
verified ground truth), "Save This Region" also writes a sidecar
`<...>_corrected_mask.annotated_slices.json` recording exactly which planes
you actually drew. registration_eval.py picks this up automatically and
restricts Dice/HD95 to just those planes -- you don't need to remember or
retype the z-indices yourself (see Config.use_annotation_hints there to
turn this off and always compare the full volume instead).

Adding more hand-drawn planes to a region later (same file, no versioning):
picking a region that already has a saved file + sidecar resumes from it --
only the exact planes recorded as hand-drawn get loaded back in (their real
values, not the whole file, since the rest of it may be an old interpolation
guess), everything else starts fresh from the true registration baseline.
Draw more planes and click "Save This Region" again: it re-interpolates
from the full union of old + new hand-drawn planes against that baseline,
overwriting the same file -- so interpolation error never compounds across
sessions, and nothing you've already hand-verified gets silently lost.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import napari
import numpy as np
import SimpleITK as sitk
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QLabel, QLineEdit, QListWidget, QPushButton,
    QShortcut, QVBoxLayout, QWidget,
)

# Resolved through the editable install of ../Registration_ants (pip install -e
# there puts it on sys.path for the antsreg env) -- no path hacking needed.
# atlas_utils/mask_utils are pure json/numpy/scipy at this import level
# (atlas_utils only imports ants lazily, inside get_allen_atlas).
from registration_ants import atlas_utils, mask_utils
from _form_dialog import run_form  # sibling module


_FORM_FIELDS = [
    {"key": "sample_path", "label": "Sample image (e.g. *_fine_25um.nii.gz)",
     "type": "open_file", "filter": "Images (*.nii *.nii.gz *.tif *.tiff);;All files (*)"},
    {"key": "labels_path", "label": "labels_in_sample.nii.gz",
     "type": "open_file", "filter": "Images (*.nii *.nii.gz);;All files (*)"},
    {"key": "output_path", "label": "Output corrected labels path",
     "type": "save_file", "filter": "NIfTI (*.nii.gz);;All files (*)"},
    {"key": "use_brainglobe", "label": "Use BrainGlobe atlas (instead of an ontology JSON)",
     "type": "checkbox", "default": False},
    {"key": "ontology_json", "label": "Ontology JSON",
     "type": "open_file", "filter": "JSON files (*.json);;All files (*)",
     "enabled_when": ("use_brainglobe", False)},
    {"key": "atlas_res_um", "label": "Atlas resolution, um (BrainGlobe only)",
     "type": "float", "default": 25.0, "minimum": 1.0, "maximum": 1000.0,
     "enabled_when": ("use_brainglobe", True)},
]


def _sample_name_from_labels_path(labels_path):
    stem = Path(labels_path).name
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem.removesuffix("_labels_in_sample")


def _region_output_path(args, region_name):
    """<name>_<region_slug>_corrected_mask.nii.gz next to labels_path. The
    slug matches exactly registration_eval.py's
    `region.lower().replace(" ", "_")` column-key convention, so a
    dice_region_masks entry keyed by the real region name maps straight onto
    this filename with no separate lookup table."""
    slug = region_name.lower().replace(" ", "_")
    name = _sample_name_from_labels_path(args.labels_path)
    return Path(args.labels_path).resolve().parent / f"{name}_{slug}_corrected_mask.nii.gz"


def _annotation_sidecar_path(mask_path):
    """<mask_path> with .nii.gz replaced by .annotated_slices.json -- records
    exactly which z-planes were actually hand-drawn for this region (as
    opposed to signed-distance-interpolated in between), so
    registration_eval.py can restrict Dice/HD95 to just those planes
    automatically instead of you having to remember and retype them."""
    name = mask_path.name
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    return mask_path.parent / f"{name}.annotated_slices.json"


def _load_prior_hand_drawn(args, region_name):
    """If this region was already saved in an earlier session (Save This
    Region writes <region>_corrected_mask.nii.gz + its
    .annotated_slices.json sidecar), return (hand_drawn_slices, mask_arr) so
    editing can resume from exactly those planes' values instead of
    starting over and losing them. Only those specific z's are trusted as
    real hand-verified correction -- everything else in that file may be
    this tool's own earlier interpolation guess, and must NOT be treated as
    ground truth to interpolate on top of again (that would compound
    interpolation error across sessions; see on_region_selected). Returns
    None if this region has no prior save yet.
    """
    out_path = _region_output_path(args, region_name)
    sidecar_path = _annotation_sidecar_path(out_path)
    if not (out_path.exists() and sidecar_path.exists()):
        return None
    hand_drawn_slices = json.loads(sidecar_path.read_text())["hand_drawn_slices"]
    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(out_path)))
    return hand_drawn_slices, mask_arr


def _load_structures(args):
    if args.ontology_json:
        return atlas_utils.load_ccf_ontology_json(args.ontology_json)
    if args.atlas_source == "brainglobe":
        _, _, structures = atlas_utils.get_allen_atlas(args.atlas_res_um)
        return structures
    raise ValueError("Pass either --ontology-json or --atlas-source brainglobe --atlas-res-um <res>")


def main():
    form = run_form("edit_sample_labels", "Edit Sample Labels", _FORM_FIELDS)
    args = SimpleNamespace(
        sample_path=form["sample_path"],
        labels_path=form["labels_path"],
        output_path=form["output_path"],
        ontology_json=form["ontology_json"],
        atlas_source="brainglobe" if form["use_brainglobe"] else None,
        atlas_res_um=form["atlas_res_um"] or 25.0,
    )

    structures = _load_structures(args)
    id_to_name = {sid: info["name"] for sid, info in structures.items()}
    levels = sorted({len(info["structure_id_path"]) for info in structures.values()})
    default_level = 4 if 4 in levels else levels[0]

    # Same axis-order reasoning as paint_mask.py:
    # SimpleITK's Array<->Image round trip gives natural (z,y,x), axis 0 = the
    # actual imaging planes -- NOT ants.image_read().numpy()'s reversed order.
    sample_sitk = sitk.ReadImage(args.sample_path)
    sample_arr = sitk.GetArrayFromImage(sample_sitk)

    labels_sitk = sitk.ReadImage(args.labels_path)
    original_labels = sitk.GetArrayFromImage(labels_sitk).astype(np.uint32)
    if original_labels.shape != sample_arr.shape:
        print(f"WARNING: labels shape {original_labels.shape} != sample shape {sample_arr.shape}")

    # Precisely which voxels the user has actually painted/erased/filled, as
    # opposed to voxels that merely look different because the level view
    # below re-collapsed them. Updated only from real edits via the `paint`
    # event -- napari fires that for paint/fill/erase, but NOT for the bulk
    # `labels_layer.data = ...` reassignment apply_level_view() does below,
    # which is what makes the two distinguishable. export() uses this too,
    # instead of diffing values against original_labels: once the display
    # can be a collapsed level, "differs from original" stops meaning
    # "the user touched this".
    touched_mask = np.zeros(original_labels.shape, dtype=bool)

    viewer = napari.Viewer(title="Edit labels in sample space")
    viewer.add_image(sample_arr, name="sample", colormap="gray")
    labels_layer = viewer.add_labels(original_labels.copy(), name="labels (edit here)", opacity=0.5)

    def on_paint(event):
        for indices, _old_values, _new_value in event.value:
            touched_mask[indices] = True

    labels_layer.events.paint.connect(on_paint)

    # A reusable binary layer for "isolate whichever region is currently
    # picked, edit it, save it on its own" -- separate from labels_layer's
    # shared multi-region editing. Filled in by on_region_selected() below.
    # Needed because target regions can nest (e.g. Cerebellum id=512 is the
    # direct parent of Cerebellar cortex id=528) -- both are legitimate,
    # independently-corrected ground-truth targets, so there's no single
    # shared "level" buffer that represents "just this one region" cleanly.
    isolate_state = {"region_id": None, "region_name": None, "init": None}
    isolate_layer = viewer.add_labels(
        np.zeros(original_labels.shape, dtype=np.uint8),
        name="[isolate] (none selected)", opacity=0.6,
    )

    status_label = QLabel(
        "Pick a level, then a region below. Toggle 'Show only selected region'\n"
        "to isolate it, repaint only the wrong area, then click Export."
    )
    status_label.setWordWrap(True)

    hover_label = QLabel("Region under cursor: -")

    level_combo = QComboBox()
    level_combo.addItems([str(lvl) for lvl in levels])
    level_combo.setCurrentText(str(default_level))

    isolate_checkbox = QCheckBox("Show only selected region")

    search_box = QLineEdit()
    search_box.setPlaceholderText("Filter regions by name...")
    region_list = QListWidget()

    def current_level():
        return int(level_combo.currentText())

    def apply_level_view():
        """Re-collapse the EDITABLE labels layer itself to the chosen level
        (not a separate reference layer) -- painting then assigns that
        level's own ids, and the whole volume declutters to that level's
        boundaries, not just the currently-selected region. touched_mask
        voxels (real prior edits) are preserved verbatim; every other voxel
        is re-derived fresh from original_labels, so switching levels back
        and forth never loses or freezes past edits."""
        lvl = current_level()
        view = atlas_utils.collapse_labels_to_level(original_labels, structures, lvl)
        view[touched_mask] = labels_layer.data[touched_mask]
        labels_layer.data = view

    def refresh_list(filter_text=""):
        lvl = current_level()
        pickable = atlas_utils.structures_at_levels(structures, lvl, lvl)
        sorted_pickable = sorted(pickable.items(), key=lambda kv: kv[1]["name"])

        region_list.clear()
        filter_text = filter_text.lower()
        for sid, info in sorted_pickable:
            if filter_text and filter_text not in info["name"].lower():
                continue
            region_list.addItem(f"{info['name']} ({sid})")
        region_list._ids = [sid for sid, info in sorted_pickable
                             if not filter_text or filter_text in info["name"].lower()]

    def on_level_changed(_text):
        refresh_list(search_box.text())
        apply_level_view()

    apply_level_view()
    refresh_list()
    search_box.textChanged.connect(refresh_list)
    level_combo.currentTextChanged.connect(on_level_changed)

    def on_region_selected():
        row = region_list.currentRow()
        if row < 0:
            return
        region_id = region_list._ids[row]
        labels_layer.selected_label = region_id

        # Isolate this region (id + all its ontology descendants, found via
        # structure_id_path containment -- no name-string matching, so
        # parent/child overlap like Cerebellum/Cerebellar cortex is handled
        # unambiguously) into its own binary layer for individual editing
        # and saving. Built from the CURRENT label state, not the pristine
        # original, so it reflects any prior general edits already made.
        region_name = structures[region_id]["name"]
        descendant_ids = {sid for sid, info in structures.items()
                           if region_id in info["structure_id_path"]}
        init = np.isin(labels_layer.data, list(descendant_ids)).astype(np.uint8)
        isolate_state.update(region_id=region_id, region_name=region_name, init=init)

        # Resume from a prior session's save, if this region has one: overlay
        # ONLY the exact z's it recorded as hand-drawn (their real values, not
        # just its overall array, since the rest of that file may be this
        # tool's own earlier interpolation guess) onto the fresh baseline.
        # save_isolated_region()'s existing diff-vs-init logic then naturally
        # re-detects those planes as touched too, alongside anything newly
        # painted -- interpolation always recomputes from the full union of
        # real hand-drawn keyframes against the true baseline, never on top
        # of a previous session's own interpolated guess.
        view = init.copy()
        prior = _load_prior_hand_drawn(args, region_name)
        if prior is not None:
            prior_slices, prior_arr = prior
            for z in prior_slices:
                view[z] = prior_arr[z]
            isolate_status_label.setText(
                f"Isolated: {region_name}. Resumed {len(prior_slices)} previously "
                f"hand-drawn plane(s): {prior_slices}. Keep painting, then Save This "
                f"Region to update.\nSwitching regions resets this view -- save first "
                f"if you want to keep it."
            )
        else:
            isolate_status_label.setText(
                f"Isolated: {region_name}. Edit here, then Save This Region.\n"
                f"Switching regions resets this view -- save first if you want to keep it."
            )

        isolate_layer.data = view
        isolate_layer.name = f"[isolate] {region_name}"

    region_list.currentRowChanged.connect(lambda _row: on_region_selected())

    def on_isolate_toggled(checked):
        labels_layer.show_selected_label = checked

    isolate_checkbox.toggled.connect(on_isolate_toggled)

    def on_mouse_move(_layer, event):
        if labels_layer.data.ndim != 3:
            return
        pos = labels_layer.world_to_data(event.position)
        z, y, x = (int(round(p)) for p in pos)
        shape = labels_layer.data.shape
        if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
            return
        region_id = int(labels_layer.data[z, y, x])
        name = id_to_name.get(region_id, f"unknown id {region_id}")
        hover_label.setText(f"Region under cursor: {name} ({region_id})")

    labels_layer.mouse_move_callbacks.append(on_mouse_move)

    def export():
        edited = labels_layer.data
        keyframe_edits = {}
        for z in range(edited.shape[0]):
            edit_mask = touched_mask[z]
            if np.any(edit_mask):
                keyframe_edits[z] = (edit_mask, edited[z])

        if not keyframe_edits:
            status_label.setText("No planes edited yet -- nothing to export.")
            return

        status_label.setText(f"Exporting... ({len(keyframe_edits)} edited planes)")
        corrected = mask_utils.interpolate_sparse_label_correction(keyframe_edits, original_labels)

        out_sitk = sitk.GetImageFromArray(corrected.astype(np.uint32))
        out_sitk.CopyInformation(labels_sitk)
        sitk.WriteImage(out_sitk, args.output_path)

        n_changed = int(np.sum(corrected != original_labels))
        msg = (f"Wrote {args.output_path}\n"
               f"Edited planes: {sorted(keyframe_edits.keys())}\n"
               f"Voxels changed from original: {n_changed}")
        status_label.setText(msg)
        print(msg)

    export_btn = QPushButton("Export Corrected Labels")
    export_btn.clicked.connect(export)

    def save_isolated_region():
        if isolate_state["region_id"] is None:
            isolate_status_label.setText("No region isolated yet -- pick one from the list first.")
            return
        edited = isolate_layer.data
        init = isolate_state["init"]
        keyframe_edits = {z: (edited[z] != init[z], edited[z])
                          for z in range(edited.shape[0]) if np.any(edited[z] != init[z])}
        if not keyframe_edits:
            isolate_status_label.setText(f"No changes to {isolate_state['region_name']} yet.")
            return

        corrected = mask_utils.interpolate_sparse_label_correction(keyframe_edits, init)
        out_path = _region_output_path(args, isolate_state["region_name"])
        out_sitk = sitk.GetImageFromArray(corrected.astype(np.uint8))
        out_sitk.CopyInformation(labels_sitk)
        sitk.WriteImage(out_sitk, str(out_path))

        hand_drawn_slices = sorted(keyframe_edits.keys())
        sidecar_path = _annotation_sidecar_path(out_path)
        sidecar_path.write_text(json.dumps({
            "region": isolate_state["region_name"],
            "hand_drawn_slices": hand_drawn_slices,
            "total_z": int(edited.shape[0]),
        }, indent=2))

        n_changed = int(np.sum(corrected != init))
        msg = (f"Wrote {out_path}\n"
               f"Wrote {sidecar_path}\n"
               f"Hand-drawn planes: {hand_drawn_slices}\n"
               f"Voxels changed: {n_changed}")
        isolate_status_label.setText(msg)
        print(msg)

    isolate_status_label = QLabel("Pick a region above to isolate it for individual saving.")
    isolate_status_label.setWordWrap(True)
    save_isolated_btn = QPushButton("Save This Region")
    save_isolated_btn.clicked.connect(save_isolated_region)

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(status_label)
    layout.addWidget(hover_label)
    layout.addWidget(QLabel("Region picker level:"))
    layout.addWidget(level_combo)
    layout.addWidget(isolate_checkbox)
    layout.addWidget(search_box)
    layout.addWidget(region_list)
    layout.addWidget(export_btn)
    layout.addWidget(isolate_status_label)
    layout.addWidget(save_isolated_btn)
    viewer.window.add_dock_widget(dock, area="right", name="Label Correction Export")

    # Application-wide (not focus-dependent) so Cmd/Ctrl+Z always reaches
    # whichever layer napari itself currently considers active -- there are
    # now two independently-paintable layers (labels_layer, isolate_layer),
    # so this can no longer be hardcoded to one of them. `.undo`/`.redo` on a
    # layer with empty history is a safe no-op (verified directly).
    def _undo():
        getattr(viewer.layers.selection.active, "undo", lambda: None)()

    def _redo():
        getattr(viewer.layers.selection.active, "redo", lambda: None)()

    undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), dock)
    undo_shortcut.setContext(Qt.ApplicationShortcut)
    undo_shortcut.activated.connect(_undo)

    redo_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), dock)
    redo_shortcut.setContext(Qt.ApplicationShortcut)
    redo_shortcut.activated.connect(_redo)

    napari.run()


if __name__ == "__main__":
    main()
