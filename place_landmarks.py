"""Interactive tool: hand-place matching anatomical landmarks on the sample
and on the atlas template, for registration_eval.py's landmark TRE (target
registration error) metric.

Two roles, used as a pair (picked from the "Role" dropdown in the form):
  sample: place landmarks on the sample (e.g. <name>_fine_25um.nii.gz,
    the same isotropic grid the registration was run on).
  atlas: place the SAME landmarks, in the SAME order, on the atlas
    template (e.g. the DeMBA P5 reference .tif).

Row i in the sample CSV and row i in the atlas CSV must be the same
anatomical landmark -- registration_eval.py's load_points()/compute_tre()
are row-aligned, not matched by position. Pick points you can identify
confidently in both images (e.g. anterior commissure, 4th ventricle tip,
hippocampal commissure crossing) -- landmark TRE is only as good as your
placement precision (see registration_eval.py's annotator_reproducibility()
for a QC check on that).

Usage (needs a display; the antsreg env has napari+PyQt5+SimpleITK alongside
antspyx -- one env for the whole pipeline):
    conda activate antsreg
    python place_landmarks.py                  # form window, then napari
    python place_landmarks.py --no-form        # straight to napari, from the config
    python place_landmarks.py configs/place_landmarks.s12t_atlas.yaml

    A form window opens for role/image/output-csv (and an optional points CSV
    to preload), pre-filled from configs/place_landmarks.yaml if that exists
    and otherwise from whatever you used last time (kept in .dialog_state/,
    gitignored). Run it once for role=sample and once for role=atlas -- since
    both roles share the same remembered fields, either keep one config file
    per role and pass it explicitly, or use Browse to repoint each run.

    Preload points from a previous export to resume/add to a session
    instead of starting from an empty Points layer.

Workflow:
    1. A napari window opens with the image and an (empty, or preloaded)
       Points layer. Sliced along axis 0 -- the actual imaging/atlas planes,
       same SimpleITK-based convention as every other interactive tool in
       this repo (paint_mask.py, edit_sample_labels.py,
       single_sample.py) -- deliberately NOT ants.image_read().numpy(), which
       gives the reverse axis order for the same file.
    2. Use napari's Points tool (press the "+" / points-add mode in the
       layer controls) to click each landmark, in the SAME order you will
       use (or already used) for the paired image. Delete/undo with napari's
       own point-selection tools if you misclick.
    3. Click "Export Landmarks". Writes a CSV with columns
       index,axis-0,axis-1,axis-2 (voxel coordinates in this image's own
       (z,y,x) array order) -- the same column layout napari's own Points
       CSV writer produces. Reading and writing that format lives in
       _landmark_io, shared with registration_eval.py's load_points() and
       fit_initial_transform.py.
"""
import argparse
from types import SimpleNamespace

import napari
import numpy as np
import SimpleITK as sitk
from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

import _landmark_io  # sibling module
import _local_config  # sibling module


_FORM_FIELDS = [
    {"key": "role", "label": "Role", "type": "choice", "options": ["sample", "atlas"], "default": "sample"},
    {"key": "image_path", "label": "Image to place landmarks on",
     "type": "open_file", "filter": "Images (*.nii *.nii.gz *.tif *.tiff);;All files (*)"},
    {"key": "output_csv", "label": "Output landmarks CSV",
     "type": "save_file", "filter": "CSV files (*.csv);;All files (*)"},
    {"key": "preload_points", "label": "Preload points from an existing CSV",
     "type": "checkbox", "default": False},
    {"key": "points_csv", "label": "Existing points CSV to preload",
     "type": "open_file", "filter": "CSV files (*.csv);;All files (*)",
     "enabled_when": ("preload_points", True)},
]


def main():
    parser = argparse.ArgumentParser(description="Hand-place matching anatomical landmarks")
    _local_config.add_config_arg(parser, "place_landmarks")
    _local_config.add_no_form_arg(parser)
    cli = parser.parse_args()

    form = _local_config.resolve_inputs(
        "place_landmarks", "Place Landmarks", _FORM_FIELDS, cli.config, cli.no_form)
    args = SimpleNamespace(
        role=form["role"],
        image_path=form["image_path"],
        output_csv=form["output_csv"],
        points_csv=form["points_csv"] if form["preload_points"] else None,
    )

    # Same axis-order reasoning as paint_mask.py/edit_sample_labels.py:
    # SimpleITK's Array<->Image round trip gives natural (z,y,x), axis 0 = the
    # actual imaging/atlas planes -- NOT ants.image_read().numpy()'s reversed order.
    base_sitk = sitk.ReadImage(args.image_path)
    arr = sitk.GetArrayFromImage(base_sitk)

    initial_points = np.empty((0, 3), dtype=float)
    if args.points_csv:
        # Checked against this image's shape: preloading a CSV placed on a
        # different image would otherwise silently seed the layer with points
        # in the wrong place (or off-screen).
        initial_points = _landmark_io.read_landmark_csv(
            args.points_csv, shape_zyx=arr.shape, role="preloaded")

    viewer = napari.Viewer(title=f"Place landmarks ({args.role})")
    viewer.add_image(arr, name=args.role, colormap="gray")
    points_layer = viewer.add_points(initial_points, name="landmarks (place here)", size=5, ndim=3)
    points_layer.mode = "add"

    status_label = QLabel(
        "Click to place landmarks, in the SAME anatomical order you use\n"
        "for the paired sample/atlas image (row i = same landmark in both).\n"
        "Then click Export."
    )
    status_label.setWordWrap(True)
    count_label = QLabel(f"Points placed: {len(initial_points)}")

    def on_data_change(_event=None):
        count_label.setText(f"Points placed: {len(points_layer.data)}")

    points_layer.events.data.connect(on_data_change)

    def export():
        data = points_layer.data
        if len(data) == 0:
            status_label.setText("No points placed yet -- nothing to export.")
            return

        # Same column layout napari_builtins' own Points CSV writer produces
        # (index, axis-0, axis-1, axis-2), written through _landmark_io so the
        # tools that read it back -- registration_eval.py's load_points(),
        # fit_initial_transform.py, napari itself -- cannot disagree about the
        # format with whatever wrote it.
        _landmark_io.write_landmark_csv(args.output_csv, data)

        msg = f"Wrote {args.output_csv}\nLandmarks: {len(data)}"
        status_label.setText(msg)
        print(msg)

    export_btn = QPushButton("Export Landmarks")
    export_btn.clicked.connect(export)

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(status_label)
    layout.addWidget(count_label)
    layout.addWidget(export_btn)
    viewer.window.add_dock_widget(dock, area="right", name="Landmark Export")

    napari.run()


if __name__ == "__main__":
    main()
