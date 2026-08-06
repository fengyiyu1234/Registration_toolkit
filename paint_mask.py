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
    Two roles, used in sequence: ROLE="sample" (paint the structure's
    current, possibly deformed extent directly on the sample -- no
    shortcut, run this first) then ROLE="atlas" (paint the same
    structure's true, undeformed extent on the atlas template; set
    EXISTING_MASK_PATH to ../Registration_ants/scripts/project_outline.py's output as a
    starting guess instead of a blank canvas). Sparse keyframes only (you
    paint a handful of representative planes and the rest is interpolated
    between them) -- unlike `mask`'s dense editing, since a guide outline
    is a bounded 3D blob, not a mask that needs a meaningful value on every
    single plane.

Usage (needs a display; runs in the antsreg conda env, which has
napari+PyQt5+SimpleITK alongside antspyx and the pip-installed-editable
registration_ants package this file imports from): edit
configs/paint_mask.yaml (gitignored -- copy it from
configs/paint_mask.example.yaml the first time), then just run the file --
no command-line arguments.

    conda activate antsreg
    python paint_mask.py
    python paint_mask.py configs/paint_mask.guide_s12t.yaml  # 或指定另一份配置

Both read images via SimpleITK (`sitk.GetArrayFromImage`), giving the
natural (z,y,x) array order with axis 0 = the actual imaging/atlas planes --
deliberately NOT `ants.image_read().numpy()`, which gives the reverse axis
order for the same file (verified against real pipeline output), so axis 0
would scroll through a left-right cross-section instead of the actual
z-planes, and you'd paint on the wrong slices without any error to warn you.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import napari
import numpy as np
import SimpleITK as sitk
from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

# Resolved through the editable install of ../Registration_ants (pip install -e
# there puts it on sys.path for the antsreg env) -- no path hacking needed.
# mask_utils is pure numpy/scipy, so this import does NOT drag in antspyx.
from registration_ants import mask_utils

import _local_config  # sibling module

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
    )


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


def _launch_viewer(arr, prefill, title, image_layer_name, mask_layer_name, opacity=None):
    viewer = napari.Viewer(title=title)
    viewer.add_image(arr, name=image_layer_name, colormap="gray")
    labels_kwargs = {"opacity": opacity} if opacity is not None else {}
    paint_layer = viewer.add_labels(prefill.copy(), name=mask_layer_name, **labels_kwargs)
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


def _sparse_keyframes(paint_layer, n_planes):
    return {z: (paint_layer.data[z] > 0) for z in range(n_planes) if np.any(paint_layer.data[z])}


def _run_mask(args):
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


def _run_guide(args):
    base_sitk, arr = _read_sitk_array(args.image_path)

    prefill = np.zeros(arr.shape, dtype=np.uint8)
    if args.existing_mask:
        loaded = _load_mask_array(args.existing_mask, arr.shape)
        if loaded is not None:
            prefill = loaded

    viewer, paint_layer = _launch_viewer(
        arr, prefill, f"Paint guide outline ({args.role})", args.role, "guide outline (paint here)")

    guess_note = "Pre-filled with the existing mask -- adjust/redraw as needed.\n" if args.existing_mask else ""
    status_label = QLabel(
        "Paint a rough outline on a few planes (start, end, and any\n"
        "planes where the shape changes a lot), then click Export.\n" + guess_note)

    def export():
        keyframes = _sparse_keyframes(paint_layer, arr.shape[0])
        if not keyframes:
            status_label.setText("No planes painted yet -- nothing to export.")
            return

        status_label.setText(f"Exporting... ({len(keyframes)} painted planes)")
        outline = mask_utils.interpolate_sparse_mask(keyframes, arr.shape)

        out_sitk = sitk.GetImageFromArray(outline.astype(np.uint8))
        out_sitk.CopyInformation(base_sitk)
        sitk.WriteImage(out_sitk, args.output_path)

        n = int(outline.sum())
        msg = f"Wrote {args.output_path}\nKeyframe planes: {sorted(keyframes.keys())}\nOutline voxels: {n}"
        status_label.setText(msg)
        print(msg)

    _make_export_dock(viewer, status_label, export, "Export Outline", "Guide Outline Export")


def main():
    parser = argparse.ArgumentParser(description="Paint a binary mask or a paired guide outline")
    _local_config.add_config_arg(parser, "paint_mask")
    args_cli = parser.parse_args()

    cfg = _load_local_config(args_cli.config)
    if cfg.kind == "mask":
        args = SimpleNamespace(sample_path=cfg.image_path, output_path=cfg.output_path,
                                existing_mask=cfg.existing_mask_path)
        _run_mask(args)
    elif cfg.kind == "guide":
        args = SimpleNamespace(image_path=cfg.image_path, output_path=cfg.output_path,
                                existing_mask=cfg.existing_mask_path, role=cfg.role)
        _run_guide(args)
    else:
        raise ValueError(f"Unknown kind: {cfg.kind!r} (expected 'mask' or 'guide')")

    napari.run()


if __name__ == "__main__":
    main()
