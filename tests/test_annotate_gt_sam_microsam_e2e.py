"""End-to-end exercise of annotate_gt_sam.py's micro_sam integration, without a
human: load a slice, fake a commit, save, load the NEXT slice through the
state-reuse path, save, then reopen the first one with redo.

What this actually pins down (the parts most likely to break on a micro_sam
upgrade):
  * annotator_2d builds the viewer, the SAM predictor and all six layers.
  * Loading a second slice reuses that machinery instead of calling annotator_2d
    again -- no duplicate "image" layer, no second widget dock, and
    committed_objects is reset rather than carrying the previous slice over.
  * A finished slice is skipped unless redo=True, and redo restores the saved
    mask into committed_objects instead of a blank canvas.
  * Embeddings are cached one file per z and never shared between slices.

What it does NOT cover: actually clicking point prompts and letting SAM
segment. That needs a human; see README.md's manual checklist.

REQUIREMENTS -- this is why it is a separate file from the plain smoke test:
  * the gt_sam env (micro_sam + napari + torch)
  * an OpenGL-capable display. On a headless box use xvfb; napari's vispy
    canvas needs real GL, so QT_QPA_PLATFORM=offscreen alone is not enough:

      conda activate gt_sam
      env -u DISPLAY QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 \
        xvfb-run -a -s "-screen 0 1280x1024x24" \
        python tests/test_annotate_gt_sam_microsam_e2e.py

  * ~410 MB of SAM checkpoints on the first run (cached in ~/.cache/micro_sam).
    Minutes on CPU the first time, seconds afterwards.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import annotate_gt_sam as gt  # noqa: E402


Z, Y, X = 12, 96, 96
SPACING_XYZ = (10.0, 25.0, 40.0)
ORIGIN_XYZ = (5.0, -2.0, 8.0)


def make_volume(path):
    """A bright disc on a dark background, growing with z -- crude, but it gives
    SAM something object-like and makes each plane distinguishable."""
    yy, xx = np.mgrid[0:Y, 0:X]
    radius = np.sqrt((yy - Y // 2) ** 2 + (xx - X // 2) ** 2)
    vol = np.stack([np.where(radius < 20 + k, 3000, 300) for k in range(Z)]).astype(np.uint16)
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing(SPACING_XYZ)
    img.SetOrigin(ORIGIN_XYZ)
    sitk.WriteImage(img, str(path))
    return vol


def disc(radius):
    yy, xx = np.mgrid[0:Y, 0:X]
    return (np.sqrt((yy - Y // 2) ** 2 + (xx - X // 2) ** 2) < radius).astype(np.uint32)


def n_docks(viewer):
    return len(viewer.window.dock_widgets)


def main():
    import napari

    tmp = Path(tempfile.mkdtemp())
    vol = make_volume(tmp / "v.nii.gz")
    (tmp / "c.yaml").write_text(
        f"brain_id: b1\nvolume: {tmp}/v.nii.gz\noutput_dir: {tmp}/out\n"
        f"device: cpu\nregions:\n  ventricle: [3, 7]\n")

    cfg = gt.load_config(tmp / "c.yaml")
    session = gt.AnnotationSession(cfg)
    session.viewer = napari.Viewer(title="e2e", show=False)

    print("1. first load: annotator_2d builds the viewer, predictor and layers")
    print("   " + session.load_slice("ventricle", 3).splitlines()[0])
    names = [layer.name for layer in session.viewer.layers]
    for expected in ("image", "point_prompts", "prompts", "current_object",
                     "auto_segmentation", "committed_objects"):
        assert expected in names, f"missing layer {expected!r}: {names}"
    assert names.count("image") == 1, names
    assert session.viewer.layers["image"].data.shape == (Y, X)
    docks_after_first = n_docks(session.viewer)
    print(f"   layers ok, {docks_after_first} dock widget(s)")

    # Stand in for "placed points, pressed s, pressed c".
    session.viewer.layers["committed_objects"].data = disc(23)
    print("   " + session.save_slice().splitlines()[0])

    print("2. second load: state is reused, not rebuilt")
    session.load_slice("ventricle", 7)
    names = [layer.name for layer in session.viewer.layers]
    assert names.count("image") == 1, f"a duplicate image layer was added: {names}"
    assert n_docks(session.viewer) == docks_after_first, \
        f"a second annotator dock was added: {docks_after_first} -> {n_docks(session.viewer)}"
    assert np.array_equal(session.viewer.layers["image"].data, vol[7]), "wrong plane loaded"
    assert not np.asarray(session.viewer.layers["committed_objects"].data).any(), \
        "committed_objects still holds the previous slice's mask"
    print("   no duplicate layer/dock, image swapped, committed_objects cleared")

    session.viewer.layers["committed_objects"].data = disc(27)
    print("   " + session.save_slice().splitlines()[0])

    print("3. a finished slice is skipped unless redo")
    message = session.load_slice("ventricle", 3)
    assert "already annotated" in message, message
    assert np.array_equal(session.viewer.layers["image"].data, vol[7]), \
        "the skipped load still swapped the image"

    print("4. redo restores the saved mask for editing")
    session.load_slice("ventricle", 3, redo=True)
    restored = np.asarray(session.viewer.layers["committed_objects"].data)
    assert np.array_equal(restored > 0, disc(23) > 0), "redo did not restore the saved plane"

    print("5. output verifies, embeddings cached one file per z")
    assert gt.verify(cfg), "verify failed after the session"
    cached = sorted(p.name for p in cfg["embedding_cache_dir"].iterdir())
    assert cached == ["b1_z0003.zarr", "b1_z0007.zarr"], cached

    print("=== micro_sam end-to-end test passed ===")


if __name__ == "__main__":
    main()
