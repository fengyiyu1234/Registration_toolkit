"""Semi-automatic ground-truth region masks on SPARSE z-slices, using napari +
micro_sam point prompts.

Built for registration validation, not for 3D segmentation. Each brain region is
annotated on a handful of pre-chosen z-slices only (5 by default); every other
plane stays empty on purpose. registration_eval.py then scores Dice/HD95 on just
those planes.

DESIGN CONSTRAINTS (these drive everything below -- do not "simplify" them away)
-------------------------------------------------------------------------------
1. Sparse, not volumetric. Slice positions are locked in the config BEFORE
   annotation starts and differ per region.
2. One file per region: {brain_id}_{region}.nii.gz, values 0/1 only, no
   sub-structure. Not one multi-label file -- slice positions differ per region,
   and neighbouring regions (cortex_surface vs cortex_wm) legitimately share
   boundary voxels, which a single label volume cannot represent.
3. A JSON manifest records which z-planes each region actually got. Evaluation
   code needs it to tell "this plane wasn't annotated" apart from "this region
   isn't present on this plane" -- both look like an empty plane in the mask.
4. micro_sam supplies interactive point-prompt segmentation; the result is a
   plain napari Labels layer, so it can be hand-corrected before saving.
5. NO Z-PROPAGATION. Ever. See below.

WHY THERE IS NO Z-PROPAGATION, AND HOW THAT IS ENFORCED
-------------------------------------------------------
Propagating one slice's prompt/mask to its neighbours would make the 5 slices
statistically dependent, which inflates any later intra-rater reliability
estimate: re-annotating a propagated slice partly re-measures the seed slice
rather than the rater. Each slice must be prompted and segmented on its own.

This is structural here, not a matter of remembering not to click something:

  * Only `micro_sam.sam_annotator.annotator_2d` is used. Its Annotator2d
    ._get_widgets() offers segment / autosegment / commit / clear -- the
    propagation machinery (`segment_nd`, `segment_all_slices`, the Shift-S
    keybinding) exists only in annotator_3d and micro_sam's
    multi_dimensional_segmentation module. Neither is imported anywhere in this
    file; `grep -E 'annotator_3d|segment_nd|segment_all_slices|
    multi_dimensional_segmentation' annotate_gt_sam.py` must return nothing.
  * What is handed to micro_sam is one 2D array, `volume[z]`. It never sees the
    volume, so it has no neighbouring slice to propagate into.
  * SAM image embeddings are cached per (brain, z) in separate files and are
    never reused across z.

CONVENTIONS INHERITED FROM THIS PROJECT
---------------------------------------
Volumes are read with SimpleITK, giving numpy order (z, y, x) with axis 0 = the
real imaging planes -- the same convention as paint_mask.py /
edit_sample_labels.py / place_landmarks.py, and deliberately NOT
`ants.image_read().numpy()`, whose axis order is reversed for the same file
(you would end up annotating left-right cross-sections with no error to warn
you). Exported masks get their geometry from `CopyInformation` on the source
volume and are then re-read and verified with align_masks.check_geometry -- the
napari/numpy round trip carries no header of its own.

USAGE
-----
    conda activate gt_sam
    cp configs/gt_annotation.example.yaml configs/gt_annotation.yaml   # edit paths
    python annotate_gt_sam.py configs/gt_annotation.yaml

    # verify existing output without opening a GUI (works headless):
    python annotate_gt_sam.py configs/gt_annotation.yaml --verify

Resuming: just run it again. Slices already recorded in the manifest are listed
as [done] and are skipped by "Next unfinished"; nothing already saved is
overwritten unless you explicitly select that slice and use "Redo slice", which
loads the existing mask back in for editing.

A REALISTIC EXPECTATION
-----------------------
SAM point prompts work best on closed, intensity-homogeneous objects -- the
ventricles (dark CSF) are an easy case. cortex_surface and cortex_wm are
interfaces rather than objects, so expect to lean much more on hand-correcting
the committed_objects layer there. That is a property of SAM, not a bug here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

import align_masks

REQUIRED_CONFIG_KEYS = ("brain_id", "volume", "output_dir", "regions")


# =====================================================================================
# Config
# =====================================================================================
def load_config(path) -> dict:
    """Read + validate the annotation config. Fails on anything wrong here
    rather than after the SAM model has been downloaded and loaded."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED_CONFIG_KEYS if not cfg.get(k)]
    if missing:
        raise ValueError(f"{path}: missing required key(s): {', '.join(missing)}")

    volume = Path(cfg["volume"])
    if not volume.exists():
        raise FileNotFoundError(f"{path}: volume not found: {volume}")

    regions = cfg["regions"]
    if not isinstance(regions, dict) or not regions:
        raise ValueError(f"{path}: 'regions' must be a non-empty mapping of "
                         f"region name -> list of z indices")
    for region, slices in regions.items():
        if not isinstance(slices, (list, tuple)) or not slices:
            raise ValueError(f"{path}: regions.{region} must be a non-empty list of z indices")
        if len(set(slices)) != len(slices):
            raise ValueError(f"{path}: regions.{region} has duplicate z indices: {slices}")
        if any((not isinstance(z, int)) or z < 0 for z in slices):
            raise ValueError(f"{path}: regions.{region} z indices must be non-negative ints: {slices}")

    output_dir = Path(cfg["output_dir"])
    cfg = dict(cfg)
    cfg["volume"] = volume
    cfg["output_dir"] = output_dir
    cfg["manifest"] = Path(cfg.get("manifest") or output_dir / "annotated_slices_manifest.json")
    cfg["embedding_cache_dir"] = Path(cfg.get("embedding_cache_dir") or output_dir / "_sam_embeddings")
    cfg["model_type"] = cfg.get("model_type", "vit_b_lm")
    cfg["device"] = cfg.get("device") or None
    cfg["regions"] = {r: sorted(int(z) for z in s) for r, s in regions.items()}
    return cfg


# =====================================================================================
# Paths / manifest
# =====================================================================================
def mask_path_for(cfg: dict, region: str) -> Path:
    return cfg["output_dir"] / f"{cfg['brain_id']}_{region}.nii.gz"


def sidecar_path_for(mask_path: Path) -> Path:
    """`<mask>.annotated_slices.json`, the sidecar convention
    edit_sample_labels.py already writes and registration_eval.py's
    load_region_annotation_hint() already reads. Writing it here means masks
    produced by this tool drop straight into eval_config.yaml's
    dice_region_masks with no change to any evaluation code."""
    name = mask_path.name
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    return mask_path.parent / f"{name}.annotated_slices.json"


def load_manifest(cfg: dict) -> dict:
    """{brain_id: {region: [z, ...]}}. Missing/empty file -> empty manifest."""
    manifest_path = cfg["manifest"]
    if not manifest_path.exists():
        return {}
    text = manifest_path.read_text().strip()
    if not text:
        return {}
    return json.loads(text)


def save_manifest(cfg: dict, manifest: dict) -> None:
    """Write the combined manifest plus, for each region, the per-mask sidecar
    derived from it. The combined manifest stays the single source of truth --
    sidecars are rewritten from it every time, so the two cannot drift apart."""
    cfg["manifest"].parent.mkdir(parents=True, exist_ok=True)
    cfg["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    for region, slices in manifest.get(cfg["brain_id"], {}).items():
        sidecar = sidecar_path_for(mask_path_for(cfg, region))
        sidecar.write_text(json.dumps(
            {"hand_drawn_slices": sorted(slices),
             "brain_id": cfg["brain_id"],
             "region": region,
             "source": "annotate_gt_sam.py"}, indent=2) + "\n")


def manifest_slices(manifest: dict, brain_id: str, region: str) -> list[int]:
    return sorted(manifest.get(brain_id, {}).get(region, []))


# =====================================================================================
# Mask I/O
# =====================================================================================
def load_region_mask(cfg: dict, region: str, reference: sitk.Image) -> np.ndarray:
    """Existing 3D mask for a region, or a fresh all-zero one.

    uint8 throughout: values are only ever 0/1, and on a 20um whole-brain grid
    the difference between uint8 and, say, int32 is hundreds of megabytes. Only
    one region's mask is ever held in memory (see AnnotationSession)."""
    shape = tuple(reversed(reference.GetSize()))     # sitk (x,y,z) -> numpy (z,y,x)
    path = mask_path_for(cfg, region)
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)

    img = sitk.ReadImage(str(path))
    check = align_masks.check_geometry(reference, img, name=path.name)
    if not check.ok:
        raise ValueError(
            f"Existing mask does not match the volume's grid -- refusing to add to it.\n"
            f"{check}\nRun align_masks.py on it first, or delete it to start over.")
    return (sitk.GetArrayFromImage(img) > 0).astype(np.uint8)


def save_region_mask(cfg: dict, region: str, mask: np.ndarray, reference: sitk.Image) -> Path:
    """Write the region's 3D mask, forcing the source volume's geometry.

    CopyInformation is what makes the export trustworthy: a numpy array coming
    back out of napari has no spacing/origin/direction of its own, and whatever
    a default writer would invent gets silently baked into every later surface
    distance. The file is read back and re-checked so a bad write fails here
    rather than in the metrics weeks later."""
    path = mask_path_for(cfg, region)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = sitk.GetImageFromArray(mask.astype(np.uint8))
    out.CopyInformation(reference)
    sitk.WriteImage(out, str(path))

    check = align_masks.check_geometry(reference, sitk.ReadImage(str(path)), name=path.name)
    if not check.ok:
        raise RuntimeError(f"Exported mask is not aligned to the volume:\n{check}")
    return path


# =====================================================================================
# Verification
# =====================================================================================
def verify(cfg: dict) -> bool:
    """Check every exported mask against the acceptance criteria. Headless-safe;
    also run automatically after each save during a session.

      * grid identical to the volume (size/spacing/origin/direction)
      * dtype uint8, values a subset of {0, 1}
      * the set of non-empty z-planes equals the manifest's list EXACTLY --
        checked in both directions, so neither a plane the manifest forgot nor a
        manifest entry whose plane came out empty can slip through
    """
    reference = sitk.ReadImage(str(cfg["volume"]))
    manifest = load_manifest(cfg)
    brain_id = cfg["brain_id"]
    all_ok = True

    print(f"=== verify {brain_id} ===")
    print(f"volume: {cfg['volume']}")
    print(f"        size={reference.GetSize()} spacing={reference.GetSpacing()}")

    recorded_regions = set(manifest.get(brain_id, {}))
    for region in sorted(set(cfg["regions"]) | recorded_regions):
        path = mask_path_for(cfg, region)
        planned = cfg["regions"].get(region, [])
        recorded = manifest_slices(manifest, brain_id, region)

        if not path.exists():
            if recorded:
                print(f"FAIL  {region}: manifest lists {recorded} but {path.name} does not exist")
                all_ok = False
            else:
                print(f"----  {region}: not started (0/{len(planned)} slices)")
            continue

        problems = []
        img = sitk.ReadImage(str(path))
        check = align_masks.check_geometry(reference, img, name=path.name)
        problems += check.problems

        arr = sitk.GetArrayFromImage(img)
        if arr.dtype != np.uint8:
            problems.append(f"dtype is {arr.dtype}, expected uint8")
        stray = np.setdiff1d(np.unique(arr), np.array([0, 1], dtype=arr.dtype))
        if stray.size:
            problems.append(f"values outside {{0, 1}}: {stray.tolist()[:10]}")

        nonempty = sorted(int(z) for z in np.flatnonzero(arr.reshape(arr.shape[0], -1).any(axis=1)))
        if nonempty != recorded:
            missing = sorted(set(recorded) - set(nonempty))
            extra = sorted(set(nonempty) - set(recorded))
            if missing:
                problems.append(f"manifest lists z={missing} but those planes are empty in the mask")
            if extra:
                problems.append(f"mask has content on z={extra}, absent from the manifest")

        unplanned = sorted(set(recorded) - set(planned))
        if unplanned:
            problems.append(f"annotated z={unplanned} are not in the config's locked slice list")

        if problems:
            all_ok = False
            print(f"FAIL  {region}  ({path.name})")
            for p in problems:
                print(f"        - {p}")
        else:
            print(f"PASS  {region}: {len(recorded)}/{len(planned)} slices {recorded}")

    print("=== all checks passed ===" if all_ok else "=== FAILURES above ===")
    return all_ok


# =====================================================================================
# Interactive session
# =====================================================================================
class AnnotationSession:
    """Drives one napari window through the (region, z) work queue.

    Holds at most ONE region's 3D mask in memory at a time; switching regions
    flushes the current one to disk and loads the next from disk.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reference = sitk.ReadImage(str(cfg["volume"]))
        # (z, y, x), axis 0 = the real imaging planes. See module docstring.
        self.volume = sitk.GetArrayFromImage(self.reference)
        self.manifest = load_manifest(cfg)
        self.manifest.setdefault(cfg["brain_id"], {})

        n_z = self.volume.shape[0]
        for region, slices in cfg["regions"].items():
            bad = [z for z in slices if z >= n_z]
            if bad:
                raise ValueError(f"regions.{region}: z {bad} out of range for a "
                                 f"{n_z}-plane volume")

        self.queue = [(region, z) for region in cfg["regions"] for z in cfg["regions"][region]]
        self.current_region: str | None = None
        self.current_mask: np.ndarray | None = None
        self.loaded_item: tuple[str, int] | None = None

        self.viewer = None
        self.annotator_started = False

    # ---- state -------------------------------------------------------------
    def is_done(self, region: str, z: int) -> bool:
        return z in manifest_slices(self.manifest, self.cfg["brain_id"], region)

    def _ensure_region_loaded(self, region: str) -> None:
        if self.current_region == region:
            return
        self._flush_current_region()
        self.current_mask = load_region_mask(self.cfg, region, self.reference)
        self.current_region = region

    def _flush_current_region(self) -> None:
        """Drop the in-memory mask. Saves are written eagerly on every "Save
        slice", so there is never unwritten work here to lose."""
        self.current_mask = None
        self.current_region = None

    # ---- micro_sam ---------------------------------------------------------
    def _embedding_path(self, z: int) -> str:
        cache = self.cfg["embedding_cache_dir"]
        cache.mkdir(parents=True, exist_ok=True)
        # One file per (brain, z). Never shared between slices -- see the
        # no-propagation note in the module docstring.
        return str(cache / f"{self.cfg['brain_id']}_z{z:04d}.zarr")

    def load_slice(self, region: str, z: int, redo: bool = False) -> str:
        """Put slice z in front of micro_sam as a standalone 2D image."""
        from micro_sam.sam_annotator import annotator_2d
        from micro_sam.sam_annotator._state import AnnotatorState

        if self.is_done(region, z) and not redo:
            return (f"{region} z={z} is already annotated. Use 'Redo slice' to "
                    f"reopen it for editing (it will be overwritten on save).")

        self._ensure_region_loaded(region)
        plane = np.ascontiguousarray(self.volume[z])
        prefill = None
        if redo and self.current_mask[z].any():
            # Existing work comes back as committed_objects so it can be edited
            # rather than redrawn. uint32 is what the Labels layers use.
            prefill = self.current_mask[z].astype(np.uint32)

        state = AnnotatorState()
        if not self.annotator_started:
            # First slice: build the viewer, the SAM predictor and the widget dock.
            self.viewer = annotator_2d(
                plane,
                embedding_path=self._embedding_path(z),
                segmentation_result=prefill,
                model_type=self.cfg["model_type"],
                device=self.cfg["device"],
                return_viewer=True,
                viewer=self.viewer,
            )
            self.annotator_started = True
        else:
            # Later slices: swap the image in place instead of calling
            # annotator_2d again -- a second call would add a duplicate "image"
            # layer and a second widget dock to the same window. Reusing
            # state.predictor/decoder also skips reloading the SAM checkpoint;
            # only the per-slice embeddings are recomputed.
            state.image_shape = plane.shape
            state.initialize_predictor(
                plane,
                model_type=self.cfg["model_type"],
                ndim=2,
                save_path=self._embedding_path(z),
                predictor=state.predictor,
                decoder=state.decoder,
                device=self.cfg["device"],
            )
            self.viewer.layers["image"].data = plane
            # Defensive: the embedding widget sets this flag, and while it is
            # True _update_image() returns early and would leave the previous
            # slice's committed_objects on screen.
            state.skip_recomputing_embeddings = False
            state.annotator._update_image(segmentation_result=prefill)

        self.loaded_item = (region, z)
        note = " (existing mask reloaded for editing)" if prefill is not None else ""
        return (f"Loaded {region} z={z}{note}.\n"
                f"Place positive/negative points in 'point_prompts', press 's' to "
                f"segment and 'c' to commit, hand-correct 'committed_objects' as "
                f"needed, then click 'Save slice'.")

    def save_slice(self) -> str:
        """Binarize committed_objects into the region's 3D mask, write both the
        mask and the manifest immediately, then verify the geometry."""
        if self.loaded_item is None:
            return "Nothing loaded yet -- pick a slice and click 'Load slice' first."
        region, z = self.loaded_item

        layer = self.viewer.layers["committed_objects"].data
        # micro_sam gives each committed object its own instance id. Any nonzero
        # id is foreground: a region that appears as two disconnected blobs on a
        # plane (both ventricles, say) belongs in the same binary mask, with no
        # left/right distinction.
        plane_mask = (np.asarray(layer) > 0).astype(np.uint8)
        if plane_mask.shape != self.volume.shape[1:]:
            return (f"committed_objects is {plane_mask.shape}, expected "
                    f"{self.volume.shape[1:]} -- not saving.")
        if not plane_mask.any():
            return (f"committed_objects is empty for {region} z={z}. Nothing saved "
                    f"(an empty plane would be indistinguishable from 'not annotated').")

        self._ensure_region_loaded(region)
        self.current_mask[z] = plane_mask

        path = save_region_mask(self.cfg, region, self.current_mask, self.reference)
        recorded = set(manifest_slices(self.manifest, self.cfg["brain_id"], region))
        recorded.add(int(z))
        self.manifest[self.cfg["brain_id"]][region] = sorted(recorded)
        save_manifest(self.cfg, self.manifest)

        return (f"Saved {region} z={z} ({int(plane_mask.sum())} voxels).\n"
                f"{path.name}: {len(recorded)}/{len(self.cfg['regions'][region])} slices done.")

    def next_unfinished(self) -> tuple[str, int] | None:
        for region, z in self.queue:
            if not self.is_done(region, z):
                return region, z
        return None


def _build_dock(session: AnnotationSession):
    """The session control panel. Same plain-Qt style as the other interactive
    tools in this repo (paint_mask.py, edit_sample_labels.py)."""
    from PyQt5.QtWidgets import (
        QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
    )

    status = QLabel("Pick a slice below, then 'Load slice'.")
    status.setWordWrap(True)
    listing = QListWidget()

    def refresh_list():
        selected = listing.currentRow()
        listing.clear()
        for region, z in session.queue:
            tag = "done" if session.is_done(region, z) else "todo"
            item = QListWidgetItem(f"[{tag}] {region}  z={z}")
            item.setData(0x0100, (region, z))  # Qt.UserRole
            listing.addItem(item)
        if 0 <= selected < listing.count():
            listing.setCurrentRow(selected)

    def selected_item():
        item = listing.currentItem()
        return item.data(0x0100) if item else None

    def do_load(redo: bool):
        target = selected_item()
        if target is None:
            status.setText("Select a (region, z) entry in the list first.")
            return
        region, z = target
        status.setText(f"Loading {region} z={z} (computing SAM embeddings, may take a moment)...")
        try:
            status.setText(session.load_slice(region, z, redo=redo))
        except Exception as exc:                       # keep the GUI alive
            status.setText(f"ERROR loading {region} z={z}: {type(exc).__name__}: {exc}")
            raise

    def do_save():
        try:
            status.setText(session.save_slice())
        except Exception as exc:
            status.setText(f"ERROR saving: {type(exc).__name__}: {exc}")
            raise
        refresh_list()

    def do_next():
        target = session.next_unfinished()
        if target is None:
            status.setText("All planned slices for this brain are annotated.")
            return
        for row in range(listing.count()):
            if listing.item(row).data(0x0100) == target:
                listing.setCurrentRow(row)
                break
        do_load(redo=False)

    def do_verify():
        ok = verify(session.cfg)
        status.setText("Verification PASSED (details in the terminal)." if ok
                       else "Verification FAILED -- see the terminal for details.")

    refresh_list()

    load_btn = QPushButton("Load slice")
    load_btn.clicked.connect(lambda: do_load(redo=False))
    redo_btn = QPushButton("Redo slice (overwrite)")
    redo_btn.clicked.connect(lambda: do_load(redo=True))
    save_btn = QPushButton("Save slice")
    save_btn.clicked.connect(do_save)
    next_btn = QPushButton("Next unfinished")
    next_btn.clicked.connect(do_next)
    verify_btn = QPushButton("Verify all output")
    verify_btn.clicked.connect(do_verify)

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(QLabel(f"<b>{session.cfg['brain_id']}</b> -- ground truth slices"))
    layout.addWidget(listing)
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.addWidget(load_btn)
    row_layout.addWidget(redo_btn)
    layout.addWidget(row)
    layout.addWidget(save_btn)
    layout.addWidget(next_btn)
    layout.addWidget(verify_btn)
    layout.addWidget(status)
    return dock


def run_session(cfg: dict) -> None:
    import napari

    session = AnnotationSession(cfg)
    done = sum(session.is_done(r, z) for r, z in session.queue)
    print(f"[{cfg['brain_id']}] {done}/{len(session.queue)} planned slices already annotated.")

    # The dock needs a viewer to attach to, and annotator_2d needs one to add
    # its own widget to; create it up front and hand the same one to both.
    session.viewer = napari.Viewer(title=f"GT annotation -- {cfg['brain_id']}")
    session.viewer.window.add_dock_widget(_build_dock(session), area="right", name="GT session")
    napari.run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Sparse-slice ground-truth region masks with napari + micro_sam.")
    parser.add_argument("config", help="annotation config YAML (see configs/gt_annotation.example.yaml)")
    parser.add_argument("--verify", action="store_true",
                        help="check existing output against the manifest and exit (no GUI)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.verify:
        return 0 if verify(cfg) else 1

    run_session(cfg)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
