# GT_tool_for_registration

Ground-truth annotation, visual QC and evaluation tooling for the LSFM mouse-brain
registration project. The registration pipeline itself lives in
[`../Registration_ants`](../Registration_ants); this repo is everything a human
touches — painting masks, placing landmarks, drawing region ground truth,
looking at the result, and scoring it.

These tools import `registration_ants` (via its `pip install -e` in the
`antsreg` env) but the pipeline does not import anything from here, so the
dependency only points one way.

---

## Environments

| env | what runs in it | why |
|---|---|---|
| `antsreg` | everything except `annotate_gt_sam.py` | already has napari + SimpleITK + antspyx + the editable `registration_ants` |
| `gt_sam` | `annotate_gt_sam.py` | `micro_sam` pulls in torch and its own pinned napari/numpy; installing it into `antsreg` would risk downgrading what the registration pipeline runs on |

Creating `gt_sam` (already done on this machine):

```bash
conda create -y -n gt_sam -c conda-forge -c pytorch python=3.11 \
    micro_sam napari pyqt simpleitk pyyaml
```

`align_masks.py` deliberately depends on nothing beyond SimpleITK/numpy/scipy,
so it runs in either env.

---

## Producing ground truth

Two different tools, for two genuinely different situations.

### `annotate_gt_sam.py` — draw region ground truth from scratch, on sparse slices

napari + micro_sam point prompts. Each region is annotated on a handful of
pre-chosen z-slices only; every other plane stays empty on purpose.

```bash
conda activate gt_sam
cp configs/gt_annotation.example.yaml configs/gt_annotation.yaml   # edit paths
python annotate_gt_sam.py configs/gt_annotation.yaml

# check existing output without a GUI (works headless):
python annotate_gt_sam.py configs/gt_annotation.yaml --verify
```

**Slice positions are locked in the config before annotation starts** and may
differ per region. `--verify` flags any annotated plane that isn't in that list.

**One file per region**: `{brain_id}_{region}.nii.gz`, values 0/1 only. Not one
multi-label file — slice positions differ per region, and neighbouring regions
(cortex_surface vs cortex_wm) legitimately share boundary voxels, which a single
label volume can't represent.

**Two manifests are written.** The combined
`annotated_slices_manifest.json` — `{"brain01": {"ventricle": [50, 80, ...]}}` —
is the source of truth. From it, a per-mask `{brain_id}_{region}.annotated_slices.json`
sidecar is rewritten each save, because that is the format
`registration_eval.py`'s `load_region_annotation_hint()` already reads: masks
from this tool drop straight into `eval_config.yaml`'s `dice_region_masks` with
no change to any evaluation code. Evaluation needs the manifest to tell
"this plane wasn't annotated" apart from "this region isn't present here" —
both look like an empty plane in the mask.

**No z-propagation, ever.** Propagating a prompt across slices would make the
5 slices statistically dependent and inflate any later intra-rater reliability
estimate. This is structural, not a rule to remember: only `annotator_2d` is
used (the propagation widgets exist solely in `annotator_3d` and
`multi_dimensional_segmentation`, neither of which is imported), micro_sam only
ever sees one 2D array, and embeddings are cached per `(brain, z)` and never
shared. `tests/test_annotate_gt_sam_smoke.py` asserts none of those APIs appear
in the file's executable code.

**Resuming**: just run it again. Finished slices show as `[done]` and are
skipped by "Next unfinished". To redo one, select it and use "Redo slice" —
the existing mask is loaded back into `committed_objects` for editing, and
overwrites on save.

**A realistic expectation**: SAM works best on closed, intensity-homogeneous
objects. Ventricles (dark CSF) are easy. `cortex_surface` and `cortex_wm` are
*interfaces*, not objects, so expect to lean much harder on hand-correcting the
`committed_objects` layer there. That's SAM's nature, not a bug.

### `edit_sample_labels.py` — hand-correct an existing registration's labels

Starts from `labels_in_sample.nii.gz` (a completed registration) and lets you
repaint whichever regions came out wrong on a few representative planes, then
interpolates the correction. "Save This Region" exports one region as a binary
mask plus the same `.annotated_slices.json` sidecar.

Use this when the registration is mostly right; use `annotate_gt_sam.py` when
you want ground truth that owes nothing to the registration being evaluated.

```bash
conda activate antsreg && python edit_sample_labels.py    # opens a path form
```

---

## The other tools

- **`paint_mask.py`** — paint a binary inclusion/exclusion mask (`KIND="mask"`,
  e.g. excluding a tissue crack) or a paired guide outline (`KIND="guide"`, fed
  to `ants.registration`'s `multivariate_extras`). Paths go in the gitignored
  `paint_mask_local.yaml`; copy it from `paint_mask_local.example.yaml`, edit,
  then `python paint_mask.py` with no arguments. Its guide outputs pair with
  `../Registration_ants/scripts/project_outline.py`.
- **`place_landmarks.py`** — click matching anatomical landmarks on the sample
  and on the atlas, in the same order, for `registration_eval.py`'s landmark TRE.
  A form window asks for role/paths, pre-filled from last time.
- **`single_sample.py`** — napari QC viewer for one registered sample: overlays
  the warped atlas, the sample, and per-region labels. Paths are edited in the
  `CONFIG` dict at the top.
- **`align_masks.py`** — see below.
- **`registration_eval.py`** — Dice/HD95, landmark TRE, Jacobian and
  inverse-consistency metrics across samples and groups.
  `python registration_eval.py configs/eval_config.yaml`.

All the napari tools read images through SimpleITK, giving numpy order
`(z, y, x)` with axis 0 = the real imaging planes — deliberately **not**
`ants.image_read().numpy()`, whose axis order is reversed for the same file. Get
that wrong and you annotate left-right cross-sections with nothing to warn you.

---

## Before you compute any metric: `align_masks.py`

A mask that lost its spacing/origin/direction on export looks pixel-perfect in a
viewer and silently poisons every surface-distance metric. Check first:

```bash
# report only, writes nothing
python align_masks.py --check-only --source sample_fine_20um.nii.gz \
                      --masks gt_*.nii.gz

# force the source's geometry onto each mask, saving copies
python align_masks.py --source sample_fine_20um.nii.gz --masks gt_*.nii.gz \
                      --out-dir aligned/ --report aligned/report.txt

python align_masks.py --selftest
```

Two behaviours worth knowing:

- **A shape mismatch is a hard error, not something it fixes.** Differing voxel
  dimensions mean a resampling or wrong-file problem upstream; rewriting the
  header would bury it. Nothing is written when this fires — shapes are all
  validated before the first output file.
- **Originals are never modified in place.** It refuses an `--out-dir` that
  would overwrite an input.

### Surface distance is in microns, not voxels

The trap is axis order: `np.argwhere` on a `GetArrayFromImage` array yields
`(z, y, x)`, while `GetSpacing()` is `(x, y, z)` — the reverse. So the
conversion reverses the spacing before multiplying:

```python
spacing_zyx = np.asarray(img.GetSpacing())[::-1]
points_um   = np.argwhere(border) * spacing_zyx
```

Every coordinate reaching the KD-tree is already in microns, so "voxel count"
never appears downstream. Under isotropic spacing a transposed vector still
gives the right answer by luck — which is why the selftests use **anisotropic**
spacing and shift along **all three axes**, and cross-check against SimpleITK's
own `TransformIndexToPhysicalPoint`.

---

## Tests

```bash
conda activate antsreg
python align_masks.py --selftest                        # also fine in gt_sam
python tests/test_registration_eval_smoke.py
python tests/test_annotate_gt_sam_smoke.py              # also fine in gt_sam

# micro_sam integration end-to-end; needs gt_sam + an OpenGL display.
# napari's canvas needs real GL, so QT_QPA_PLATFORM=offscreen alone won't do:
conda activate gt_sam
env -u DISPLAY QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 \
  xvfb-run -a -s "-screen 0 1280x1024x24" \
  python tests/test_annotate_gt_sam_microsam_e2e.py
```

Manual-assert style, no pytest, matching `../Registration_ants/tests/`.

### What automated tests can't cover

Actually clicking point prompts and letting SAM segment needs a human. On a
machine with a display, walk `annotate_gt_sam.py` through:

1. Select a `(region, z)` row → **Load slice** (first one downloads the SAM
   checkpoint and computes embeddings; slower than the rest).
2. Place a few positive points in `point_prompts`, press `s` to segment, `c` to
   commit. Toggle positive/negative with `t`.
3. Hand-correct `committed_objects` with napari's brush/eraser.
4. **Save slice** → the row turns `[done]`, and the terminal prints the voxel
   count and slice progress.
5. **Next unfinished** → the next slice loads with a blank
   `committed_objects` and no second widget dock.
6. Re-select a `[done]` row → **Load slice** refuses; **Redo slice** reopens it
   with the saved mask loaded.
7. **Verify all output** → PASS.

Steps 4–7 are covered by `test_annotate_gt_sam_microsam_e2e.py`; steps 2–3 are
the genuinely manual part.
