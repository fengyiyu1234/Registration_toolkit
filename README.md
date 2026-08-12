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

Both are checked in as env files, so a new machine is three commands:

```bash
conda env create -f environment-antsreg.yml
conda activate antsreg
pip install -e ../Registration_ants     # path-dependent, so not in the yml

conda env create -f environment-gtsam.yml   # only if you need annotate_gt_sam.py
```

`align_masks.py` deliberately depends on nothing beyond SimpleITK/numpy/scipy,
so it runs in either env.

Linux, macOS and Windows all work — `antspyx` ships `win_amd64` wheels, and
nothing here shells out to the ANTs command-line binaries, so there is no
native ANTs install to worry about. On Windows, run napari from a normal
terminal (Anaconda Prompt or PowerShell); it is a native GUI, no X11 involved.

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

- **`paint_mask.py`** — paint a binary inclusion/exclusion mask (`kind: mask`,
  e.g. excluding a tissue crack) or a guide outline (`kind: guide`, fed to
  `ants.registration`'s `multivariate_extras`). A guide export can cover
  several brain regions at once — one napari brush label per region, named in
  the config's `region_labels`, each interpolated separately and exported as
  one multi-label volume plus a `<output>.regions.json` sidecar recording which
  label is which region. That mapping is what lets `../Registration_ants` build
  the matching atlas-side outline automatically from the atlas annotation
  volume, so normally only the sample side is painted here.
- **`place_landmarks.py`** — click matching anatomical landmarks on the sample
  and on the atlas, in the same order, for `registration_eval.py`'s landmark TRE
  and for `fit_initial_transform.py`.
- **`fit_initial_transform.py`** — turn a pair of `place_landmarks.py` CSVs into
  the two deformation fields `../Registration_ants` starts its registration
  from, for the regions where intensity alone cannot establish correspondence.
  See below.
- **`single_sample.py`** — napari QC viewer for one registered sample: overlays
  the warped atlas, the sample, and per-region labels.
- **`align_masks.py`** — see below.
- **`registration_eval.py`** — Dice/HD95, landmark TRE, Jacobian and
  inverse-consistency metrics across samples and groups.
  `python registration_eval.py configs/eval_config.yaml`.

All the napari tools read images through SimpleITK, giving numpy order
`(z, y, x)` with axis 0 = the real imaging planes — deliberately **not**
`ants.image_read().numpy()`, whose axis order is reversed for the same file. Get
that wrong and you annotate left-right cross-sections with nothing to warn you.

### How every tool gets its paths

One convention, no paths hardcoded in any script:

```bash
cp configs/<tool>.example.yaml configs/<tool>.yaml    # then edit the paths
python <tool>.py                    # uses configs/<tool>.yaml
python <tool>.py configs/other.yaml # or an explicit one, e.g. per sample
```

`configs/*.yaml` is gitignored and only the `*.example.yaml` templates are
tracked, so changing samples never shows up as a git diff. `~` and `${VAR}` are
expanded, so one config works across machines with different mount points.

The tools that open a **form window** (`place_landmarks.py`,
`edit_sample_labels.py`, `fit_initial_transform.py`) treat the config as
optional: with one, its values pre-fill the form ahead of the last-used values
remembered in `.dialog_state/`; without one, the form behaves as it always did.
`--no-form` skips the window entirely and runs straight from the config.

`fit_initial_transform.py` additionally accepts all its paths as CLI flags — but
either give all of them or none, since half a command line makes "where does the
rest come from" ambiguous.

---

## Landmark-driven initialisation: `fit_initial_transform.py`

The P5 sample is flattened to 61% of the atlas's dorsoventral extent, and around
the deep midline structures there is no intensity feature for the registration to
lock onto. This script lets you state the correspondence by hand instead: place
matching points with `place_landmarks.py`, fit a deformation field through them,
and hand it to the pipeline as `registration.initial_transform`.

```bash
conda activate antsreg
python fit_initial_transform.py --selftest        # synthetic phantom, ~6 s

python fit_initial_transform.py \
    --sample-landmarks landmarks_sample.csv --atlas-landmarks landmarks_atlas.csv \
    --sample-image        registration.tif --sample-voxel-size 2.6 2.6 32.0 \
    --sample-domain-image tsc12t_fine_20um.nii.gz \
    --atlas-image  P04_LSFM_20um_1_-3_2__320-640_full_full.nii.gz \
    --out-prefix   /path/to/s12t_ --fitting-levels 5

python fit_initial_transform.py                   # no args -> the same form
```

Six things worth knowing before you use the output:

- **The two sides may be at different resolutions, and normally are.** Place the
  sample landmarks on the **raw acquisition TIFF** — the 20 µm resample drops xy
  by 8× and the structures stop being identifiable — and the atlas landmarks on
  the 20 µm atlas. The fit runs in physical coordinates, so each side only has to
  know its own voxel size. Cropping is not a problem either: `crop_for_registration`
  moves the origin, so physical space stays continuous.
- **A TIFF forces `--sample-voxel-size`.** TIFF headers carry no spacing and
  SimpleITK reports `(1.0, 1.0, 1.0)`, which would scale every physical coordinate
  wrong by 2.6–32× silently, so its absence is a hard error. The three values are
  **x y z** — the same order as `sample.voxel_size_um` in the pipeline config, and
  the reverse of the CSV's `axis-0/1/2`.
- **Two fields are written, not one.** ANTs cannot invert a deformation field it
  did not produce, so `<prefix>init_fwd.nii.gz` (atlas grid) and
  `<prefix>init_inv.nii.gz` (20 µm sample grid) are two independent fits of the same
  landmark pairs with the roles swapped. Without the second one the reverse chain
  (`labels_in_sample`, cell-to-region assignment) has a missing link.
- **`--sample-domain-image` is the grid the inverse field lives on**, and it is
  *not* the image the landmarks were placed on. Downstream that field warps atlas
  labels into sample space against `{name}_fine_{N}um.nii.gz`, so pass that file.
  Without it one is derived — from `--sample-image` if that is already a `.nii.gz`,
  otherwise a synthetic `--target-um` (default 20) isotropic grid padded 10 %
  beyond the acquisition extent.
- **`--atlas-image` must be the pipeline's reoriented+cropped atlas cache**, not a
  raw DevCCF release file — different axis order, different origin. Points that
  land outside the image are a hard error, as is a non-identity direction matrix.
- **Read the per-landmark residual table.** The two CSVs are paired by row order
  only, so a pair naming two different anatomical locations does not fail — it
  silently drags the field, and its residual is the only place it shows up.
  Outliers are marked `<-- CHECK`. Two different causes, distinguished for you:
  *uniformly* large residuals mean the B-spline lattice is too coarse (on the real
  atlas `--fitting-levels 4` leaves ~100 µm, level 5 gives 0.9 µm), while a few
  large ones among small ones mean a landmark ordering problem — unless the row
  sits near the fitting domain's edge, where the lattice is unconstrained and more
  levels will *not* help. The report says which case you are in.

Budget ~8 minutes, ~15 GB of RAM and ~2 GB of disk for a real 20 µm run.

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
python paint_mask.py --selftest                         # also fine in gt_sam
python fit_initial_transform.py --selftest
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
