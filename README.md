# GT_tool_for_registration

Ground-truth annotation, visual QC and evaluation tooling for the LSFM mouse-brain
registration project. The registration pipeline itself lives in
[`../Registration_ants`](../Registration_ants); this repo is everything a human
touches — painting masks, drawing region ground truth, looking at the result,
and scoring it.

These tools import `registration_ants` (via its `pip install -e` in the
`antsreg` env) but the pipeline does not import anything from here, so the
dependency only points one way.

---

## Layout

Three kinds of file, kept apart:

```
paint_mask.py            the three main scripts, run from the repo root
single_sample.py
registration_eval.py

shared/                  imported, never run
  local_config.py          configs/<tool>.yaml -> a dict, + the form window
  form_dialog.py           the Qt input form local_config falls back to
  landmark_io.py           the landmark CSV format
  atlas_reference.py       GUI-free atlas loading + ontology math
  label_partition.py       brush label <-> atlas region, and refining it per region
  ontology_tree_ui.py      the searchable Qt ontology tree

tools/                   the smaller runnable tools
  atlas_view.py
  edit_sample_labels.py
  qc_guide_mask.py
  convert_regions_ontology.py

configs/                 <tool>.example.yaml tracked, <tool>.yaml gitignored
tests/
```

`configs/` and `.dialog_state/` live at the **repo root**, not inside `shared/`,
so a tool in `tools/` and a main script in the root find the same ones. Anything
`__file__`-relative in `shared/` therefore anchors on `parents[1]`.

Run everything from the repo root:

```bash
python paint_mask.py
python tools/atlas_view.py
```

Scripts in `tools/` put the repo root on `sys.path` themselves before importing
from `shared/`, so this works with no `PYTHONPATH` and no package install.

---

## Environment

One env, `antsreg`. It already has napari + SimpleITK + antspyx + the editable
`registration_ants`, which is everything here needs:

```bash
conda env create -f environment-antsreg.yml
conda activate antsreg
pip install -e ../Registration_ants     # path-dependent, so not in the yml
```

Linux, macOS and Windows all work — `antspyx` ships `win_amd64` wheels, and
nothing here shells out to the ANTs command-line binaries, so there is no
native ANTs install to worry about. On Windows, run napari from a normal
terminal (Anaconda Prompt or PowerShell); it is a native GUI, no X11 involved.

---

## The main scripts

- **`paint_mask.py`** — paint the guide outline fed to `ants.registration`'s
  `multivariate_extras`. Two modes, chosen by `mode:` in the config; both
  export one multi-label volume plus a `<output>.regions.json` sidecar
  recording which brush label is which atlas region. That mapping is what lets
  `../Registration_ants` build the matching atlas-side outline automatically
  from the atlas annotation volume, so normally only the sample side is
  painted here.

  - `mode: guide` (default) — trace regions by hand on the raw sample, from
    blank planes, and assign each brush number to atlas structures in the
    ontology tree. Sparse keyframes, each label interpolated separately.
  - `mode: labels` — start from a finished registration
    (`<name>_labels_in_sample.nii.gz`) collapsed into a *partition* of brush
    labels, and correct only what came out wrong. A plane that differs from
    the registration anywhere becomes a whole-plane keyframe (every region on
    it, not just the edit), and two volumes come out: a sparse guide to
    re-register with, and a dense one to re-open and carry on from. The
    partition is refined one region at a time — expand `Hippocampal
    formation` into CA/DG without changing how coarsely the cerebellum is
    described — and the atlas-side subtraction that nesting requires
    (`atlas_exclude_ids`) is derived rather than maintained by hand. See
    `shared/label_partition.py` for the measured reason a uniform ontology
    depth is not a usable knob.
- **`single_sample.py`** — napari QC viewer for one registered sample: overlays
  the warped atlas, the sample, and per-region labels. Reads either pipeline's
  output directory, auto-detected: ANTs (`*_fine_*um.nii.gz` +
  `*_labels_in_sample.nii.gz`) or ClearMap's own `cellMap.py`
  (`resampled.tif` + `volume/result.mhd`). Two things differ between them and
  are handled automatically — ClearMap registers on a *cropped* copy of the
  resampled image, so its warped label volume is shifted relative to the cell
  coordinates (the offset is read back from the run's `log.txt`); and its
  `cell_registration.csv` stores `graph_order` rather than raw atlas ids, which
  collide numerically with ids for most regions (the id space is decided per
  file from the region-name column). Missing pieces degrade instead of
  disabling the view: with no sample-space label volume there is no region
  outline and no hover lookup, but the cell points, the region search and the
  per-region filter all still work, since those come from the CSV.
- **`registration_eval.py`** — Dice/HD95, landmark TRE, Jacobian and
  inverse-consistency metrics across samples and groups.
  `python registration_eval.py configs/eval_config.yaml`.

---

## The tools

### `tools/edit_sample_labels.py` — hand-correct an existing registration's labels

Starts from `labels_in_sample.nii.gz` (a completed registration) and lets you
repaint whichever regions came out wrong on a few representative planes, then
interpolates the correction across the volume — no re-registration involved.
"Save This Region" exports one region as a binary mask plus an
`.annotated_slices.json` sidecar, which is exactly what `registration_eval.py`'s
`dice_region_masks` consumes. This is where the Dice/HD95 ground truth comes
from.

```bash
python tools/edit_sample_labels.py              # opens a path form
python tools/edit_sample_labels.py --no-form    # straight from the config
```

### `tools/atlas_view.py` — browse an atlas against its ontology

Read-only. Three synced ortho panes (grayscale template, full annotation in
colour, and whatever the ontology tree selects), an ontology panel that can
highlight several regions at once as a union, and a hover panel showing the
full ancestor chain under the mouse. Nothing is written and nothing is
registered.

The panes are three mutually orthogonal **planes** that tilt as a rigid frame,
not the atlas's own voxel axes — shift+drag inside a pane rotates the frame
about that pane's normal, so a sample cut at an angle can be matched by
reslicing the atlas at that angle instead of eyeballing it between two
axis-aligned slices.

```bash
python tools/atlas_view.py
python tools/atlas_view.py configs/atlas_view.devccf.yaml
python tools/atlas_view.py --selftest      # plane geometry, no display needed
```

### `tools/qc_guide_mask.py` — check a guide mask before you register with it

Catches the two silent failure modes of a hand-painted
`guide_regions.regions_mask`: a **stray keyframe** (one accidental brush click
registers that plane as a keyframe, and the signed-distance interpolation then
collapses the region across both intervals around it), and a **pairing
mismatch** against the atlas structure the guide will be matched to. Both
otherwise end in a run that finishes, a metric that goes down, and labels that
come out wrong.

```bash
python tools/qc_guide_mask.py                  # uses configs/qc_guide_mask.yaml
python tools/qc_guide_mask.py --napari         # + open the mask in napari
python tools/qc_guide_mask.py --no-atlas       # interpolation only, no atlas
```

### `tools/convert_regions_ontology.py` — DevCCF ↔ Allen CCFv3 for a painted mask

Translates a painted mask's `.regions.json` `region_ids` between the two
ontologies, for when the same hand-painted sample gets registered against a
different atlas (e.g. moving from DevCCF P04 to DeMBA P5, which is labelled in
CCFv3). The painted outlines are never touched — only the atlas-side pairing,
which is the only thing that has to change.

It reads the DevCCF paper's voxel-overlap crosswalk (Supplementary Data 3),
expands the painted coarse node down to the labels the crosswalk actually
covers, then aggregates that mass back up the target ontology and reports both
**share** (how much of what you painted a target node accounts for) and
**purity** (how much of that node is the thing you painted). Two-sided on
purpose: DevCCF `midbrain` puts 96.7% of its mass inside CCFv3 `MB`, but `MB` is
~1.9× its volume, and pairing a guide against a structure twice its size is
worse than no guide — so such a node is reported as a coarse alternative for a
human to accept, never picked silently. Prints a ready-to-paste `atlas_ids`
block; `-o`/`--in-place` also writes a converted sidecar `paint_mask.py` can
resume from.

```bash
python tools/convert_regions_ontology.py \
    ../Registration_ants/atlas/mask/s12t_guide7.regions.json
```

---

## Conventions

### Axis order

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

`tools/edit_sample_labels.py` opens a **form window** and treats the config as
optional: with one, its values pre-fill the form ahead of the last-used values
remembered in `.dialog_state/`; without one, the form behaves as it always did.
`--no-form` skips the window entirely and runs straight from the config.

### Landmarks for `registration_eval.py`'s TRE

Placed in napari by hand — open the sample and the atlas template, add a Points
layer to each, click the SAME anatomical points in the SAME order in both, and
save each layer to CSV. Row *i* in one file and row *i* in the other are one
landmark; nothing in the format names them, so the pairing is the row order and
only the row order. `shared/landmark_io.py` reads and validates both sides.

### Mask geometry before any metric

A mask that lost its spacing/origin/direction on export looks pixel-perfect in a
viewer and silently poisons every surface-distance metric. Masks written by
`tools/edit_sample_labels.py` inherit the source geometry, but anything that
went through another annotation tool (ilastik, ITK-SNAP) is worth checking
before it reaches `registration_eval.py`.

Surface distance is reported in **microns, not voxels**, and the trap there is
axis order: `np.argwhere` on a `GetArrayFromImage` array yields `(z, y, x)`,
while `GetSpacing()` is `(x, y, z)` — the reverse. So the conversion reverses
the spacing before multiplying:

```python
spacing_zyx = np.asarray(img.GetSpacing())[::-1]
points_um   = np.argwhere(border) * spacing_zyx
```

Under isotropic spacing a transposed vector still gives the right answer by
luck — which is why the selftests use **anisotropic** spacing and shift along
**all three axes**, and cross-check against SimpleITK's own
`TransformIndexToPhysicalPoint`.

---

## Tests

```bash
conda activate antsreg
python paint_mask.py --selftest
python tools/atlas_view.py --selftest
python shared/atlas_reference.py --selftest
python tests/test_tool_inputs_smoke.py
python tests/test_registration_eval_smoke.py
```

Manual-assert style, no pytest, matching `../Registration_ants/tests/`.
