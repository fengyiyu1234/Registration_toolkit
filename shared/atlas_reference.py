"""Shared, GUI-free atlas loading + ontology math.

Used by two independent GUI tools that both need to know about an atlas's
ontology and annotation volume, for different reasons:

  paint_mask.py assigns ontology structures to brush labels -- it needs the
    tree plus per-node voxel counts (to grey out regions this annotation has
    no voxels for), but never displays the atlas itself.

  tools/atlas_view.py renders the atlas as a three-pane viewer with a region
    picker of its own -- it needs the same tree, plus the grayscale
    template and a per-region highlight mask, and optionally a SAMPLE volume
    (load_sample_volume) to hold the atlas up against, which is an ordinary
    grayscale grid with no ontology attached to it at all.

Splitting this out of paint_mask.py is what lets tools/atlas_view.py exist as an
independent script instead of a window paint_mask.py opens, and it is also
what keeps this half importable with no PyQt5/napari -- the same reason
paint_mask.py's own _interpolate_sparse_mask() is imported lazily inside a
function rather than at module scope: a --selftest of the ontology math
should run with nothing but numpy/SimpleITK -- no display, and no
../Registration_ants editable install required.

Not runnable as a tool -- `python shared/atlas_reference.py --selftest` runs the
synthetic tests below, but the actual atlas configs live in paint_mask.py's
and tools/atlas_view.py's own configs/*.yaml, each pointing at the same
atlas_annotation_path / ontology_path keys.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk


def _atlas_helpers():
    """registration_ants.atlas_utils -- the ontology half of it is
    deliberately free of the antspyx import, which is what lets this
    GUI-free module reuse it."""
    from registration_ants import atlas_utils
    return atlas_utils


def atlas_reference_config(cfg):
    """Resolve the optional atlas reference -> SimpleNamespace, or None when no
    atlas is configured.

    Only the annotation + ontology are strictly needed -- they are what the
    region picker resolves against. The template is just the grayscale
    backdrop and may be omitted; resolution_um and orientation only affect how
    the atlas is drawn.

    The ontology has to be the SAME one the pipeline registers against, since
    what any export carries over is ontology ids. The paths themselves do not:
    the atlas here is never registered to a sample, so a whole-brain atlas
    beside a half-brain sample is the normal case.
    """
    paths = {
        "template_path": cfg.get("atlas_template_path"),
        "annotation_path": cfg.get("atlas_annotation_path"),
        "ontology_path": cfg.get("ontology_path") or cfg.get("atlas_ontology_path"),
    }
    if not any(paths.values()):
        return None

    for required in ("annotation_path", "ontology_path"):
        if not paths[required]:
            key = "ontology_path" if required == "ontology_path" else f"atlas_{required}"
            raise ValueError(f"the atlas reference needs {key} (missing from the config)")
        if not Path(paths[required]).exists():
            raise FileNotFoundError(f"atlas {required} does not exist: {paths[required]}")
    if paths["template_path"] and not Path(paths["template_path"]).exists():
        raise FileNotFoundError(f"atlas template_path does not exist: {paths['template_path']}")

    downsample = cfg.get("atlas_downsample")
    downsample = 1 if downsample is None else int(downsample)
    if downsample < 1:
        raise ValueError(f"atlas_downsample must be >= 1, got {downsample}")

    return SimpleNamespace(
        template_path=paths["template_path"] or None,
        annotation_path=paths["annotation_path"],
        ontology_path=paths["ontology_path"],
        resolution_um=float(cfg.get("atlas_resolution_um") or 0) or None,
        orientation=cfg.get("atlas_orientation") or None,
        downsample=downsample,
        # Three synced canvases instead of one -- only meaningful to
        # tools/atlas_view.py's window; paint_mask.py's tree-only use ignores it.
        ortho=bool(cfg.get("atlas_ortho_views", True)),
    )


def _read_array(path):
    """Just the voxels, with the SimpleITK image dropped immediately.

    GetArrayFromImage copies, so the ITK buffer and the numpy array are both
    live until the image goes out of scope -- and binding it to a throwaway
    `_` keeps it alive for the rest of the enclosing function. That is 1.1 GB
    of nothing for the DevCCF annotation, doubling peak memory during the
    atlas load for a handle the atlas path never uses.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _reorient_zyx(arr_zyx, orientation, atlas_utils):
    """Apply a ClearMap-style orientation spec to a (z,y,x) array.

    atlas_utils.reorient_volume operates on (x,y,z)-ordered arrays -- the
    order ants.image_read().numpy() produces -- while everything here is
    (z,y,x), the order sitk.GetArrayFromImage produces. So the array is
    flipped into (x,y,z), reoriented, and flipped back. Getting this wrong
    would silently show the atlas permuted relative to the pipeline's view
    of it.

    Purely cosmetic: nothing that reads ontology ids off this atlas cares
    about voxel layout, so no orientation mistake here can reach an export.
    It matters only so the reference LOOKS like the atlas the pipeline uses.

    Returns a VIEW, not a copy -- transposes and flips are both views, and
    the caller downsamples before materializing anything. Copying here
    instead would double peak memory on a volume that is 1.1 GB to begin
    with.
    """
    if not orientation:
        return arr_zyx
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    arr_xyz = atlas_utils.reorient_volume(arr_xyz, orientation)
    return np.transpose(arr_xyz, (2, 1, 0))


def _reoriented_axis_order(orientation, atlas_utils):
    """Which SOURCE axis each axis of a reoriented (z,y,x) array came from.

    Only anisotropic voxels need this: a light-sheet sample is routinely 5 um
    in plane and 25 um between sheets, so its three voxel sizes are three
    different numbers and reorienting the array permutes them along with the
    axes. Rather than reimplement reorient_volume's permutation here -- where
    it could silently drift from the real one and quietly stretch a volume
    along the wrong axis -- this runs the ACTUAL reorientation on a probe whose
    three axis lengths are distinct, and reads the permutation back off the
    resulting shape.
    """
    probe = np.zeros((2, 3, 4), dtype=np.uint8)
    reoriented = _reorient_zyx(probe, orientation, atlas_utils).shape
    return tuple(probe.shape.index(int(n)) for n in reoriented)


def sample_volume_config(cfg):
    """Resolve the optional SAMPLE volume -> SimpleNamespace, or None.

    A second, ordinary grayscale volume for tools/atlas_view.py to hold the
    atlas up against: the user's own brain, unregistered and never rotated.
    Nothing about the ontology or the annotation depends on it, which is why
    it is optional and why nothing else in this module reads it -- an atlas
    browser with no sample configured behaves exactly as it did before.

    `sample_resolution_um` is a scalar or three numbers in the FILE's own
    (x, y, z) order -- the same order ../Registration_ants writes
    `sample.voxel_size_um` in -- and is optional: a NIfTI written by the
    pipeline already carries its spacing, and load_sample_volume falls back to
    that. `sample_orientation` is the same ClearMap-style spec the atlas uses,
    for the usual case of a sample whose axes are not stored in the order the
    atlas's are.
    """
    path = cfg.get("sample_path")
    if not path:
        return None
    if not Path(path).exists():
        raise FileNotFoundError(f"sample_path does not exist: {path}")

    downsample = cfg.get("sample_downsample")
    downsample = 1 if downsample is None else int(downsample)
    if downsample < 1:
        raise ValueError(f"sample_downsample must be >= 1, got {downsample}")

    return SimpleNamespace(
        path=str(path),
        resolution_um=cfg.get("sample_resolution_um"),
        orientation=cfg.get("sample_orientation") or None,
        downsample=downsample,
    )


def _sample_voxel_zyx(resolution_um, spacing_xyz, path):
    """Per-axis voxel size of the sample AS STORED, in (z,y,x) order.

    Config first, file second: `sample_resolution_um` is written in (x, y, z)
    to match the pipeline's own `voxel_size_um`, and reversed here into the
    (z,y,x) order sitk.GetArrayFromImage produces. With nothing configured the
    file's own spacing is used -- right for anything the pipeline wrote, and
    (1, 1, 1) for a plain TIFF, which is flagged rather than silently believed
    because it would make every micron figure in the viewer meaningless.
    """
    if resolution_um is None:
        if all(abs(s - 1.0) < 1e-9 for s in spacing_xyz):
            print(f"WARNING: {Path(path).name} carries no voxel size and none is configured; "
                  f"assuming 1 um. Set sample_resolution_um for the overlay to be to scale.")
        return tuple(float(s) for s in reversed(spacing_xyz))

    values = ([float(resolution_um)] * 3 if np.isscalar(resolution_um)
              else [float(v) for v in resolution_um])
    if len(values) != 3:
        raise ValueError("sample_resolution_um must be one number or three (x, y, z), "
                         f"got {resolution_um}")
    return tuple(reversed(values))


def load_sample_volume(sample_cfg):
    """Load the sample volume the atlas is to be compared against.

    Returns SimpleNamespace(volume, voxel_um, path, downsample): a contiguous
    (z,y,x) array in the SAME axis order the atlas reference is loaded in
    (both go through _reorient_zyx), and its per-axis voxel size in microns,
    permuted and scaled to match that array rather than the file on disk.

    The array is materialized rather than left as a reoriented view: unlike
    the atlas, every plane of this one is re-read on every mouse move, and a
    transposed view turns each of those into a strided gather over the whole
    volume.
    """
    atlas_utils = _atlas_helpers()

    print(f"[sample] volume: {sample_cfg.path}")
    image = sitk.ReadImage(str(sample_cfg.path))
    spacing_xyz = tuple(float(s) for s in image.GetSpacing())
    array = sitk.GetArrayFromImage(image)
    del image                                   # see _read_array

    voxel_zyx = _sample_voxel_zyx(sample_cfg.resolution_um, spacing_xyz, sample_cfg.path)
    step = sample_cfg.downsample
    sub = (slice(None, None, step),) * 3
    volume = np.ascontiguousarray(
        _reorient_zyx(array, sample_cfg.orientation, atlas_utils)[sub])
    del array

    order = _reoriented_axis_order(sample_cfg.orientation, atlas_utils)
    voxel_um = np.array([voxel_zyx[axis] for axis in order], dtype=float) * step

    extent = np.array(volume.shape, dtype=float) * voxel_um / 1000.0
    print(f"[sample] grid {volume.shape}, "
          f"voxels {tuple(round(float(v), 3) for v in voxel_um)} um, "
          f"extent {tuple(round(float(e), 2) for e in extent)} mm"
          + (f", downsampled {step}x" if step > 1 else ""))

    return SimpleNamespace(volume=volume, voxel_um=voxel_um,
                           path=sample_cfg.path, downsample=step)


def _compact_annotation(annotation_zyx, chunk=64):
    """(compact, present_ids) -- annotation relabelled to small consecutive
    indices into `present_ids`, so highlight lookups run on a uint8/uint16
    array instead of the raw one.

    The DevCCF P04 annotation reads back as float32 at 800x560x640: 1.1 GB
    resident. It only carries 193 distinct labels, so an index array costs
    287 MB (uint8) and makes np.isin cheap besides.

    Everything here runs a slab at a time, including the label scan, because
    every obvious one-shot spelling materializes another array the size of
    the input or worse: np.unique sorts a full flattened copy, and
    np.unique(return_inverse=True) / np.searchsorted over the whole array both
    return int64 -- 2.3 GB for this atlas, worse than the problem being
    solved. `annotation_zyx` is also typically a non-contiguous reoriented
    VIEW (see _reorient_zyx), so a whole-array operation would quietly
    materialize that too.
    """
    present = set()
    for z0 in range(0, annotation_zyx.shape[0], chunk):
        present.update(np.unique(annotation_zyx[z0:z0 + chunk]).tolist())
    present_ids = np.array(sorted(present))

    dtype = np.uint8 if len(present_ids) <= np.iinfo(np.uint8).max + 1 else np.uint16
    compact = np.empty(annotation_zyx.shape, dtype=dtype)
    for z0 in range(0, annotation_zyx.shape[0], chunk):
        sl = slice(z0, z0 + chunk)
        compact[sl] = np.searchsorted(present_ids, annotation_zyx[sl]).astype(dtype)
    return compact, present_ids.astype(np.int64)


def voxels_per_ontology_node(structures, present_ids, counts):
    """{structure id: voxel count including descendants} for EVERY ontology
    node, from the per-label counts of the annotation.

    Every voxel is credited to its own label and to each of that label's
    ancestors (structure_id_path is the full root->node chain), so a node at
    any depth reports the size of the whole subtree under it -- which is
    exactly what a picker highlights when that node is clicked.

    Nodes absent from the result have no voxels in this annotation at all.
    That is the common case, not an edge case: the DevCCF P04 annotation
    carries 193 of the ontology's 2552 structures. Assigning one of the empty
    ones would make the pipeline refuse the whole run
    (_build_guide_regions_from_labels raises when a region matches nothing),
    so callers grey them out instead of letting the user pick them.
    """
    totals = {}
    for label, count in zip(present_ids, counts):
        info = structures.get(int(label))
        if info is None:            # background (0), or a label this ontology doesn't describe
            continue
        for ancestor in info["structure_id_path"]:
            totals[ancestor] = totals.get(ancestor, 0) + int(count)
    return totals


def load_atlas_reference(atlas_cfg, include_template=True):
    """Load the atlas ontology (+ annotation, + optionally the grayscale
    template) that a region picker resolves against.

    include_template=False skips reading atlas_cfg.template_path entirely --
    paint_mask.py's region-assignment tree never displays it (that lives in
    the separate tools/atlas_view.py now), and the DevCCF template is another
    ~1 GB SimpleITK read plus a downsample/reorient pass it has no use for.

    Returns SimpleNamespace(template, compact, present_ids, index_of_id,
    structures, node_voxels, downsample) -- or raises with a message pointing
    at the offending config key. `template` is None when include_template is
    False, or when no template_path is configured; a tree-only caller works
    without it, a display caller just gets no grayscale underneath.
    """
    atlas_utils = _atlas_helpers()
    structures = atlas_utils.load_ccf_ontology_json(atlas_cfg.ontology_path)

    step = atlas_cfg.downsample
    sub = (slice(None, None, step),) * 3

    print(f"[atlas] annotation: {atlas_cfg.annotation_path}")
    annotation = _read_array(atlas_cfg.annotation_path)
    # Reorient and downsample as views, so the only full-size array alive is
    # the one SimpleITK just read; _compact_annotation reads through the view
    # a slab at a time and the 1.1 GB original is dropped immediately after.
    compact, present_ids = _compact_annotation(
        _reorient_zyx(annotation, atlas_cfg.orientation, atlas_utils)[sub])
    del annotation

    counts = np.bincount(compact.ravel(), minlength=len(present_ids))
    node_voxels = voxels_per_ontology_node(structures, present_ids, counts)

    template = None
    if include_template and atlas_cfg.template_path:
        print(f"[atlas] template:   {atlas_cfg.template_path}")
        raw_template = _read_array(atlas_cfg.template_path)
        template = np.ascontiguousarray(
            _reorient_zyx(raw_template, atlas_cfg.orientation, atlas_utils)[sub])
        del raw_template
        if template.shape != compact.shape:
            print(f"WARNING: atlas template shape {template.shape} != annotation shape "
                  f"{compact.shape}; showing the annotation only.")
            template = None

    known = sum(1 for sid in present_ids if int(sid) in structures)
    print(f"[atlas] ontology: {len(structures)} structures, {len(present_ids)} labels in the "
          f"annotation ({known} of them in the ontology), {len(node_voxels)} nodes non-empty; "
          f"grid {compact.shape}" + (f", downsampled {step}x" if step > 1 else ""))

    return SimpleNamespace(
        template=template,
        compact=compact,
        present_ids=present_ids,
        index_of_id={int(sid): i for i, sid in enumerate(present_ids)},
        structures=structures,
        node_voxels=node_voxels,
        downsample=step,
    )


def highlight_mask(atlas, structure_id):
    """Binary (z,y,x) mask of one ontology node INCLUDING all its descendants.

    Descendants are resolved through structure_id_path, never by name, so
    picking a node at any depth lights up the whole subtree -- which is the
    only way a depth-3 node can highlight anything at all here, since the
    DevCCF annotation's own labels sit at depths 2 through 12 and a parent
    node usually owns no voxels under its own id.

    Resolved through atlas_utils.descendant_ids_of, the same function the
    pipeline's atlas_ids path uses, so what lights up here is exactly what
    registration would pair against.
    """
    wanted = _atlas_helpers().descendant_ids_of(atlas.structures, [structure_id])
    indices = [atlas.index_of_id[sid] for sid in wanted if sid in atlas.index_of_id]
    if not indices:
        return np.zeros(atlas.compact.shape, dtype=np.uint8)
    return np.isin(atlas.compact, indices).astype(np.uint8)


def mask_centre_index(mask):
    """(i0, i1, i2) voxel index at the middle of `mask`'s extent on each axis,
    or None when the mask is empty.

    The median-of-occupied-planes per axis, not the centre of mass: for a
    C-shaped or bilateral structure the centre of mass can land in a plane the
    structure never touches, and the whole point of jumping there is to see it
    in all three panes at once.
    """
    centre = []
    for axis in range(mask.ndim):
        planes = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis)))
        if not planes.size:
            return None
        centre.append(int(planes[len(planes) // 2]))
    return tuple(centre)


def format_ancestry(structures, structure_id):
    """The whole root -> leaf chain of `structure_id`, one level per line.

    What a status bar can show is the ONE structure a voxel is labelled
    with, which for a fine-grained annotation is a leaf like "layer 5 of
    primary motor area" -- true, and useless for deciding whether you are
    looking at cortex. The ontology already carries the answer in
    structure_id_path; this just renders it, so the level you actually care
    about is on screen next to the level the annotation happens to store.

    The voxel's own structure is marked, since it is usually NOT the last line
    a user cares about, and every line carries its id -- ids are what an
    export records, so being able to read one off directly is the point.
    """
    info = structures.get(structure_id)
    if info is None:
        return (f"id {structure_id}: not in the ontology (the annotation uses this label, "
                "but no structure describes it)")
    lines = []
    for depth, sid in enumerate(info["structure_id_path"]):
        node = structures.get(sid)
        marker = "▶" if sid == structure_id else "·"
        indent = "  " * depth
        if node is None:
            lines.append(f"{indent}{marker} [{sid}] ?")
            continue
        acronym = node.get("acronym")
        name = f"{node['name']} ({acronym})" if acronym else node["name"]
        lines.append(f"{indent}{marker} {name}  [{sid}]")
    return "\n".join(lines)


def annotation_features(atlas):
    """Hover text for a full-annotation napari Labels layer, one row per
    compact label.

    That layer holds COMPACT indices (see _compact_annotation), which mean
    nothing on their own -- index 37 is not structure 37. napari's Labels
    layer appends a features row to the status bar next to the value under
    the cursor, matched through the 'index' column, so mapping compact index
    -> the real structure's name and id here is what makes the layer readable
    without hunting for the label in the tree.
    """
    names, ids = [], []
    for sid in atlas.present_ids:
        sid = int(sid)
        info = atlas.structures.get(sid)
        if sid == 0:
            names.append("background")
        elif info is None:
            names.append(f"id {sid} is not in the ontology")
        else:
            acronym = info.get("acronym")
            names.append(f"{info['name']} ({acronym})" if acronym else info["name"])
        ids.append(sid)
    return {"index": np.arange(len(atlas.present_ids)), "region": names, "id": ids}


def visible_tree_ids(structures, node_voxels, text, hide_empty):
    """Which ontology ids a search box showing `text` should leave visible.

    Every ancestor of a match is included, otherwise a deep hit would be
    unreachable -- the tree can only show a node if the whole chain down to
    it is showing. Matching is on name and acronym, since the acronym
    (DevCCF's "SPall", "THyA", ...) is often what you actually remember.
    """
    text = text.strip().lower()
    visible = set()
    for sid, info in structures.items():
        if hide_empty and not node_voxels.get(sid):
            continue
        if text and text not in info["name"].lower() and text not in (info.get("acronym") or "").lower():
            continue
        visible.update(info["structure_id_path"])
    return visible


# =====================================================================================
# selftests -- synthetic data only, no GUI, no display, no atlas files on disk
# =====================================================================================
def _fake_ontology():
    """A 6-node ontology shaped exactly like load_ccf_ontology_json's output.
    Branch B is the "in the ontology but not in this annotation" case that
    the real DevCCF pairing has 2359 of."""
    return {
        1:   {"id": 1,   "name": "root",     "acronym": "R",  "structure_id_path": [1]},
        10:  {"id": 10,  "name": "branch A", "acronym": "BA", "structure_id_path": [1, 10]},
        100: {"id": 100, "name": "leaf A1",  "acronym": "A1", "structure_id_path": [1, 10, 100]},
        101: {"id": 101, "name": "leaf A2",  "acronym": "A2", "structure_id_path": [1, 10, 101]},
        20:  {"id": 20,  "name": "branch B", "acronym": "BB", "structure_id_path": [1, 20]},
        200: {"id": 200, "name": "leaf B1",  "acronym": "B1", "structure_id_path": [1, 20, 200]},
    }


def selftest_ontology_node_voxels():
    print("1. ontology: voxels roll up to every ancestor; empty subtrees stay empty")
    structures = _fake_ontology()
    # Label 0 is background and has no ontology entry -- it must not crash and
    # must not be credited to anything.
    totals = voxels_per_ontology_node(structures, [0, 100, 101], [50, 3, 7])

    assert totals == {1: 10, 10: 10, 100: 3, 101: 7}, totals
    assert 20 not in totals and 200 not in totals, totals

    # The invariant a tree builder relies on to disable empty nodes without
    # ever hiding a pickable descendant: a node with no voxels can have no
    # non-empty descendant, because every voxel is credited to the whole
    # ancestor chain.
    for sid, info in structures.items():
        if not totals.get(sid):
            descendants = [d for d, i in structures.items() if sid in i["structure_id_path"]]
            assert not any(totals.get(d) for d in descendants), (sid, descendants)
    print("   ok")


def selftest_ontology_tree_filter():
    print("2. ontology: search reveals ancestors, matches acronyms, respects hide-empty")
    structures = _fake_ontology()
    node_voxels = {1: 10, 10: 10, 100: 3, 101: 7}

    assert visible_tree_ids(structures, node_voxels, "", True) == {1, 10, 100, 101}
    assert visible_tree_ids(structures, node_voxels, "", False) == {1, 10, 100, 101, 20, 200}

    # A deep hit is unreachable unless its whole ancestor chain shows too.
    assert visible_tree_ids(structures, node_voxels, "leaf A2", True) == {1, 10, 101}
    # Acronyms are searchable: "SPall"/"THyA" is often what you remember.
    assert visible_tree_ids(structures, node_voxels, "a1", True) == {1, 10, 100}

    # An empty region stays hidden while hide-empty is on, and is reachable
    # (to look at, not to assign) once it's off.
    assert visible_tree_ids(structures, node_voxels, "leaf B1", True) == set()
    assert visible_tree_ids(structures, node_voxels, "leaf B1", False) == {1, 20, 200}
    print("   ok")


def selftest_compact_annotation():
    print("3. atlas: chunked relabelling is exact, and small enough to be worth it")
    rng = np.random.default_rng(3)
    ids = np.array([0, 7, 15564, 21558], dtype=np.float32)   # real DevCCF-scale ids
    annotation = ids[rng.integers(0, len(ids), size=(70, 12, 9))].astype(np.float32)

    compact, present_ids = _compact_annotation(annotation, chunk=16)   # chunk < z, so it wraps

    assert np.array_equal(present_ids, ids.astype(np.int64)), present_ids
    assert compact.dtype == np.uint8, compact.dtype
    # Round-tripping through the index must reproduce the annotation exactly:
    # a chunk-boundary bug would corrupt only some planes, which is precisely
    # the kind of thing that would go unnoticed in the GUI.
    assert np.array_equal(present_ids[compact], annotation.astype(np.int64))

    counts = np.bincount(compact.ravel(), minlength=len(present_ids))
    assert counts.sum() == annotation.size
    for i, sid in enumerate(present_ids):
        assert counts[i] == int((annotation == sid).sum()), (sid, counts[i])

    # >255 distinct labels has to widen, or ids would silently collide.
    many = np.arange(300, dtype=np.float32).reshape(300, 1, 1) * np.ones((300, 2, 2), np.float32)
    wide, wide_ids = _compact_annotation(many, chunk=32)
    assert wide.dtype == np.uint16, wide.dtype
    assert np.array_equal(wide_ids[wide], many.astype(np.int64))
    print("   ok")


def selftest_annotation_features():
    print("4. atlas layer: compact index -> real region name/id hover table")
    atlas = SimpleNamespace(
        present_ids=np.array([0, 5, 7, 999], dtype=np.int64),
        structures={5: {"name": "Cortex", "acronym": "CTX"}, 7: {"name": "Thalamus"}})
    feats = annotation_features(atlas)
    # napari matches a features row to the value under the cursor through the
    # 'index' column, so this has to be the COMPACT index, not the real id.
    assert list(feats["index"]) == [0, 1, 2, 3], feats["index"]
    assert feats["id"] == [0, 5, 7, 999], feats["id"]
    assert feats["region"][1] == "Cortex (CTX)", feats["region"]
    assert feats["region"][2] == "Thalamus", feats["region"]
    assert "999" in feats["region"][3], feats["region"]  # in the annotation, not in the ontology
    print("   ok")


def selftest_format_ancestry():
    print("5. atlas hover: shows the whole ancestor chain, not just the finest level")
    structures = {
        1: {"name": "root", "acronym": "root", "structure_id_path": [1]},
        2: {"name": "Cerebrum", "acronym": "CH", "structure_id_path": [1, 2]},
        5: {"name": "Cortex", "acronym": "CTX", "structure_id_path": [1, 2, 5]},
        9: {"name": "layer 5", "acronym": None, "structure_id_path": [1, 2, 5, 9]},
    }
    text = format_ancestry(structures, 9)
    lines = text.splitlines()
    assert len(lines) == 4, lines
    # Root first, leaf last, every ancestor present with its id -- the ids are
    # what an export records, so they have to be readable off this panel.
    for sid in (1, 2, 5, 9):
        assert f"[{sid}]" in text, (sid, text)
    assert lines[0].startswith("· root"), lines[0]
    assert "▶" in lines[-1] and lines[-1].count("▶") == 1, lines[-1]
    assert sum("▶" in line for line in lines) == 1, lines
    assert "(CTX)" in lines[2], lines[2]      # acronym shown when the node has one
    assert "(None)" not in lines[3], lines[3] # and not faked when it does not

    # A mid-level pick marks itself, not the deepest node it knows about.
    assert "▶ Cortex" in format_ancestry(structures, 5)
    # A label the annotation carries but the ontology never describes.
    assert "not in the ontology" in format_ancestry(structures, 4242)
    print("   ok")


def selftest_mask_centre_index():
    print("6. atlas jump centre: a C-shaped structure lands inside the region, "
          "not in the hollow at its centre of mass")
    # A C-shaped structure: the centre of MASS of this mask sits at z=2 in the
    # hollow, which is exactly the plane where nothing would be visible.
    mask = np.zeros((5, 9, 7), dtype=np.uint8)
    mask[0, 1:8, 1:6] = 1
    mask[4, 1:8, 1:6] = 1
    centre = mask_centre_index(mask)
    assert mask[centre] == 1, (centre, "jump landed outside the region")
    assert centre == (4, 4, 3), centre

    solid = np.zeros((5, 9, 7), dtype=np.uint8)
    solid[1:4, 2:5, 3:6] = 1
    assert mask_centre_index(solid) == (2, 3, 4), mask_centre_index(solid)
    assert mask_centre_index(np.zeros((3, 3, 3), dtype=np.uint8)) is None
    print("   ok")


def selftest_sample_voxel_order():
    print("7. sample: voxel sizes follow the axes through a reorientation")
    # Config order is the pipeline's (x, y, z); the array is (z, y, x).
    assert _sample_voxel_zyx([2.6, 2.6, 32.0], (1.0, 1.0, 1.0), "cfg.tif") == (32.0, 2.6, 2.6)
    assert _sample_voxel_zyx(20, (1.0, 1.0, 1.0), "cfg.tif") == (20.0, 20.0, 20.0)
    # Nothing configured -> the file's own spacing, likewise reversed.
    assert _sample_voxel_zyx(None, (2.6, 2.6, 32.0), "file.nii") == (32.0, 2.6, 2.6)

    try:
        atlas_utils = _atlas_helpers()
    except ImportError:
        print("   ok (axis permutation skipped: registration_ants not importable)")
        return
    assert _reoriented_axis_order(None, atlas_utils) == (0, 1, 2)
    # A volume whose axis sizes are all different is its own witness: reorient
    # it, and the shape says where every axis (and so every voxel size) went.
    for orientation in ([1, 3, 2], [1, -3, 2], [-2, 1, 3], [3, 2, 1]):
        order = _reoriented_axis_order(orientation, atlas_utils)
        assert sorted(order) == [0, 1, 2], (orientation, order)
        probe = np.zeros((5, 7, 9), dtype=np.uint8)
        shape = _reorient_zyx(probe, orientation, atlas_utils).shape
        assert shape == tuple(probe.shape[a] for a in order), (orientation, order, shape)
    print("   ok")


def run_selftests():
    print("=== shared/atlas_reference.py selftests (synthetic data only, no GUI) ===")
    selftest_ontology_node_voxels()
    selftest_ontology_tree_filter()
    selftest_compact_annotation()
    selftest_annotation_features()
    selftest_format_ancestry()
    selftest_mask_centre_index()
    selftest_sample_voxel_order()
    print("=== all selftests passed ===")
    return 0


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Shared atlas/ontology loading used by paint_mask.py and tools/atlas_view.py "
                     "(not a standalone tool -- this only runs its selftests)")
    parser.add_argument("--selftest", action="store_true",
                        help="run the built-in synthetic tests (no GUI, no atlas files) and exit")
    args = parser.parse_args()
    if not args.selftest:
        parser.print_help()
        return 1
    return run_selftests()


if __name__ == "__main__":
    raise SystemExit(main())
