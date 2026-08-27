"""A *partition*: the brush-label -> ontology-region mapping that both of
paint_mask.py's modes export, and the tree operations that let it be
refined region-by-region instead of at one uniform ontology depth.

Why not just pick an ontology depth. `atlas_utils.collapse_labels_to_level`
maps every voxel to its ancestor at depth N, which sounds like the natural
way to choose "how finely am I correcting". Measured against the real
DeMBA P5 annotation + CCF_v3_ontology.json it is not usable as one:

    depth 2 -> 4 real structures (root / Basic cell groups and regions /
               fiber tracts / ventricular systems) + 20 annotation ids that
               are not in the ontology at all and pass straight through
    depth 3 -> 35 labels
    depth 4 -> 59 labels, already down to `abducens nerve`
    depth 6 -> 184 labels
    depth 8 -> 90 labels under `Cerebral cortex` ALONE, of which ~20 are
               over 1 mm^3 and the tail ends at `Perirhinal area, layer 6b`
               at 0.00 mm^3

Every one of those labels becomes its own `multivariate_extras` term in
ants.registration (see Registration_ants/pipeline.py's
_build_guide_regions_from_labels), and a guide region under ~1 mm^3 hurts
rather than helps -- the hand-drawn boundary error is a large fraction of
the structure, and a systematic over/under-shoot actively drags the
deformation the wrong way. So a partition is NOT a depth. It is a
hand-curated set of groups, refined one node at a time:

    label 1  Cerebral cortex (688)
       + expand -> label 8 Cortical plate (695), label 9 Cortical subplate (703)
         + expand 8 -> Isocortex (315), Olfactory areas (698),
                       Hippocampal formation (1089), ...

Two ways to refine, and the GUI drives the second one: expand() takes a
group one level down (every child at once), while split_out() gives ONE
node picked in an ontology tree its own label, at whatever depth it sits.
Expanding to reach `Field CA1` means four levels of children nobody asked
for; picking it splits exactly it. Both leave the parent as the residual,
so everything below applies unchanged either way.

NESTING IS THE POINT, not an edge case. Expanding keeps the parent group
alive as a *residual*: its roots are unchanged, and the deepest-matching
rule below sends every voxel that a child now covers to that child, leaving
the parent holding only what nothing finer claims. Two consequences fall
straight out of that and are why this is worth its own module:

  1. children that are too small to be their own guide term are simply not
     given a group, and are therefore still covered by the residual parent
     -- no separate "rest of X" bookkeeping.
  2. the atlas side needs the mirror-image subtraction, because
     `mask.guide_regions.atlas_ids` expands each id to ALL its descendants:
     label 1's atlas outline would otherwise re-include everything labels
     8/9 were split out to guide separately, and the same atlas voxels get
     pulled towards two different sample outlines. `atlas_exclude_ids`
     computes exactly that subtraction from the nesting, instead of the
     user maintaining it by hand -- which is what configs/s12t.yaml does
     today (`atlas_exclude_ids: {1: [507, 151]}`, because CCFv3 files the
     olfactory bulb under Cerebral cortex while label 6 guides it
     separately) and which silently goes stale the moment the painting
     changes.

GUI-free and dependency-light (numpy + the ontology dict), so:

    python shared/label_partition.py --selftest
"""
import argparse
import json

import numpy as np

MAX_LABEL = 255           # brush volumes are uint8, as in paint_mask.py

# A guide region smaller than this is not worth its own multivariate_extras
# term -- see the module docstring. Used to decide which children an expand
# actually splits out; the rest stay with the residual parent.
DEFAULT_MIN_MM3 = 1.0


class Group:
    """One brush label and the ontology roots it stands for.

    `ids` is a tuple because one brush label routinely needs several
    unrelated structures (configs/s12t.yaml's label 5 is Cerebellum +
    Hindbrain + cerebellum related fiber tracts), and because DevCCF has no
    single "cortex" node at all.
    """

    def __init__(self, label, ids, name, parent=None):
        self.label = int(label)
        self.ids = tuple(int(i) for i in ids)
        self.name = name
        self.parent = parent          # the Group this was split out of, or None

    def __repr__(self):
        return f"Group(label={self.label}, ids={list(self.ids)}, name={self.name!r})"


class Partition:
    """An ordered collection of Groups, keyed by brush label."""

    def __init__(self, groups=()):
        self.groups = {g.label: g for g in groups}

    # ---- construction ------------------------------------------------

    @classmethod
    def from_region_ids(cls, region_ids, structures):
        """{brush label: [ontology id, ...]} -> Partition. This is the
        `region_ids` block of a .regions.json sidecar, i.e. the format
        paint_mask.py already writes and Registration_ants already reads."""
        groups = []
        for label in sorted(region_ids, key=int):
            ids = [int(i) for i in region_ids[label]]
            groups.append(Group(int(label), ids, group_name(ids, structures)))
        return cls(groups)

    @classmethod
    def from_regions_json(cls, path, structures):
        meta = json.loads(open(path, encoding="utf-8").read())
        region_ids = meta.get("region_ids") or {}
        if not region_ids:
            raise ValueError(
                f"{path} has no region_ids block -- a partition can only be seeded from a "
                f"sidecar whose labels were assigned in the ontology tree (names alone are "
                f"matched as substrings and cannot be resolved back to ids reliably).")
        return cls.from_region_ids(region_ids, structures)

    # ---- queries -----------------------------------------------------

    def __len__(self):
        return len(self.groups)

    def __iter__(self):
        return iter(self.ordered())

    def ordered(self):
        return [self.groups[lab] for lab in sorted(self.groups)]

    def region_ids(self):
        return {g.label: list(g.ids) for g in self.ordered()}

    def region_names(self, structures):
        return {g.label: [structures[i]["name"] for i in g.ids if i in structures]
                for g in self.ordered()}

    def next_free_label(self):
        used = set(self.groups)
        for candidate in range(1, MAX_LABEL + 1):
            if candidate not in used:
                return candidate
        raise ValueError(f"no brush label left: a partition can hold at most {MAX_LABEL} groups")

    def root_to_label(self):
        """{ontology id: brush label} for the group ROOTS only (not their
        descendants) -- the input to the deepest-match rule."""
        return {i: g.label for g in self.ordered() for i in g.ids}

    # ---- the deepest-match rule --------------------------------------

    def label_of_id(self, structure_id, structures, roots=None):
        """Which brush label a single ontology id belongs to: the group whose
        root is the DEEPEST ancestor-or-self of it. 0 when nothing claims it
        (including ids the ontology doesn't describe at all -- the DeMBA P5
        annotation has 20 of those).

        Deepest rather than first-match is what makes nesting work: with
        label 1 = Cerebral cortex(688) and label 8 = Cortical plate(695),
        an Isocortex voxel (path ... 688, 695, 315) has both as ancestors
        and must go to 8, leaving 1 holding only what 8 does not cover.
        """
        roots = self.root_to_label() if roots is None else roots
        info = structures.get(int(structure_id))
        if info is None:
            return 0
        for ancestor in reversed(info["structure_id_path"]):
            if ancestor in roots:
                return roots[ancestor]
        return 0

    def collapse(self, label_arr, structures):
        """A label volume in ontology-id space -> a uint8 volume in brush
        space, via label_of_id.

        Mapped through np.unique/searchsorted rather than a dense lookup
        table indexed by id: CCFv3 ids reach 6.1e8 (e.g. 614454272 in the
        DeMBA P5 annotation), so `np.arange(max_id + 1)` would allocate
        ~2.4 GB to describe the few hundred ids actually present.
        """
        roots = self.root_to_label()
        uniq = np.unique(label_arr)
        mapped = np.array([self.label_of_id(int(u), structures, roots) for u in uniq],
                          dtype=np.uint8)
        return mapped[np.searchsorted(uniq, label_arr)]

    # ---- refinement --------------------------------------------------

    def expandable(self, label, structures, node_voxels, voxel_mm3, min_mm3=DEFAULT_MIN_MM3):
        """What expanding `label` one ontology level would give.

        Returns (kept, skipped): both are [(id, name, mm3), ...] sorted big
        first. `kept` become their own groups; `skipped` are the children
        under min_mm3, which stay with the residual parent (see the module
        docstring) rather than becoming guide terms that would hurt.

        Children with no voxels in this annotation at all are in neither
        list: Registration_ants refuses the whole run when a guide region
        matches nothing, so they must never reach a group.
        """
        group = self.groups[int(label)]
        taken = set(self.root_to_label())
        kept, skipped = [], []
        for child in direct_children(group.ids, structures):
            if child in taken:                    # already split out earlier
                continue
            voxels = node_voxels.get(child, 0)
            if not voxels:
                continue
            mm3 = voxels * voxel_mm3
            row = (child, structures[child]["name"], mm3)
            (kept if mm3 >= min_mm3 else skipped).append(row)
        kept.sort(key=lambda r: -r[2])
        skipped.sort(key=lambda r: -r[2])
        return kept, skipped

    def expand(self, label, structures, node_voxels, voxel_mm3, min_mm3=DEFAULT_MIN_MM3):
        """Split `label` one ontology level down, in place. The parent group
        stays as the residual. Returns (kept, skipped) as `expandable` does;
        `kept` are now groups of their own.
        """
        group = self.groups[int(label)]
        kept, skipped = self.expandable(label, structures, node_voxels, voxel_mm3, min_mm3)
        if not kept:
            return kept, skipped
        if len(self.groups) + len(kept) > MAX_LABEL:
            raise ValueError(
                f"expanding label {label} would need {len(kept)} more brush labels, but only "
                f"{MAX_LABEL - len(self.groups)} are left (the export is uint8)")
        for child, name, _mm3 in kept:
            new_label = self.next_free_label()
            self.groups[new_label] = Group(new_label, [child], name, parent=group)
        return kept, skipped

    def owner_of(self, structure_id, structures):
        """The Group whose voxels currently include `structure_id`, or None
        when nothing in the partition covers it."""
        return self.groups.get(self.label_of_id(structure_id, structures))

    def split_out(self, structure_id, structures):
        """Give ONE ontology node, picked at any depth, its own brush label.

        The tree-picker counterpart of expand(), and the reason the GUI no
        longer offers "expand one level": the level you want is almost never
        the next one down. CCFv3 puts `Field CA1` five levels under
        `Cerebral cortex`, so reaching it by expanding means four rounds and
        ~40 groups nobody asked for, each of which then has to be merged
        back one at a time. Picking the node splits exactly it.

        Nothing else about the partition changes: the deepest-match rule
        (label_of_id) takes the node's voxels out of whichever group held
        them and leaves that group as the residual, exactly as expand()
        does -- so atlas_exclude_ids and merge_back keep working unchanged.

        A node NOT covered by any group is allowed too, and is how a region
        the seed partition never mentioned (the registration collapsed it to
        background) gets painted at all: the new group simply has no parent.

        Returns the new Group. Raises ValueError if the node is unknown to
        the ontology or is already some group's root -- both are "your click
        did nothing" cases the panel has to be able to say out loud.
        """
        sid = int(structure_id)
        if sid not in structures:
            raise ValueError(f"id {sid} is not in this ontology")
        roots = self.root_to_label()
        if sid in roots:
            raise ValueError(f"{structures[sid]['name']} is already brush label {roots[sid]}")
        parent = self.groups.get(self.label_of_id(sid, structures, roots))
        new_label = self.next_free_label()
        group = Group(new_label, [sid], structures[sid]["name"], parent=parent)
        self.groups[new_label] = group

        # Re-parent the groups that are now nested INSIDE the new one. Split
        # Isocortex first and Cortical plate second and the ontology nesting
        # is plate -> iso while the parent links would still say both were
        # split out of Cerebral cortex -- merge_back(cortex) would then drop
        # iso without dropping what it was split out of. The links have to
        # follow the ids, not the order they were clicked in.
        for other in self.ordered():
            if other.label == new_label or other.parent is not parent:
                continue
            if all(sid in structures.get(int(i), {}).get("structure_id_path", [])[:-1]
                   for i in other.ids):
                other.parent = group
        return group

    def drop(self, label):
        """Remove one group and everything split out of it. Returns the
        removed groups.

        merge_back() is the same operation on a group's CHILDREN; this one
        includes the group itself, which is what undoes a split_out. The
        voxels fall back to whatever still claims them -- the parent group,
        or background when the group had no parent.
        """
        label = int(label)
        if label not in self.groups:
            return []
        removed = self.merge_back(label)
        removed.append(self.groups.pop(label))
        return removed

    def children_of(self, label):
        """The groups split out of `label` by a previous expand()."""
        parent = self.groups.get(int(label))
        return [g for g in self.ordered() if g.parent is parent and parent is not None]

    def merge_back(self, label):
        """Undo one expand(): drop every group split out of `label`, and
        recursively whatever was split out of those. The parent group is
        already present (it never left), so its voxels simply stop being
        claimed by anything finer. Returns the removed groups.
        """
        removed = []
        for child in self.children_of(label):
            removed += self.merge_back(child.label)
            removed.append(self.groups.pop(child.label))
        return removed

    # ---- the atlas side ----------------------------------------------

    def atlas_exclude_ids(self, structures):
        """{brush label: [ontology id, ...]} to subtract from that label's
        atlas-side outline -- the roots of every OTHER group that is a
        strict descendant of one of this group's roots.

        This is mask.guide_regions.atlas_exclude_ids in the pipeline config.
        Without it a residual parent's atlas outline silently re-includes
        every child that was split out of it, and the same atlas voxels are
        pulled towards two different sample outlines at once.
        """
        roots = {g.label: set(g.ids) for g in self.ordered()}
        out = {}
        for group in self.ordered():
            exclude = set()
            for other in self.ordered():
                if other.label == group.label:
                    continue
                for oid in other.ids:
                    if oid in roots[group.label]:
                        continue
                    path = structures.get(oid, {}).get("structure_id_path", [])
                    if roots[group.label] & set(path[:-1]):
                        exclude.add(oid)
            if exclude:
                out[group.label] = sorted(_maximal(exclude, structures))
        return out

    def empty_atlas_side(self, structures, own_voxels):
        """Labels whose atlas outline is empty once atlas_exclude_ids is
        applied -- a fully-expanded residual parent is the normal way to get
        one (expand Cerebral cortex all the way and label 1 keeps no atlas
        voxels of its own, because CCFv3 labels no voxel with 688 itself).

        Worth its own query because the failure is otherwise late and
        confusing: _build_guide_regions_from_labels raises and aborts the
        whole registration run when a configured label matches no atlas
        voxel, so a residual label that still carries paint from before the
        expand has to be caught at export time instead.

        own_voxels is {id: voxels labelled EXACTLY that id} -- i.e.
        dict(zip(*np.unique(annotation, return_counts=True))), NOT
        atlas_reference.voxels_per_ontology_node's subtree totals. Subtree
        totals credit every voxel to all of its ancestors, so a parent whose
        children have all been split out would still report their voxels and
        never look empty.
        """
        exclude = self.atlas_exclude_ids(structures)
        out = []
        for group in self.ordered():
            kept = set()
            for root in group.ids:
                kept |= subtree_ids(root, structures)
            for dropped in exclude.get(group.label, []):
                kept -= subtree_ids(dropped, structures)
            if not sum(own_voxels.get(sid, 0) for sid in kept):
                out.append(group.label)
        return out

    def summary(self, structures, node_voxels, voxel_mm3):
        """One line per group for the GUI panel and the export log."""
        lines = []
        for g in self.ordered():
            mm3 = sum(node_voxels.get(i, 0) for i in g.ids) * voxel_mm3
            kids = self.children_of(g.label)
            note = f"  (residual; {len(kids)} split out)" if kids else ""
            lines.append(f"  label {g.label:>3} = {g.name}  [{', '.join(str(i) for i in g.ids)}]"
                         f"  ~{mm3:.1f} mm3{note}")
        return "\n".join(lines)


def _maximal(ids, structures):
    """Drop every id that is already a descendant of another id in the set.

    atlas_exclude_ids expands each entry to its whole subtree (same rule as
    atlas_ids), so listing both Cortical plate and Isocortex subtracts
    exactly what Cortical plate alone would. Keeping only the maximal ones
    is purely for the human reading the emitted config block -- a residual
    Cerebral cortex otherwise excludes eight ids where three say the same
    thing.
    """
    ids = {int(i) for i in ids}
    return {i for i in ids
            if not (ids & set(structures.get(i, {}).get("structure_id_path", [])[:-1]))}


def subtree_ids(root, structures):
    """`root` plus every descendant of it -- the id set atlas_ids/
    atlas_exclude_ids each expand to, and the one a picked tree node has to
    light up (a node at depth 3 owns no voxels under its own id)."""
    root = int(root)
    return {sid for sid, info in structures.items() if root in info["structure_id_path"]}


def direct_children(root_ids, structures):
    """Ontology ids whose direct parent is one of `root_ids`, deduplicated.

    Direct parent is structure_id_path[-2] (the path is the full root->node
    chain ending in the node itself), so this is one level down regardless
    of how deep the roots themselves sit -- a group can hold roots at
    different depths (Cerebellum is depth 4, cerebellum related fiber tracts
    is not) and expanding it still means "one level down from each".
    """
    roots = {int(i) for i in root_ids}
    return sorted({
        sid for sid, info in structures.items()
        if len(info["structure_id_path"]) >= 2 and info["structure_id_path"][-2] in roots
    })


def group_name(ids, structures):
    names = [structures[i]["name"] for i in ids if i in structures]
    if not names:
        return "unknown"
    return names[0] if len(names) == 1 else f"{names[0]} +{len(names) - 1}"


# =====================================================================================
# selftests -- synthetic ontology, no files, no GUI
# =====================================================================================
def _fake_ontology():
    """root(1) -> cortex(10) -> {plate(100) -> {iso(1000), hpf(1001)}, subplate(101)}
                 -> cerebellum(20) -> {cbx(200)}
    Depths are what structure_id_path encodes; ids are readable on purpose."""
    paths = {
        1: [1],
        10: [1, 10], 100: [1, 10, 100], 101: [1, 10, 101],
        1000: [1, 10, 100, 1000], 1001: [1, 10, 100, 1001],
        20: [1, 20], 200: [1, 20, 200],
    }
    names = {1: "root", 10: "cortex", 100: "plate", 101: "subplate",
             1000: "iso", 1001: "hpf", 20: "cerebellum", 200: "cbx"}
    return {sid: {"name": names[sid], "structure_id_path": path} for sid, path in paths.items()}


def selftest_deepest_match_wins():
    print("1. nested groups: the deepest root claims the voxel...")
    s = _fake_ontology()
    p = Partition.from_region_ids({1: [10], 2: [20]}, s)
    assert p.label_of_id(1000, s) == 1, "iso should fall to the cortex group"

    p.groups[3] = Group(3, [100], "plate")
    assert p.label_of_id(1000, s) == 3, "once plate is its own group it must win over cortex"
    assert p.label_of_id(101, s) == 1, "subplate is not under plate, so it stays with cortex"
    assert p.label_of_id(200, s) == 2
    assert p.label_of_id(999999, s) == 0, "an id the ontology doesn't describe belongs to nobody"
    print("   ok")


def selftest_collapse_volume():
    print("2. collapse() maps an id volume into brush space...")
    s = _fake_ontology()
    p = Partition.from_region_ids({1: [10], 2: [20]}, s)
    # 614454272 stands in for the real DeMBA ids that are absent from CCFv3:
    # it must map to 0 and must not make collapse allocate a 2.4 GB table.
    arr = np.array([[1000, 1001, 200, 614454272, 0]], dtype=np.uint32)
    out = p.collapse(arr, s)
    assert out.dtype == np.uint8
    assert out.tolist() == [[1, 1, 2, 0, 0]], out.tolist()

    p.groups[3] = Group(3, [100], "plate", parent=p.groups[1])
    assert p.collapse(arr, s).tolist() == [[3, 3, 2, 0, 0]]
    print("   ok")


def selftest_expand_skips_small_and_absent():
    print("3. expand() splits big children, leaves small ones with the parent...")
    s = _fake_ontology()
    p = Partition.from_region_ids({1: [10]}, s)
    # plate is big, subplate is under the 1 mm^3 floor, and 20/200 are not
    # children of 10 at all.
    node_voxels = {100: 5000, 101: 100, 1000: 4000, 1001: 1000}
    kept, skipped = p.expand(1, s, node_voxels, voxel_mm3=0.001, min_mm3=1.0)

    assert [k[0] for k in kept] == [100], kept
    assert [k[0] for k in skipped] == [101], skipped
    assert len(p) == 2, "one child split out, parent kept as the residual"

    # The skipped child is not lost -- the residual parent still claims it.
    assert p.label_of_id(101, s) == 1
    assert p.label_of_id(1000, s) == p.groups[2].label

    # A child with no voxels in this annotation must never become a group:
    # the pipeline refuses the whole run when a guide region matches nothing.
    p2 = Partition.from_region_ids({1: [10]}, s)
    kept2, skipped2 = p2.expand(1, s, {100: 5000}, voxel_mm3=0.001, min_mm3=1.0)
    assert [k[0] for k in kept2] == [100] and skipped2 == [], (kept2, skipped2)
    print("   ok")


def selftest_merge_back_is_recursive():
    print("4. merge_back() undoes an expand, including deeper ones...")
    s = _fake_ontology()
    p = Partition.from_region_ids({1: [10]}, s)
    node_voxels = {100: 5000, 101: 5000, 1000: 4000, 1001: 1000}
    p.expand(1, s, node_voxels, voxel_mm3=0.001)
    plate = next(g for g in p if 100 in g.ids)
    p.expand(plate.label, s, node_voxels, voxel_mm3=0.001)
    assert len(p) == 5, p.ordered()          # cortex + plate + subplate + iso + hpf

    removed = p.merge_back(1)
    assert len(p) == 1 and set(p.groups) == {1}, p.ordered()
    assert len(removed) == 4, removed
    assert p.label_of_id(1000, s) == 1, "everything falls back to the residual parent"
    print("   ok")


def selftest_split_out_picks_any_depth():
    print("5. split_out() gives any node its own label, at any depth...")
    s = _fake_ontology()
    p = Partition.from_region_ids({1: [10]}, s)

    # Two levels down in one click -- what expand() would need two rounds
    # (and a group for subplate nobody asked for) to reach.
    iso = p.split_out(1000, s)
    assert iso.parent is p.groups[1], iso.parent
    assert len(p) == 2 and p.label_of_id(1000, s) == iso.label
    assert p.label_of_id(1001, s) == 1, "the rest of the subtree stays with the residual"

    # Splitting an ANCESTOR of an existing group afterwards has to re-parent
    # it, or merge_back(1) would drop iso without dropping plate.
    plate = p.split_out(100, s)
    assert iso.parent is plate, "iso sits under plate, whatever order they were clicked in"
    assert p.label_of_id(1001, s) == plate.label
    assert len(p.merge_back(1)) == 2 and len(p) == 1

    # A node no group covers is allowed: that is how a region the seed
    # partition never mentioned becomes paintable at all.
    orphan = p.split_out(200, s)
    assert orphan.parent is None and p.label_of_id(200, s) == orphan.label

    # ...and drop() undoes exactly that, where merge_back only takes children.
    assert [g.label for g in p.drop(orphan.label)] == [orphan.label]
    assert p.label_of_id(200, s) == 0

    for bad in (999999, 10):
        try:
            p.split_out(bad, s)
        except ValueError:
            pass
        else:
            raise AssertionError(f"split_out({bad}) must refuse, not add a duplicate group")
    print("   ok")


def selftest_atlas_exclude_ids():
    print("6. atlas_exclude_ids() mirrors the nesting onto the atlas side...")
    s = _fake_ontology()
    p = Partition.from_region_ids({1: [10], 2: [20]}, s)
    assert p.atlas_exclude_ids(s) == {}, "no nesting yet, nothing to subtract"

    node_voxels = {10: 10000, 100: 5000, 101: 5000, 1000: 4000, 1001: 1000, 20: 1, 200: 1}
    p.expand(1, s, node_voxels, voxel_mm3=0.001)
    exclude = p.atlas_exclude_ids(s)
    assert exclude == {1: [100, 101]}, exclude

    # Expanding plate too must NOT lengthen label 1's list: iso/hpf are
    # already inside the subtree that excluding plate removes.
    plate = next(g for g in p if 100 in g.ids)
    p.expand(plate.label, s, node_voxels, voxel_mm3=0.001)
    exclude = p.atlas_exclude_ids(s)
    assert exclude[1] == [100, 101], exclude
    assert exclude[plate.label] == [1000, 1001], exclude

    # cortex and plate are now fully covered by their children, so guiding
    # either would ask the pipeline for an empty atlas outline. Only leaves
    # carry voxels of their own, which is the real annotations' shape too.
    own = {1000: 4000, 1001: 1000, 101: 5000, 200: 1}
    assert p.empty_atlas_side(s, own) == [1, plate.label], p.empty_atlas_side(s, own)
    assert 9 not in p.empty_atlas_side(s, own)

    # This is the shape of the rule configs/s12t.yaml maintains by hand
    # today: label 6 (olfactory bulb) sits under label 1 (Cerebral cortex).
    p2 = Partition.from_region_ids({1: [688], 6: [507, 151]}, {
        688: {"name": "Cerebral cortex", "structure_id_path": [997, 8, 567, 688]},
        507: {"name": "MOB", "structure_id_path": [997, 8, 567, 688, 695, 698, 507]},
        151: {"name": "AOB", "structure_id_path": [997, 8, 567, 688, 695, 698, 151]},
    })
    assert p2.atlas_exclude_ids({
        688: {"name": "Cerebral cortex", "structure_id_path": [997, 8, 567, 688]},
        507: {"name": "MOB", "structure_id_path": [997, 8, 567, 688, 695, 698, 507]},
        151: {"name": "AOB", "structure_id_path": [997, 8, 567, 688, 695, 698, 151]},
    }) == {1: [151, 507]}
    print("   ok")


def selftest_label_budget():
    print("7. the uint8 brush-label budget is enforced, not silently wrapped...")
    s = _fake_ontology()
    p = Partition.from_region_ids({1: [10]}, s)
    for lab in range(2, MAX_LABEL + 1):
        p.groups[lab] = Group(lab, [20], "filler")
    try:
        p.next_free_label()
    except ValueError:
        pass
    else:
        raise AssertionError("next_free_label must refuse past 255")

    try:
        p.expand(1, s, {100: 5000, 101: 5000}, voxel_mm3=0.001)
    except ValueError as exc:
        assert "uint8" in str(exc), exc
    else:
        raise AssertionError("expand must refuse when the labels would not fit")
    print("   ok")


def run_selftests():
    selftest_deepest_match_wins()
    selftest_collapse_volume()
    selftest_expand_skips_small_and_absent()
    selftest_merge_back_is_recursive()
    selftest_split_out_picks_any_depth()
    selftest_atlas_exclude_ids()
    selftest_label_budget()
    print("\nall label_partition selftests passed")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Brush-label <-> ontology-region partitions")
    parser.add_argument("--selftest", action="store_true", help="run the synthetic tests and exit")
    if not parser.parse_args().selftest:
        parser.error("nothing to do without --selftest (this module is a library)")
    return run_selftests()


if __name__ == "__main__":
    raise SystemExit(main())
