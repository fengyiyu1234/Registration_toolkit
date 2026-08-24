"""Translate a painted mask's `.regions.json` region_ids from one atlas ontology
to the other (DevCCF <-> Allen CCFv3), using the DevCCF paper's published
voxel-overlap crosswalk (Supplementary Data 3 of 41467_2024_53254).

WHY this is not a dictionary lookup
-----------------------------------
The crosswalk is a table of *label-level* voxel overlaps: for each pair
(DevCCF label, CCFv3 label) it gives how many voxels the two share, measured at
P56 by putting CCFv3 into DevCCF-P56 morphology. It only covers ids that
actually appear as voxel labels -- 288 of DevCCF's 2552 ontology entries, 669 of
CCFv3's 1327.

The regions a guide mask is painted against are almost never those leaves. They
are coarse ontology NODES ("pallium", "hindbrain", "diencephalon"), which carry
no voxels of their own. So converting one is three steps, not one:

  1. expand the source id down to every descendant that the crosswalk knows,
  2. sum their overlap voxels per target label -> a mass distribution over the
     target ontology's leaves,
  3. aggregate that mass UP the target tree and pick the smallest set of target
     nodes that covers it.

Step 3 is where the judgement is, and it is deliberately two-sided:

  share  (recall)    = this node's subtree mass / the painted region's total mass
                       "how much of what I painted this node accounts for"
  purity (precision) = this node's subtree mass FROM THIS REGION / that node's
                       subtree mass from ALL source labels
                       "how much of this node is actually the thing I painted"

A node is accepted when purity >= --purity, otherwise the walk recurses into its
children; nodes contributing less than --min-share of the region are dropped.
Requiring purity is what stops the walk from handing back a node far larger than
what was painted -- and that failure is not hypothetical. DevCCF "midbrain" puts
96.7% of its mass inside CCFv3 `MB`, but only 52% of CCFv3 `MB` comes back from
it: the prosomeric model files the pretectum under diencephalon and the isthmus
under hindbrain, both of which CCFv3 calls midbrain, so `MB` is roughly twice the
volume of what "midbrain" meant while painting. Pairing a guide region against a
structure 2x its size is worse than having no guide at all (see
qc_guide_mask.py's "systematic volume mismatch"), so the tool reports such a node
as a COARSE ALTERNATIVE for a human to accept or reject, and never picks it
silently.

WHAT IT DOES NOT DO
-------------------
Nothing is re-registered and no voxels are touched: the hand-painted outlines in
the .nii.gz stay exactly as drawn. Only the atlas-side pairing (`region_ids`,
and the `regions` names that go with them) is rewritten, which is all that has
to change when the same painted sample is registered against a different atlas.

Two honest caveats about the crosswalk itself:
  * it was computed at P56, between two ADULT parcellations. Applied to a P04/P05
    mask it is a nomenclature crosswalk, not a developmental one.
  * DevCCF (prosomeric) and CCFv3 (columnar) genuinely draw different boundaries.
    Where they do, no id mapping can fix it -- the region has to be repainted.
    The coverage number printed per label is the honest measure of that: it is
    the fraction of the painted region's mass the chosen ids account for.

Usage (antsreg env; no GUI, runs headless):

    python convert_regions_ontology.py ../Registration_ants/atlas/mask/s12t_guide7.regions.json
    python convert_regions_ontology.py <sidecar> -o s12t_guide7_ccfv3.regions.json
    python convert_regions_ontology.py <sidecar> --in-place        # backs up first
    python convert_regions_ontology.py <sidecar> --purity 0.8 --min-share 0.02

Prints a per-label report plus a ready-to-paste `mask.guide_regions.atlas_ids`
snippet; with -o/--in-place also writes a converted sidecar that paint_mask.py
can resume from (it reads region_ids/regions/annotated_slices, all of which are
carried over).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from registration_ants import atlas_utils

# Paths of the three reference files, relative to the pipeline repo next door.
# Defaults only -- every one of them is overridable on the command line.
_ANTS = Path(__file__).resolve().parent.parent / "Registration_ants"
DEFAULT_CROSSWALK = _ANTS / "atlas" / "DevCCF" / "41467_2024_53254_MOESM4_ESM.xlsx"
DEFAULT_DEVCCF_ONTOLOGY = _ANTS / "atlas" / "DevCCF" / "DevCCFv1_ontology.json"
DEFAULT_CCFV3_ONTOLOGY = _ANTS / "atlas" / "DeMBA" / "CCF_v3_ontology.json"

CROSSWALK_SHEET = "SupplementaryData3"
CROSSWALK_HEADER_ROW = 3          # three title/blurb rows above the real header
DEVCCF_COL = "DevCCF Label ID"
CCFV3_COL = "CCFv3 Label ID"
OVERLAP_COL = "Overlapping Voxels"

# A "coarse alternative" is only reported when it both covers most of the painted
# region and stays within a few times its volume. Without the second bound every
# region gets `root` and `grey` listed, which is true and useless.
COARSE_MIN_SHARE = 0.5
COARSE_MAX_BLOAT = 3.0

DIRECTIONS = {
    # name: (source column, target column, source ontology default, target ontology default)
    "devccf2ccfv3": (DEVCCF_COL, CCFV3_COL, DEFAULT_DEVCCF_ONTOLOGY, DEFAULT_CCFV3_ONTOLOGY),
    "ccfv32devccf": (CCFV3_COL, DEVCCF_COL, DEFAULT_CCFV3_ONTOLOGY, DEFAULT_DEVCCF_ONTOLOGY),
}


def load_crosswalk(path):
    """Supplementary Data 3 as a 3-column frame (source id, target id, voxels).

    Kept as the raw pairs rather than collapsed to a majority-vote dict: the
    many-to-many structure IS the information here, and collapsing it early
    (what scripts/relabel_labels_to_devccf.py does, correctly, for its own
    per-voxel job) would throw away the mass distribution the tree walk needs.
    """
    df = pd.read_excel(path, sheet_name=CROSSWALK_SHEET, header=CROSSWALK_HEADER_ROW)
    missing = [c for c in (DEVCCF_COL, CCFV3_COL, OVERLAP_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} sheet {CROSSWALK_SHEET!r} has no column(s) {missing} -- "
                         f"found {list(df.columns)}. Wrong sheet, or the paper's file changed shape.")
    df = df[[DEVCCF_COL, CCFV3_COL, OVERLAP_COL]].dropna()
    df[DEVCCF_COL] = df[DEVCCF_COL].astype(int)
    df[CCFV3_COL] = df[CCFV3_COL].astype(int)
    df[OVERLAP_COL] = df[OVERLAP_COL].astype("int64")
    return df


def subtree_mass(mass_by_label, ontology):
    """{leaf id: voxels} -> {node id: total voxels in that node's subtree}.

    Walks each label's own structure_id_path (its ancestor chain, root first)
    instead of recursing down the tree, so it is one pass over the labels that
    actually carry mass rather than one pass over the whole 1300-2500 node
    ontology per query. Labels absent from the ontology keep their own mass and
    contribute to nothing else -- they are reported separately, not dropped
    silently.
    """
    totals = {}
    for label_id, mass in mass_by_label.items():
        entry = ontology.get(label_id)
        chain = entry["structure_id_path"] if entry else [label_id]
        for ancestor in chain:
            totals[ancestor] = totals.get(ancestor, 0) + mass
    return totals


def children_of(ontology):
    """{parent id: [child id, ...]} built once, so the tree walk below is not
    quadratic in the ontology size (it visits a node per accepted region, and
    scanning all 2552 entries for each one's children adds up)."""
    kids = {}
    for sid, entry in ontology.items():
        path = entry["structure_id_path"]
        if len(path) > 1:
            kids.setdefault(path[-2], []).append(sid)
    return kids


def descendants_of(ontology, root_ids):
    """Every id whose ancestor chain contains any of root_ids (roots included).
    Same "descendants included" rule the pipeline's own id matching uses, so what
    this expands is exactly what the registration would have matched."""
    roots = set(root_ids)
    return {sid for sid, entry in ontology.items() if roots & set(entry["structure_id_path"])}


class Crosswalk:
    """One direction of the DevCCF<->CCFv3 mapping, with both ontologies loaded."""

    def __init__(self, table, source_col, target_col, source_ontology, target_ontology):
        self.table = table
        self.source_col = source_col
        self.target_col = target_col
        self.source = source_ontology
        self.target = target_ontology
        self.children = children_of(target_ontology)
        self.roots = [sid for sid, e in target_ontology.items() if len(e["structure_id_path"]) == 1]
        # Denominator for purity: every target node's subtree mass summed over
        # the WHOLE crosswalk, i.e. that structure's size in the overlap volume
        # regardless of which source region it came from.
        all_mass = table.groupby(target_col)[OVERLAP_COL].sum().to_dict()
        self.total_mass = subtree_mass(all_mass, target_ontology)

    def region_mass(self, source_ids):
        """Mass distribution over target labels for one painted region."""
        expanded = descendants_of(self.source, source_ids)
        sub = self.table[self.table[self.source_col].isin(expanded)]
        by_target = sub.groupby(self.target_col)[OVERLAP_COL].sum().to_dict()
        return {int(k): int(v) for k, v in by_target.items()}, expanded

    def convert(self, source_ids, purity, min_share):
        """-> (picked, coarse, grand_total). picked/coarse are lists of
        (target_id, mass, share, purity), picked being the minimal covering set
        and coarse the high-recall/low-purity ancestors the walk refused (see
        the module docstring for why those are surfaced rather than chosen)."""
        mass_by_target, _ = self.region_mass(source_ids)
        grand = sum(mass_by_target.values())
        if not grand:
            return [], [], 0
        mine = subtree_mass(mass_by_target, self.target)

        picked, refused = [], []

        def visit(node_id):
            node_mass = mine.get(node_id, 0)
            if node_mass < min_share * grand:
                return
            node_total = self.total_mass.get(node_id, 0)
            node_purity = node_mass / node_total if node_total else 0.0
            if node_total and node_purity >= purity:
                picked.append((node_id, node_mass, node_mass / grand, node_purity))
                return
            # High recall but impure: this node is the natural coarse answer and
            # is what a human would otherwise pick by hand, so record it before
            # descending past it. Bounded on BOTH sides -- every region is
            # trivially covered by `root` and `grey`, and a 20x-too-big node is
            # not an alternative anyone would weigh, so only nodes within
            # COARSE_MAX_BLOAT of the painted volume are worth a line.
            if node_mass / grand >= COARSE_MIN_SHARE and node_purity >= 1 / COARSE_MAX_BLOAT:
                refused.append((node_id, node_mass, node_mass / grand, node_purity))
            kids = self.children.get(node_id)
            if not kids:
                picked.append((node_id, node_mass, node_mass / grand, node_purity))
                return
            for kid in kids:
                visit(kid)

        for root in self.roots:
            visit(root)
        picked.sort(key=lambda row: -row[1])
        refused.sort(key=lambda row: -row[1])
        return picked, refused, grand


def detect_direction(region_ids, devccf_ontology, ccfv3_ontology):
    """Which way round the sidecar's ids are, decided by which ontology contains
    them. Ambiguity is an error rather than a coin flip -- picking the wrong
    direction produces a plausible-looking but entirely wrong mapping."""
    ids = {i for entries in region_ids.values() for i in entries}
    if not ids:
        raise ValueError("sidecar has no region_ids to convert")
    in_dev = len(ids & set(devccf_ontology))
    in_ccf = len(ids & set(ccfv3_ontology))
    if in_dev and not in_ccf:
        return "devccf2ccfv3"
    if in_ccf and not in_dev:
        return "ccfv32devccf"
    raise ValueError(
        f"cannot tell which ontology these ids come from ({in_dev}/{len(ids)} are DevCCF, "
        f"{in_ccf}/{len(ids)} are CCFv3) -- pass --direction explicitly")


def format_report(names, result, crosswalk):
    out = []
    for label in sorted(result):
        picked, refused, grand, source_ids = result[label]
        src_names = ", ".join(crosswalk.source[i]["name"] if i in crosswalk.source else f"<{i}?>"
                              for i in source_ids)
        coverage = sum(row[1] for row in picked) / grand if grand else 0.0
        flag = "  <-- LOW, see caveats" if coverage < 0.9 else ""
        out.append(f"\nlabel {label}: {src_names}")
        out.append(f"  painted as: {names.get(label, [])}")
        out.append(f"  crosswalk mass {grand:,} voxels, covered {coverage:.1%}{flag}")
        if not grand:
            out.append("  !! no crosswalk entry for this region or any of its descendants")
            continue
        for tid, mass, share, pur in picked:
            e = crosswalk.target[tid]
            out.append(f"    {tid:7d}  {e['acronym']:<14s} {e['name'][:44]:<44s} "
                       f"share={share:6.1%}  purity={pur:6.1%}")
        for tid, mass, share, pur in refused:
            e = crosswalk.target[tid]
            out.append(f"    [coarse alternative] {tid:7d} {e['acronym']:<10s} {e['name'][:34]:<34s} "
                       f"share={share:6.1%} purity={pur:6.1%} -- covers the region but is "
                       f"~{1 / pur:.1f}x its volume")
    return "\n".join(out)


def yaml_snippet(result):
    lines = ["    atlas_ids:"]
    for label in sorted(result):
        picked = result[label][0]
        ids = [tid for tid, *_ in picked]
        acr = ", ".join(f"{tid}" for tid in ids)
        lines.append(f"      {label}: [{acr}]")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert a painted mask's .regions.json region_ids between the DevCCF and "
                    "CCFv3 ontologies via the DevCCF paper's voxel-overlap crosswalk.")
    ap.add_argument("sidecar", type=Path, help="path to <mask stem>.regions.json")
    ap.add_argument("--direction", choices=["auto", *DIRECTIONS], default="auto",
                    help="default auto: decided by which ontology the sidecar's ids are in")
    ap.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK,
                    help=f"DevCCF Supplementary Data 3 xlsx (default {DEFAULT_CROSSWALK})")
    ap.add_argument("--source-ontology", type=Path, default=None,
                    help="override the source ontology json (default: per --direction)")
    ap.add_argument("--target-ontology", type=Path, default=None,
                    help="override the target ontology json (default: per --direction)")
    ap.add_argument("--purity", type=float, default=0.9,
                    help="accept a target node once this fraction of it comes from the painted "
                         "region; below it the walk descends into its children (default 0.9)")
    ap.add_argument("--min-share", type=float, default=0.03,
                    help="drop target nodes holding less than this fraction of the painted "
                         "region's mass (default 0.03 -- below that a node is a sliver of a "
                         "neighbouring structure the two atlases disagree about, not part of "
                         "what was painted)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="write the converted sidecar here")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the input sidecar (a .bak copy is kept)")
    args = ap.parse_args(argv)

    meta = json.loads(args.sidecar.read_text())
    region_ids = {int(k): [int(i) for i in v] for k, v in (meta.get("region_ids") or {}).items()}
    names = {int(k): list(v) for k, v in (meta.get("regions") or {}).items()}
    if not region_ids:
        ap.error(f"{args.sidecar} has no region_ids -- nothing to convert (was the mask painted "
                 "without picking regions in paint_mask.py's ontology tree?)")

    if args.direction == "auto":
        # Detection needs both ontologies as themselves, so it always uses the
        # defaults -- an override cannot be assigned to a side before the side
        # is known, which is the very thing being detected.
        direction = detect_direction(
            region_ids,
            atlas_utils.load_ccf_ontology_json(DEFAULT_DEVCCF_ONTOLOGY),
            atlas_utils.load_ccf_ontology_json(DEFAULT_CCFV3_ONTOLOGY))
    else:
        direction = args.direction
    source_col, target_col, src_default, tgt_default = DIRECTIONS[direction]
    source_ont = atlas_utils.load_ccf_ontology_json(args.source_ontology or src_default)
    target_ont = atlas_utils.load_ccf_ontology_json(args.target_ontology or tgt_default)

    unknown = sorted({i for ids in region_ids.values() for i in ids} - set(source_ont))
    if unknown:
        ap.error(f"region_ids {unknown} are not in the source ontology "
                 f"({args.source_ontology or src_default}) -- wrong --direction, or the sidecar "
                 "was written against a different ontology file")

    crosswalk = Crosswalk(load_crosswalk(args.crosswalk), source_col, target_col,
                          source_ont, target_ont)

    print(f"direction: {direction}   purity>={args.purity}   min-share>={args.min_share}")
    print(f"crosswalk: {args.crosswalk.name} "
          f"({crosswalk.table[source_col].nunique()} source labels -> "
          f"{crosswalk.table[target_col].nunique()} target labels)")

    result = {}
    for label, ids in region_ids.items():
        picked, refused, grand = crosswalk.convert(ids, args.purity, args.min_share)
        result[label] = (picked, refused, grand, ids)

    print(format_report(names, result, crosswalk))
    print("\n--- paste into mask.guide_regions (Registration_ants sample config) ---")
    print(yaml_snippet(result))

    empty = [lab for lab, (picked, *_rest) in result.items() if not picked]
    if empty:
        print(f"\nWARNING: label(s) {empty} got no target ids at all -- their region is absent "
              "from the crosswalk. Repaint them against the new atlas, or pick ids by hand.")

    if args.in_place and args.output:
        ap.error("--in-place and -o are mutually exclusive")
    out_path = args.sidecar if args.in_place else args.output
    if out_path is None:
        print("\n(no sidecar written -- pass -o PATH or --in-place)")
        return 0

    converted = dict(meta)
    converted["region_ids"] = {str(lab): [tid for tid, *_ in result[lab][0]] for lab in sorted(result)}
    converted["regions"] = {str(lab): [target_ont[tid]["name"] for tid, *_ in result[lab][0]]
                            for lab in sorted(result)}
    # Provenance: what the ids USED to be, and exactly how they were translated.
    # Without it the next reader cannot tell a converted sidecar from one that
    # was painted against the new atlas in the first place.
    converted["converted_from"] = {
        "direction": direction,
        "crosswalk": str(args.crosswalk),
        "source_ontology": str(args.source_ontology or src_default),
        "target_ontology": str(args.target_ontology or tgt_default),
        "purity": args.purity,
        "min_share": args.min_share,
        "region_ids": {str(lab): list(ids) for lab, ids in sorted(region_ids.items())},
        "regions": {str(lab): list(v) for lab, v in sorted(names.items())},
        "coverage": {str(lab): (sum(r[1] for r in result[lab][0]) / result[lab][2]
                                if result[lab][2] else 0.0) for lab in sorted(result)},
        "note": "Voxel-overlap crosswalk (DevCCF paper Supplementary Data 3) measured at P56 "
                "between two adult parcellations; the painted outlines themselves are unchanged.",
    }
    if args.in_place:
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        shutil.copy2(out_path, backup)
        print(f"\nbacked up {out_path.name} -> {backup.name}")
    out_path.write_text(json.dumps(converted, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
