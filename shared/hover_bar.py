"""The wide "region under the cursor" bar along the bottom of a window.

Shared by tools/atlas_view.py and paint_mask.py's `mode: labels`, for the
same reason shared/ontology_tree_ui.py is shared: it is one widget with a
fair amount of behaviour in it (fold the chain to fit, scale the type to the
dock's height, paint the strip in the region's own colour) and two copies
would drift.

WHY A BAR AND NOT A SIDE PANEL. What an annotation stores for one voxel is a
leaf like "layer 5 of primary motor area" -- true, and rarely the level you
are actually looking for, so the root -> leaf chain is what has to be
readable. Shown as an indented block in a dock on the right it costs a
column of window width permanently and sits as far from the cursor as it can
get. One line along the bottom instead: the DEEPEST few levels, big enough
to read without looking away from the picture, folded from the SHALLOW end
(a leading ellipsis) when the window is too narrow, and painted in the
label's own colour so the strip and the voxel under the cursor are visibly
the same region.

The line-fitting and colour maths are pure functions with no Qt in them, so:

    python shared/hover_bar.py --selftest
"""
import argparse
from types import SimpleNamespace

# Starting height in px (draggable afterwards, like any dock), the type size
# that height corresponds to, and how many ontology levels the bar will show
# at most before it starts folding the shallow end away. Dragging the bar
# taller or shorter scales the type with it, between the two limits -- a
# one-line bar has nothing else to do with the height, and "make the text
# bigger" is the reason to want a taller one.
HEIGHT = 64
MIN_HEIGHT = 26
FONT_PT = 15
FONT_PT_RANGE = (8, 36)
MAX_LEVELS = 6
PADDING_PX = 14
SEPARATOR = "  ›  "
ELLIPSIS = "…"

QLabel = QFont = QFontMetrics = QSize = Qt = None


def _import_qt():
    """PyQt5, imported lazily -- same pattern as ontology_tree_ui, so
    --selftest runs the fitting maths with no PyQt5 and no display."""
    global QLabel, QFont, QFontMetrics, QSize, Qt
    from PyQt5.QtCore import QSize as _QSize, Qt as _Qt
    from PyQt5.QtGui import QFont as _QFont, QFontMetrics as _QFontMetrics
    from PyQt5.QtWidgets import QLabel as _QLabel
    QLabel, QFont, QFontMetrics, QSize, Qt = _QLabel, _QFont, _QFontMetrics, _QSize, _Qt


def ancestry_labels(structures, structure_id):
    """The root -> voxel chain of `structure_id`, one short label per level.

    The same chain atlas_reference.format_ancestry renders as an indented
    block, flattened for a one-line bar: no indent, no markers, just the
    names (with acronyms, which are what stays recognisable when a name is
    long) in ontology order, shallowest first.
    """
    info = structures.get(structure_id)
    if info is None:
        return [f"id {structure_id} (not in the ontology)"]
    labels = []
    for sid in info["structure_id_path"]:
        node = structures.get(sid)
        if node is None:
            labels.append(f"[{sid}] ?")
            continue
        acronym = node.get("acronym")
        labels.append(f"{node['name']} ({acronym})" if acronym else node["name"])
    return labels


def render_ancestry_line(labels, folded):
    """`labels` joined into one bar line, with a leading ellipsis iff levels
    above them were folded away."""
    line = SEPARATOR.join(labels)
    return f"{ELLIPSIS}{SEPARATOR}{line}" if folded else line


def fit_ancestry_line(labels, fits, max_levels=MAX_LEVELS):
    """The DEEPEST levels of `labels` (root -> leaf) that `fits` accepts, as
    one line.

    The bar is one row across the window, so which levels get dropped when
    the chain is too long has to be decided rather than left to clipping. The
    deep end is the useful end -- the voxel's own region is the level the
    annotation actually stores, and the levels just above it are what say
    which structure it belongs to -- so the chain is kept from the leaf
    backwards and folded away from the SHALLOW (root) end, behind a leading
    ellipsis. The voxel's own region always survives, even when it alone is
    too wide to fit.

    `fits` is a predicate on the rendered string, which keeps the decision
    testable without a window: the GUI passes "QFontMetrics says this is
    narrower than the bar", the selftest passes a character count.
    """
    labels = [str(label) for label in labels if str(label)]
    if not labels:
        return ""
    kept = labels[-max_levels:] if max_levels else list(labels)
    folded = len(labels) - len(kept)
    line = render_ancestry_line(kept, folded)
    while len(kept) > 1 and not fits(line):
        kept, folded = kept[1:], folded + 1
        line = render_ancestry_line(kept, folded)
    return line


def font_pt(height, reference=(HEIGHT, FONT_PT), limits=FONT_PT_RANGE):
    """Type size in points for a bar `height` px tall.

    Straight proportion off the bar's default height/size pair, clamped: the
    bar holds ONE line, so a taller bar is only worth anything if the line
    grows with it, and a shorter one has to give the line back or it clips.
    The clamp keeps a dock dragged to either extreme legible rather than
    microscopic or absurd.
    """
    ref_height, ref_pt = reference
    lo, hi = limits
    scaled = int(round(ref_pt * float(height) / float(ref_height)))
    return max(lo, min(hi, scaled))


def colours(rgba):
    """(background, foreground) CSS colours for a bar painted in one label's
    own atlas colour.

    The background is exactly the RGB napari draws that label with, so the
    bar and the voxel under the cursor are visibly the same region. The text
    colour is then forced to whichever of near-black/white the background can
    actually carry -- the annotation colormap spans everything from dark
    navy to pale yellow, and one fixed text colour is unreadable on half of
    it. Fully transparent (i.e. background label 0) gets a neutral strip.
    """
    rgba = [float(v) for v in rgba]
    if len(rgba) > 3 and rgba[3] <= 0.0:
        return "#20232a", "#c8ccd4"
    r, g, b = (min(max(v, 0.0), 1.0) for v in rgba[:3])
    background = "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in (r, g, b))
    # Rec. 709 luminance: how bright the colour actually looks, not its mean.
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return background, ("#101010" if luminance > 0.5 else "#ffffff")


def add_hover_bar(viewer, structures, colour_of, below=None,
                  name="Region under cursor", resting=None):
    """Add the bar to `viewer`'s BOTTOM dock area and return the handle that
    drives it.

    colour_of(structure_id) -> RGBA: asked for on every change rather than
    given a colormap, so the bar can take its colour straight off whichever
    layer is drawing the region and cannot drift out of step with the
    picture.

    `below` is another bottom dock, if there is one: the bar is split beneath
    it so it spans the whole width at the very bottom rather than being
    parked beside it (Qt fills a dock area left to right).

    Both the bar's width and its HEIGHT are the user's to drag, like every
    other dock: the width decides how many levels fit and the height decides
    the type size (font_pt).

    Returns SimpleNamespace(label=, show=, dock=), where `show(structure_id,
    extra=())` is the whole of what a mouse callback does -- which is what
    makes the behaviour testable without synthesising Qt mouse events. Pass
    None for "the cursor is not over the volume" and 0 for "background";
    `extra` is appended to the chain as further (deepest, always kept)
    segments, for whatever the calling tool knows and the ontology does not.
    """
    _import_qt()
    resting = resting or "Hover over the image to read the region under the cursor."

    class HoverBar(QLabel):
        """A QLabel that re-fits itself whenever it is resized: how many
        levels fit is a function of the bar's width (so narrowing the window
        folds levels away rather than clipping them) and the type size is a
        function of its height (so dragging the dock taller enlarges the
        line instead of padding it with empty colour).

        It also REMEMBERS the height it was last given, and reports it as its
        size hint. QMainWindow re-divides a dock area on every relayout --
        every window resize, every other dock that opens -- handing each dock
        what its contents ask for, so a one-line label whose hint is one line
        tall gets squashed straight back to the minimum and the height the
        user dragged the splitter to is lost. Hinting the current height
        makes that redistribution a no-op instead.
        """

        def __init__(self, text):
            super().__init__(text)
            self._height = HEIGHT

        def sizeHint(self):
            return QSize(super().sizeHint().width(), self._height)

        def resizeEvent(self, event):
            super().resizeEvent(event)
            if self.height() >= MIN_HEIGHT:
                self._height = self.height()
            render()

    bar = HoverBar(resting)
    bar.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    bar.setTextInteractionFlags(Qt.TextSelectableByMouse)
    # No word wrap and no minimum width: the line is fitted to the bar by
    # hand (see render), and a wrapping label would drive the window's
    # minimum height instead -- see ontology_tree_ui.scrollable.
    bar.setWordWrap(False)
    bar.setMinimumWidth(1)
    # A floor, and no ceiling: the splitter above the bar is then draggable
    # in both directions, which is the whole point of scaling the type with
    # the height. The floor is explicit rather than left to the label's own
    # minimumSizeHint so that neither the text nor the stylesheet padding can
    # push the dock's minimum height back up.
    bar.setMinimumHeight(MIN_HEIGHT)

    def bar_font():
        """The font the line is drawn in: this bar's height, in points.

        Built by hand rather than read back off the widget because
        `bar.font()` is not the last word here -- napari sets a font-size on
        every QLabel from its APPLICATION stylesheet, which in Qt beats
        setFont() outright. The size therefore has to be written into the
        bar's OWN stylesheet (a widget stylesheet beats the application's)
        by paint(), and measured from a QFont assembled to match.
        """
        font = QFont(bar.font())
        font.setPointSize(font_pt(bar.height()))
        font.setBold(True)
        return font

    def paint(background, foreground, point_size):
        style = (f"background: {background}; color: {foreground}; "
                 f"padding: 0px {PADDING_PX}px; "
                 f"font-size: {point_size}pt; font-weight: bold;")
        # Re-applying an identical stylesheet still re-polishes the widget,
        # which can resize it, which lands back here: only write on a change.
        if style != bar.styleSheet():
            bar.setStyleSheet(style)

    # mouse_move fires continuously; re-rendering the same chain on every
    # pixel of travel is pure waste, and the flicker is visible.
    last = {"id": None, "extra": ()}

    def render():
        sid, extra = last["id"], list(last["extra"])
        font = bar_font()
        if not sid:
            paint(*colours([0.0, 0.0, 0.0, 0.0]), font.pointSize())
            bar.setText(resting if sid is None else "(background -- no region)")
            return
        metrics = QFontMetrics(font)
        room = max(bar.width() - 2 * PADDING_PX - 4, 40)
        bar.setText(fit_ancestry_line(
            ancestry_labels(structures, int(sid)) + extra,
            lambda text: metrics.horizontalAdvance(text) <= room))
        paint(*colours(colour_of(int(sid))), font.pointSize())

    def show(structure_id, extra=()):
        structure_id = None if structure_id is None else int(structure_id)
        extra = tuple(extra)
        if (structure_id, extra) == (last["id"], last["extra"]):
            return
        last["id"], last["extra"] = structure_id, extra
        render()

    render()
    dock = viewer.window.add_dock_widget(bar, area="bottom", name=name)
    qt_window = viewer.window._qt_window
    if below is not None:
        # addDockWidget would put the bar BESIDE the other dock (Qt fills a
        # dock area left to right); splitting it off vertically is what makes
        # it a full-width strip under everything.
        qt_window.splitDockWidget(below, dock, Qt.Vertical)
    # A starting height only: the bar is one line, but it is a draggable
    # dock like the rest, and the line's type size follows whatever height
    # it is dragged to.
    qt_window.resizeDocks([dock], [HEIGHT], Qt.Vertical)
    return SimpleNamespace(label=bar, show=show, dock=dock)


# =====================================================================================
# selftests -- the fitting/colour maths, no Qt, no window
# =====================================================================================
def selftest_ancestry_line():
    """Asserts only, no printing: tools/atlas_view.py runs this as one of its
    own numbered selftests, and prints its own heading around it."""
    labels = ["root", "grey matter", "cerebrum", "cortex", "motor area", "layer 5"]

    # Room for everything: no ellipsis, ontology order preserved.
    line = fit_ancestry_line(labels, lambda _t: True)
    assert ELLIPSIS not in line, line
    assert line.split(SEPARATOR) == labels, line

    # Narrower and narrower bars drop shallow levels first, and the voxel's
    # own region survives every one of them.
    seen = set()
    for room in (200, 60, 40, 25, 10, 1):
        line = fit_ancestry_line(labels, lambda text: len(text) <= room)
        assert line.endswith("layer 5"), (room, line)
        parts = [p for p in line.split(SEPARATOR) if p != ELLIPSIS]
        # Whatever survives is a contiguous DEEP tail of the chain.
        assert labels[-len(parts):] == parts, (room, line)
        assert (ELLIPSIS in line) == (len(parts) < len(labels)), (room, line)
        if len(line) > room:
            # The only line allowed to overflow is the last level on its own.
            assert parts == labels[-1:], (room, line)
        seen.add(len(parts))
    assert len(seen) > 1, seen              # the bar really does fold, not just clip

    # max_levels caps the chain even when the bar is infinitely wide.
    capped = fit_ancestry_line(labels, lambda _t: True, max_levels=2)
    assert capped == render_ancestry_line(labels[-2:], 4), capped
    assert fit_ancestry_line([], lambda _t: True) == ""

    # Whatever the caller knows and the ontology does not rides at the deep
    # end, i.e. it is the one segment that can never be folded away.
    with_extra = fit_ancestry_line(labels + ["repainted"], lambda text: len(text) <= 30)
    assert with_extra.endswith("repainted"), with_extra

    # The strip is painted the label's own colour, with text that survives it.
    for rgba, expected_text in (([0.05, 0.05, 0.2, 1.0], "#ffffff"),
                                ([0.9, 0.95, 0.4, 1.0], "#101010")):
        background, foreground = colours(rgba)
        assert background == "#%02x%02x%02x" % tuple(
            int(round(v * 255)) for v in rgba[:3]), background
        assert foreground == expected_text, (rgba, foreground)
    assert colours([0.0, 0.0, 0.0, 0.0])[0] == "#20232a"

    # Height is a size the user drags, not a constant: the type follows it,
    # monotonically, and stays legible at both extremes.
    lo, hi = FONT_PT_RANGE
    assert font_pt(HEIGHT) == FONT_PT
    sizes = [font_pt(h) for h in (10, 30, 64, 120, 400)]
    assert sizes == sorted(sizes), sizes
    assert sizes[0] == lo and sizes[-1] == hi, sizes
    assert all(lo <= pt <= hi for pt in sizes), sizes


def selftest_ancestry_labels():
    """The chain itself: ontology order, acronyms kept, unknown ids survive."""
    structures = {
        1: {"name": "root", "acronym": "root", "structure_id_path": [1]},
        10: {"name": "cortex", "acronym": "CTX", "structure_id_path": [1, 10]},
        100: {"name": "no acronym here", "structure_id_path": [1, 10, 100]},
    }
    assert ancestry_labels(structures, 100) == [
        "root (root)", "cortex (CTX)", "no acronym here"]
    assert ancestry_labels(structures, 999) == ["id 999 (not in the ontology)"]


def run_selftests():
    print("1. hover bar: keeps the deepest levels, folds the shallow end away")
    selftest_ancestry_line()
    print("   ok")
    print("2. hover bar: the chain is the ontology path, acronyms and all")
    selftest_ancestry_labels()
    print("   ok")
    print("\nall hover_bar selftests passed")
    return 0


def main():
    parser = argparse.ArgumentParser(description="The 'region under cursor' bottom bar")
    parser.add_argument("--selftest", action="store_true", help="run the tests and exit")
    if not parser.parse_args().selftest:
        parser.error("nothing to do without --selftest (this module is a library)")
    return run_selftests()


if __name__ == "__main__":
    raise SystemExit(main())
