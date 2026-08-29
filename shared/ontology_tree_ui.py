"""Shared Qt widgets for browsing an ontology tree.

Used by both paint_mask.py's region-assignment panel and tools/atlas_view.py's
region-selection panel -- the same searchable QTreeWidget, just wired to a
different action once a node is picked (assign it to a brush label, vs.
highlight it in the atlas). `scrollable` is a second, more general widget
both panels also need for the same reason: see its own docstring, and
`shrinkable`/`set_dock_width` for the width half of the same problem.

The rest of the module is dock LAYOUT, shared by all three GUI tools for the
same reason the widgets are: `scroll_wrap_dock` /
`free_layer_controls_height` are the height half of what `shrinkable` does
for width (including for napari's OWN layer-controls dock, which otherwise
refuses to shrink past a dozen rows of controls), and `tabify` folds a
column of docks into one tab bar.

PyQt5 is imported lazily inside the functions below rather than at module
scope, the same pattern paint_mask.py's own _import_gui() uses -- both
callers already import PyQt5 that way before touching this module, so this
just avoids adding a THIRD place that would break if PyQt5 ever stopped
being a hard requirement of the GUI tools.
"""

Qt = QTreeWidgetItem = QScrollArea = None


def _import_qt():
    global Qt, QTreeWidgetItem, QScrollArea
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtWidgets import QTreeWidgetItem as _QTreeWidgetItem, QScrollArea as _QScrollArea
    Qt, QTreeWidgetItem, QScrollArea = _Qt, _QTreeWidgetItem, _QScrollArea


def shrinkable(widget):
    """Let a panel be dragged to any width, by dropping the minimum width Qt
    would otherwise compute from its contents.

    Without this a dock is effectively fixed-width, and it is never obvious
    why: Qt derives a widget's minimum width from its children's
    minimumSizeHint, and a QLabel's is the width of its longest line (or, for
    a word-wrapped one, its whole laid-out text). That propagates up
    dock content -> QDockWidget -> QMainWindow, so one long caption pins the
    panel and the splitter simply refuses to move past it. An explicit
    minimum of 1 overrides the hint; the caption then clips or scrolls
    instead of dictating the layout.

    Note this is only half the job when the panel ALSO carries an explicit
    setMaximumWidth -- that caps widening, which no minimum can undo. Use
    set_dock_width for a starting width instead of a maximum.

    tools/atlas_view.py has its own private copy of this (_shrinkable) from
    before it was shared; the two are the same one-liner.
    """
    _import_qt()
    widget.setMinimumWidth(1)
    return widget


def set_dock_width(dock_widget, width):
    """Give a dock a sensible STARTING width without pinning it there.

    The obvious way to make a panel start at 320 px is setMaximumWidth(320),
    and that is what makes it un-widenable forever after -- the complaint
    this function exists to remove. QMainWindow.resizeDocks sets the width
    once, as a layout request, leaving both edges draggable afterwards.

    Reaches the main window through the dock's parent rather than napari's
    private _qt_window, and does nothing if that is not a QMainWindow: a
    napari layout change should cost the starting width, not raise.
    """
    _import_qt()
    window = dock_widget.parent() if hasattr(dock_widget, "parent") else None
    if window is not None and hasattr(window, "resizeDocks"):
        window.resizeDocks([dock_widget], [width], Qt.Horizontal)
    return dock_widget


def tree_label(sid, info, voxels):
    acronym = info.get("acronym")
    name = f"{info['name']} ({acronym})" if acronym else info["name"]
    return [name, f"{voxels:,}" if voxels else "—", str(sid)]


def populate_ontology_tree(tree, structures, node_voxels):
    """Fill a QTreeWidget with the whole ontology; returns {sid: item}.

    Built shallowest-first so every node's parent (structure_id_path[-2])
    already exists when the node is added -- structure_id_path is the full
    root->node chain, so the tree shape comes straight out of it with no
    separate child lists to walk.

    Nodes with no voxels in this annotation are marked disabled rather than
    dropped: an empty node's whole subtree is necessarily empty too (a voxel
    is credited to every ancestor of its label, so a non-empty child forces a
    non-empty parent), which makes disabling safe -- it can never hide a
    pickable descendant -- and keeps the real tree shape visible instead of
    silently rewriting the ontology.
    """
    _import_qt()
    items = {}
    for sid, info in sorted(structures.items(), key=lambda kv: len(kv[1]["structure_id_path"])):
        path = info["structure_id_path"]
        voxels = node_voxels.get(sid, 0)
        item = QTreeWidgetItem(tree_label(sid, info, voxels))
        item.setData(0, Qt.UserRole, sid)
        if not voxels:
            item.setDisabled(True)
        parent = items.get(path[-2]) if len(path) >= 2 else None
        (parent or tree.invisibleRootItem()).addChild(item)
        items[sid] = item
    return items


def scrollable(label, min_height):
    """A growing text label wrapped in a scroll area, so its content length
    stops driving the whole window's minimum size.

    Load-bearing, not cosmetic. A word-wrapped QLabel reports its ENTIRE
    laid-out text as its minimumSizeHint, and Qt propagates that up the chain
    dock content -> QDockWidget -> the main QMainWindow. Measured on
    paint_mask.py's panel layout: filling the export label with a normal
    4-label report took the main window's minimum height from 854 px to
    1238 px, i.e. past the work area of a 1080p screen. Qt then resizes the
    window to satisfy the new minimum, and resizing a maximized/fullscreen
    window drops it back to normal -- that is the "clicking Export exits
    fullscreen" bug -- after which the window can never be made shorter
    again, so the z slider at the bottom sits under the taskbar for good.

    Widening/narrowing is no escape either, it is the wrong way round: a
    word-wrapped label has heightForWidth, so a NARROWER dock needs MORE
    height (measured on the same report: 648 px at 600 px wide, 1201 px at
    260 px wide). The one dimension still free makes it worse.

    A QScrollArea's own minimum is a small fixed size, so the report scrolls
    inside the panel instead of pushing the window around.
    """
    _import_qt()
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    scroll = QScrollArea()
    scroll.setWidget(label)
    scroll.setWidgetResizable(True)
    scroll.setMinimumHeight(min_height)
    return scroll


def scroll_wrap_dock(dock, min_height=48):
    """Put a dock's existing content inside a scroll area and drop the
    minimum height Qt derived from it, so the dock can be dragged short.

    The height twin of `shrinkable`, and needed for the same reason: Qt gives
    a dock the larger of its explicit minimum and its contents'
    minimumSizeHint, so a panel that asks for 400 px of rows is a hard floor
    on the splitter above it. `shrinkable` fixes that for width by setting an
    explicit minimum -- qSmartMinSize prefers an explicit minimum over the
    hint -- but doing only that for height would CLIP the rows that no longer
    fit, and a clipped slider is a control you cannot reach. So the content
    is re-parented into a QScrollArea first (whose own minimumSizeHint is a
    couple of text lines regardless of what it holds) and the rows that fall
    off the bottom scroll into view instead.

    Idempotent: wrapping an already-wrapped dock only re-applies the minimum.
    """
    _import_qt()
    from PyQt5.QtWidgets import QFrame, QScrollArea as _QScrollArea
    inner = dock.widget() if hasattr(dock, "widget") else None
    if inner is None:
        return dock
    if not isinstance(inner, _QScrollArea):
        scroll = _QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        # Reparent BEFORE dock.setWidget: QScrollArea.setWidget takes the
        # widget out of the dock's layout on its own, so the dock is never
        # left holding a layout item for a widget that has moved.
        scroll.setWidget(inner)
        inner.show()
        # napari paints its docks from the app stylesheet; an opaque viewport
        # here would draw a default-grey rectangle behind them.
        scroll.viewport().setAutoFillBackground(False)
        dock.setWidget(scroll)
        inner_scroll = scroll
    else:
        inner_scroll = inner
    inner_scroll.setMinimumHeight(min_height)
    dock.setMinimumHeight(min_height)
    return dock


def napari_layer_docks(viewer):
    """napari's own 'layer controls' and 'layer list' docks, as
    (controls, list) -- either is None if this napari does not have it
    where it used to.

    Reached through _qt_viewer, which is private, hence the getattr dance:
    every caller uses this for layout polish only, so a napari that moved
    them should cost the polish, not raise on startup.
    """
    qt_viewer = getattr(getattr(viewer, "window", None), "_qt_viewer", None)
    return tuple(getattr(qt_viewer, name, None)
                 for name in ("dockLayerControls", "dockLayerList"))


def free_layer_controls_height(viewer, min_height=48):
    """Let napari's own layer-controls/layer-list docks be dragged short.

    Without this the layer controls stop shrinking well before they are out
    of the way: the controls widget is a QStackedWidget whose minimumSizeHint
    is the tallest per-layer control form it has built (a Labels layer's is
    a dozen-plus rows), and napari stacks it above the layer list in the same
    column, so that floor is subtracted from every other panel sharing the
    left side -- with no way to trade the space back on a laptop screen.
    """
    docks = [dock for dock in napari_layer_docks(viewer) if dock is not None]
    for dock in docks:
        scroll_wrap_dock(dock, min_height)
    return docks


def tabify(viewer, docks, current=None):
    """Stack `docks` into ONE tabbed group, in the order given.

    napari's default is to stack docks down a column, which only works while
    there are two of them: paint_mask has four on the right and three on the
    left, and stacked they arrive as a pile of slivers that each have to be
    dragged open before they can be used. Tabbed, one panel gets the whole
    column and the rest are one click away.

    Chained pairwise (previous, dock) rather than (first, dock) because
    tabifyDockWidget inserts the second argument immediately after the first
    -- tabifying everything against docks[0] would reverse the tab order.
    """
    _import_qt()
    window = getattr(getattr(viewer, "window", None), "_qt_window", None)
    docks = [dock for dock in docks if dock is not None]
    if window is None or not hasattr(window, "tabifyDockWidget") or len(docks) < 2:
        # A side with one dock has no tab bar to build, but it still has to
        # end up SHOWN: a dock added to a window whose event loop has not run
        # yet is not visible until something shows it, which for every other
        # side is the raise_() below.
        for dock in docks:
            dock.show()
        return docks
    previous = docks[0]
    for dock in docks[1:]:
        window.tabifyDockWidget(previous, dock)
        previous = dock
    front = current if current in docks else docks[0]
    front.show()
    front.raise_()
    return docks
