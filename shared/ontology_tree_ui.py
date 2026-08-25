"""Shared Qt widgets for browsing an ontology tree.

Used by both paint_mask.py's region-assignment panel and tools/atlas_view.py's
region-selection panel -- the same searchable QTreeWidget, just wired to a
different action once a node is picked (assign it to a brush label, vs.
highlight it in the atlas). `scrollable` is a second, more general widget
both panels also need for the same reason: see its own docstring, and
`shrinkable`/`set_dock_width` for the width half of the same problem.

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
