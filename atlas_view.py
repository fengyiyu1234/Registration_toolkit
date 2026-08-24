"""Standalone tool: browse an atlas's annotation volume against its ontology.

A read-only reference viewer -- three synced ortho panes (grayscale template,
full annotation in colour, and whatever the ontology tree currently selects),
plus an ontology panel to pick one or several regions at once (their union
highlighted together -- see _add_region_panel) and a hover panel to read off
the full ancestor chain of whatever the mouse is over. Nothing here writes
anything or registers to anything; it exists purely so you can look at an
atlas and understand its ontology.

This used to be a second window paint_mask.py opened from its own ontology
picker, kept in sync with paint_mask's brush-label assignment. It is
standalone now: paint_mask.py's own ontology panel (for assigning tree nodes
to brush labels, feeding the guide-outline export) no longer opens or drives
any atlas display, and this viewer no longer knows paint_mask exists. Point
both at the same atlas_annotation_path / ontology_path if you want to browse
the atlas while painting -- just as two independent windows, not two views of
one state. The atlas loading + ontology math both tools share lives in
atlas_reference.py; this file is what turns that data into a window.

Usage (needs a display; runs in the antsreg conda env -- same napari+PyQt5+
SimpleITK requirement as paint_mask.py): edit configs/atlas_view.yaml
(gitignored -- copy it from configs/atlas_view.example.yaml the first time),
then just run the file -- no command-line arguments.

    conda activate antsreg
    python atlas_view.py
    python atlas_view.py configs/atlas_view.devccf.yaml   # or point at another config

The geometry/crosshair logic is separately runnable with no display and no
config, on purely synthetic data:

    python atlas_view.py --selftest
"""

import argparse
from types import SimpleNamespace

import numpy as np

import _local_config       # sibling module
import atlas_reference      # sibling module -- shared, GUI-free atlas loading
import ontology_tree_ui     # sibling module -- shared Qt ontology tree widget

# napari/PyQt5 are imported lazily by _import_gui(), for the same reason
# paint_mask.py does it: --selftest (pure numpy, no window) should run
# without a display or even PyQt5 installed.
napari = QLabel = QPushButton = QVBoxLayout = QWidget = None
QAbstractItemView = QCheckBox = QLineEdit = QSplitter = QTreeWidget = Qt = None
ViewerModel = QtViewer = None


def _import_gui():
    global napari, QLabel, QPushButton, QVBoxLayout, QWidget
    global QAbstractItemView, QCheckBox, QLineEdit, QSplitter, QTreeWidget, Qt
    global ViewerModel, QtViewer
    import napari as _napari
    from napari.components import ViewerModel as _ViewerModel
    from napari.qt import QtViewer as _QtViewer
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtWidgets import (QAbstractItemView as _QAbstractItemView,
                                 QCheckBox as _QCheckBox, QLabel as _QLabel,
                                 QLineEdit as _QLineEdit, QPushButton as _QPushButton,
                                 QSplitter as _QSplitter, QTreeWidget as _QTreeWidget,
                                 QVBoxLayout as _QVBoxLayout, QWidget as _QWidget)
    napari, QLabel, QPushButton = _napari, _QLabel, _QPushButton
    QVBoxLayout, QWidget = _QVBoxLayout, _QWidget
    QAbstractItemView = _QAbstractItemView
    QCheckBox, QLineEdit, QSplitter = _QCheckBox, _QLineEdit, _QSplitter
    QTreeWidget, Qt = _QTreeWidget, _Qt
    # ViewerModel + QtViewer are the "a napari canvas without its own window"
    # pair the extra ortho panes are built from.
    ViewerModel, QtViewer = _ViewerModel, _QtViewer


_LEGACY_CONFIG_PATHS = ()


def _load_local_config(cli_path=None):
    """configs/atlas_view.yaml -> the atlas SimpleNamespace atlas_reference
    functions expect. Unlike paint_mask.py, the atlas here is not optional --
    it is the whole tool -- so annotation_path/ontology_path are required by
    _local_config itself rather than left to silently produce an empty
    viewer."""
    cfg = _local_config.load_config(
        "atlas_view", cli_path=cli_path,
        required=("atlas_annotation_path", "ontology_path"),
        legacy_paths=_LEGACY_CONFIG_PATHS)
    return atlas_reference.atlas_reference_config(cfg)


# (dims.order, pane title). napari displays the LAST `ndisplay` axes of
# dims.order, so each entry names the axis kept on the slider, i.e. the axis
# the plane is perpendicular to.
_ORTHO_PANES = (
    ((0, 1, 2), "Axis 0 slice"),
    ((1, 0, 2), "Axis 1 slice"),
    ((2, 0, 1), "Axis 2 slice"),
)

# Layer attributes mirrored from the main pane onto the ortho panes. Only the
# main pane has a layer list, so these are the knobs a user can actually reach;
# everything else about the sub-panes is fixed at construction.
_MIRRORED_LAYER_ATTRS = ("visible", "opacity", "contrast_limits", "gamma", "colormap")

# Starting height in px of the bottom dock holding the two reconstructed
# views, leaving the main canvas the rest. Draggable afterwards.
_ORTHO_DOCK_HEIGHT = 300

# How far the mouse may travel between press and release and still count as a
# click rather than the start of a camera pan (see _click_moves_crosshair).
_CLICK_SLOP_PX = 4


def crosshair_vectors(centre, shape):
    """napari Vectors data (3, 2, 3): one full-extent line along each axis,
    all three crossing at voxel `centre`.

    Two of the three land in any given pane and the third does not, which is
    exactly the classic ortho crosshair and costs no per-pane bookkeeping. A
    pane slicing axis k draws the axes it displays: those lines start at 0 on
    the axes it slices, i.e. at `centre` on the slider axis, so they are in the
    current slice. The line ALONG axis k starts at 0 on axis k, so it only
    matches slice 0 and is otherwise (correctly) invisible -- it is the line
    pointing straight at the viewer.
    """
    ndim = len(shape)
    vectors = np.zeros((ndim, 2, ndim), dtype=float)
    for axis in range(ndim):
        start = list(centre)
        start[axis] = 0
        vectors[axis, 0] = start
        vectors[axis, 1, axis] = shape[axis]      # length: the whole extent
    return vectors


def _add_atlas_layers(model, atlas, scale_kwargs, features):
    """The atlas's four layers, added to one ViewerModel in draw order.

    Called once per ortho pane. Every pane gets its OWN Layer objects -- one
    Layer cannot belong to two ViewerModels -- but they are all backed by the
    same numpy arrays, so three panes cost three sets of slice textures, not
    three copies of a 287 MB annotation.
    """
    template = None
    if atlas.template is not None:
        template = model.add_image(atlas.template, name="reference (template grayscale)",
                                   colormap="gray", **scale_kwargs)
    # atlas.compact itself, not a copy: 190-odd labels over an 800^3 grid is
    # the biggest array in the process, and the layer is read-only anyway.
    annotation = model.add_labels(atlas.compact, name="annotation (all regions)",
                                  opacity=0.45, features=features, **scale_kwargs)
    annotation.editable = False
    highlight = model.add_labels(np.zeros(atlas.compact.shape, dtype=np.uint8),
                                 name="selection (selected regions)", opacity=0.85,
                                 colormap=napari.utils.DirectLabelColormap(
                                     color_dict={None: "transparent", 1: "red"}),
                                 **scale_kwargs)
    highlight.editable = False              # a reference; painting here would mean nothing
    centre = tuple(dim // 2 for dim in atlas.compact.shape)
    cross = model.add_vectors(crosshair_vectors(centre, atlas.compact.shape),
                              name="crosshair", edge_color="cyan", edge_width=1.5,
                              vector_style="line", opacity=0.9, **scale_kwargs)
    return SimpleNamespace(template=template, annotation=annotation,
                           highlight=highlight, cross=cross)


def _open_atlas_window(atlas, resolution_um, ortho=True):
    """The atlas window: the app's one and only napari window.

    THREE LAYERS, bottom to top: the grayscale template, the COMPLETE
    annotation in colour, and the region the ontology tree currently selects.
    All three are ordinary napari layers, so the layer list's own checkboxes
    and opacity slider are the controls -- annotation off to read the plain
    template, annotation on to see where the selection sits among its
    neighbours. The annotation starts semi-transparent and the selection is
    forced to solid red rather than taking a colormap colour, so it stays
    distinguishable on top of 190-odd annotation colours.

    THREE PANES (`ortho`), one per anatomical axis, sharing one slice position:
    a region's shape in the other two planes is often what tells you whether
    the tree selection is the structure you meant. napari has no ortho mode, so
    each pane is its own ViewerModel with its own dims.order, and
    `_sync_ortho_panes` keeps their sliders together; pane 0 is the real napari
    Viewer (it owns the window, the layer list and the layer controls) and the
    other two are bare QtViewer canvases in a dock beneath it.

    ONE CROSSHAIR over all of them: left-clicking any pane recentres all three
    on the clicked point, the way every other ortho viewer behaves. The
    crosshair is derived from dims.current_step rather than stored, so clicks,
    slider drags and "jump to region" all drive it through the same path and
    it cannot end up pointing at a plane no pane is showing.

    A HOVER PANEL (_add_ancestry_panel) on the right, because the one structure
    a voxel is labelled with is usually a leaf ("layer 5 of primary motor
    area") and the level you are actually picking is one of its ancestors. A
    separate REGION-SELECTION panel (_add_region_panel) on the left is what
    drives the tree click -> highlight path -- see that function for why it
    gets its own dedicated side rather than sharing one with the hover panel.

    The atlas grid here is INDEPENDENT of any sample grid -- nothing in this
    window is registered to anything, it is a reference only. What the atlas's
    own extent, orientation and downsampling affect is exactly this display,
    nothing else.
    """
    scale_kwargs = {"scale": [float(resolution_um) * atlas.downsample] * 3} if resolution_um else {}
    features = atlas_reference.annotation_features(atlas)
    panes = tuple(_ORTHO_PANES if ortho else _ORTHO_PANES[:1])

    viewer = napari.Viewer(title="Atlas viewer")
    main = SimpleNamespace(model=viewer, qt=None, order=panes[0][0],
                           layers=_add_atlas_layers(viewer, atlas, scale_kwargs, features))
    built = [main]

    for order, title in panes[1:]:
        model = ViewerModel(ndisplay=2)
        # Canvas BEFORE layers: napari 0.8.0's QtViewer.__init__ walks the
        # layers already in the model and reorders them against a visual map
        # it has not filled in yet, which raises KeyError on the second layer.
        # Adding to an empty model routes through the same code path one layer
        # at a time, where the map is always current.
        qt = QtViewer(model)
        built.append(SimpleNamespace(
            model=model, qt=qt, order=order,
            layers=_add_atlas_layers(model, atlas, scale_kwargs, features)))
        model.dims.order = order
        model.reset_view()

    if len(built) > 1:
        splitter = QSplitter(Qt.Horizontal)
        for pane, (_order, title) in zip(built[1:], panes[1:]):
            wrapper = QWidget()
            layout = QVBoxLayout(wrapper)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QLabel(title))
            layout.addWidget(pane.qt)
            # A bare QtViewer asks for 800x626 and Qt would honour both of
            # them: two side by side in a bottom dock open a ~1600px-wide
            # window whose main canvas is a letterbox. A small floor plus the
            # explicit resizeDocks below makes the ortho row a strip that the
            # user can then drag to whatever they actually want.
            pane.qt.setMinimumSize(160, 120)
            splitter.addWidget(wrapper)
        dock = viewer.window.add_dock_widget(splitter, area="bottom", name="Ortho views (synced)")
        viewer.window._qt_window.resizeDocks([dock], [_ORTHO_DOCK_HEIGHT], Qt.Vertical)
        _sync_ortho_panes(built)
        _mirror_layer_attrs(built)
        # QtViewer holds a vispy canvas and its GL context; napari only cleans
        # up the ones its own Window created, so these two have to be closed by
        # hand or the context outlives the window it was drawn in.
        viewer.window._qt_window.destroyed.connect(
            lambda *_a: [pane.qt.close() for pane in built[1:]])

    def set_highlight(mask, name=None):
        """The selection layer's data, on every pane at once.

        One array shared by all three layers rather than one copy each: the
        mask is a full-volume allocation and highlight_mask already made it.
        """
        for pane in built:
            pane.layers.highlight.data = mask
            if name:
                pane.layers.highlight.name = name

    def centre_on(index):
        """Point every pane at voxel `index` -- so the two reconstructions land
        on it too, not just the pane whose slider was moved or clicked."""
        for axis, value in enumerate(index):
            value = int(np.clip(round(float(value)), 0, atlas.compact.shape[axis] - 1))
            viewer.dims.set_point(axis, value * float(main.layers.annotation.scale[axis]))

    def refresh_cross(*_args):
        """The crosshair is DERIVED from the slice position, never stored
        separately: slider drag, click, and 'jump to region' then all reach it
        through one path, and none of them can leave it pointing somewhere the
        panes are not showing."""
        step = viewer.dims.current_step
        if len(step) != atlas.compact.ndim:
            # Closing the window empties the layer list, which collapses
            # dims.ndim to 2 and emits current_step on the way down. There is
            # nothing to draw a crosshair on any more.
            return
        data = crosshair_vectors(tuple(int(v) for v in step), atlas.compact.shape)
        for pane in built:
            pane.layers.cross.data = data

    # Connected to the MAIN pane only: _sync_ortho_panes pushes every pane's
    # change into it, so its event is the one place all of them converge.
    viewer.dims.events.current_step.connect(refresh_cross)
    for pane in built:
        pane.model.mouse_drag_callbacks.append(_click_moves_crosshair(pane, centre_on))
    refresh_cross()

    hover = _add_ancestry_panel(viewer, atlas, built)

    return SimpleNamespace(viewer=viewer, panes=built, highlight=main.layers.highlight,
                           annotation=main.layers.annotation, hover=hover,
                           set_highlight=set_highlight, centre_on=centre_on)


def _click_moves_crosshair(pane, centre_on):
    """Left-click anywhere in a pane -> crosshair there, in every pane.

    A generator callback, which is napari's way of telling a click from the
    start of a drag: napari resumes it on every mouse_move and once more on
    release, so the decision can be made after the fact. A plain press handler
    cannot -- the same press that begins a camera pan would jump the crosshair
    on its way, and the atlas would slide out from under the pan.
    """
    def callback(_model, event):
        if event.button != 1 or event.modifiers:
            return                          # right-click menus and modified drags stay napari's
        origin = np.asarray(event.pos, dtype=float)
        dragged = False
        yield
        while event.type == "mouse_move":
            if np.abs(np.asarray(event.pos, dtype=float) - origin).max() > _CLICK_SLOP_PX:
                dragged = True
            yield
        if not dragged:
            # event.position is world coords on ALL axes -- the two the pane
            # draws come from the cursor, the third from its own slider -- so
            # this is a full 3D point, not just the 2D one that was clicked.
            centre_on(pane.layers.annotation.world_to_data(event.position))
    return callback


def _sync_ortho_panes(panes):
    """Make every pane's slider position follow every other one's.

    All panes carry the same layers on the same grid and scale, so
    `dims.current_step` is one shared (i0, i1, i2) and can be copied across
    verbatim -- what differs between panes is only dims.order, i.e. which of
    the three a pane draws instead of slides.

    The `busy` flag is what stops the obvious infinite loop: assigning
    current_step on pane B emits B's own current_step event, which would push
    straight back at A.
    """
    busy = {"in": False}

    def follow(source):
        def _handler(*_args):
            if busy["in"]:
                return
            busy["in"] = True
            try:
                step = source.model.dims.current_step
                for pane in panes:
                    if pane is source or pane.model.dims.current_step == step:
                        continue
                    if pane.model.dims.ndim != len(step):
                        continue        # a pane already torn down by window close
                    pane.model.dims.current_step = step
            finally:
                busy["in"] = False
        return _handler

    for pane in panes:
        pane.model.dims.events.current_step.connect(follow(pane))


def _mirror_layer_attrs(panes):
    """Push the main pane's layer settings onto the ortho panes.

    Only the main pane has a layer list and layer controls, so a visibility
    checkbox or opacity slider there has to reach the other two canvases or
    they drift out of agreement with the window they live in. Attributes are
    connected by name and skipped where the layer type has no such event
    (an Image has contrast_limits, a Labels does not), so the same list covers
    the template, the annotation and the selection.
    """
    main, followers = panes[0], panes[1:]
    for role in ("template", "annotation", "highlight", "cross"):
        source = getattr(main.layers, role)
        if source is None:
            continue
        targets = [getattr(pane.layers, role) for pane in followers]
        for attr in _MIRRORED_LAYER_ATTRS:
            emitter = getattr(source.events, attr, None)
            if emitter is None or not all(hasattr(t, attr) for t in targets):
                continue

            def _push(*_args, _src=source, _targets=targets, _attr=attr):
                value = getattr(_src, _attr)
                for target in _targets:
                    if getattr(target, _attr) != value:
                        setattr(target, _attr, value)

            emitter.connect(_push)


def _add_ancestry_panel(viewer, atlas, panes):
    """A dock on the atlas window showing the full ontology chain of whatever
    the mouse is over, updated live from every pane.

    Returns SimpleNamespace(label=, show=) so the behaviour is testable without
    synthesising Qt mouse events: `show` takes a compact index and is the whole
    of what the mouse callbacks do.
    """
    label = QLabel("Hover over the atlas to see the full region hierarchy of that voxel.")
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.setWordWrap(False)

    # mouse_move fires continuously; re-rendering the same chain on every pixel
    # of travel is pure waste, and the flicker is visible.
    last = {"index": -1}

    def show(compact_index):
        if compact_index is None:
            compact_index = 0
        compact_index = int(compact_index)
        if compact_index == last["index"]:
            return
        last["index"] = compact_index
        if compact_index == 0:
            label.setText("(background -- no region)")
            return
        sid = int(atlas.present_ids[compact_index])
        voxels = atlas.node_voxels.get(sid, 0)
        label.setText(atlas_reference.format_ancestry(atlas.structures, sid)
                      + f"\n\nThis structure (with descendants) covers {voxels:,} voxels")

    def watcher(pane):
        def on_move(_model, event):
            dims_displayed = getattr(event, "dims_displayed", None)
            # get_value concatenates this onto a list internally, so it has to
            # BE a list -- napari's own canvas passes one, but dims.displayed
            # itself is a tuple and reaches here unchanged on other paths.
            if dims_displayed is not None:
                dims_displayed = list(dims_displayed)
            show(pane.layers.annotation.get_value(
                event.position,
                view_direction=getattr(event, "view_direction", None),
                dims_displayed=dims_displayed,
                world=True))
        return on_move

    for pane in panes:
        pane.model.mouse_move_callbacks.append(watcher(pane))

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(QLabel("Full hierarchy of the region under the cursor "
                            "(▶ = the level the annotation actually stores)"))
    layout.addWidget(ontology_tree_ui.scrollable(label, 220))
    viewer.window.add_dock_widget(dock, area="right", name="Region hierarchy")
    return SimpleNamespace(label=label, show=show)


def _add_region_panel(viewer, atlas, win):
    """The ontology tree, as its own dock: pick one or more nodes to
    highlight them (and every descendant) across all three panes at once.

    This is what used to live in paint_mask.py's ontology picker, driving
    THIS window from a dock on the sample's window instead. Now the atlas
    window owns its own region picker -- nothing here is aware of a brush
    label or an export, selecting nodes only ever highlights them.

    MULTI-SELECT: Ctrl/Shift-click (or drag) picks several nodes at once, and
    the highlight is their UNION -- highlight_mask() per node, OR'd together,
    so overlapping structures (e.g. a node and one of its own ancestors) only
    count their shared voxels once rather than fighting over a single-region
    colour. `tree.itemSelectionChanged` is what this is wired to rather than
    `currentItemChanged`: the latter fires on which item has focus, not on
    which ones are selected, so it misses exactly the ctrl-click case this
    feature exists for.

    A large, DEDICATED dock on the LEFT, opposite the hover panel on the
    right: the ontology sits 2-12 levels deep, so a tree squeezed into a
    fraction of a shared column leaves most of it scrolled out of view. Each
    side gets the window's full height instead.
    """
    search = QLineEdit()
    search.setPlaceholderText("Filter regions by name/acronym...")
    hide_empty = QCheckBox("Only regions with voxels in this annotation")
    hide_empty.setChecked(True)

    tree = QTreeWidget()
    tree.setHeaderLabels(["Region", "Voxels", "id"])
    # Ctrl/Shift-click (or a drag) adds to the selection instead of replacing
    # it -- the default SingleSelection would make multi-region highlighting
    # impossible no matter what on_select() does with it.
    tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
    # A floor, not a target: with the whole left side to itself (no other
    # dock sharing this column), the tree fills the rest of the window's
    # height regardless.
    tree.setMinimumHeight(400)
    items = ontology_tree_ui.populate_ontology_tree(tree, atlas.structures, atlas.node_voxels)
    # Content-driven, not a hardcoded pixel count -- the name column then fits
    # whatever's actually loaded instead of permanently claiming a fixed slice
    # of the window's width regardless of what's on screen. Still just the
    # starting point: Interactive is the header's default resize mode, so the
    # user can still drag it (and the dock) narrower or wider afterwards.
    tree.resizeColumnToContents(0)

    status = QLabel()
    jump_btn = QPushButton("Jump to selection centre (all three views)")

    def selected_ids():
        return [item.data(0, Qt.UserRole) for item in tree.selectedItems()]

    def refresh_filter():
        visible = atlas_reference.visible_tree_ids(
            atlas.structures, atlas.node_voxels, search.text(), hide_empty.isChecked())
        for sid, item in items.items():
            item.setHidden(sid not in visible)
        if search.text().strip():
            tree.expandAll()

    def on_select():
        sids = selected_ids()
        if not sids:
            return
        # OR, not sum: two selected nodes that share voxels (e.g. a node and
        # one of its own ancestors) must not double-highlight that overlap,
        # and highlight_mask's dtype is already the uint8 the layer wants.
        combined = np.zeros(atlas.compact.shape, dtype=np.uint8)
        names = []
        for sid in sids:
            names.append(atlas.structures[sid]["name"])
            combined |= atlas_reference.highlight_mask(atlas, sid)
        total_voxels = int(combined.sum())

        if len(sids) == 1:
            sid, info = sids[0], atlas.structures[sids[0]]
            voxels = atlas.node_voxels.get(sid, 0)
            status.setText(f"{info['name']} [{sid}]: {voxels:,} voxels including descendants."
                           if voxels else
                           f"{info['name']} [{sid}] has no voxels in this annotation.")
            layer_name = f"selection: {info['name']}"
        else:
            shown = ", ".join(names[:5]) + (", ..." if len(names) > 5 else "")
            status.setText(f"{len(sids)} regions selected ({shown}); their union highlights "
                           f"{total_voxels:,} voxels (overlap counted once).")
            layer_name = f"selection: {len(sids)} regions"
        win.set_highlight(combined, name=layer_name)

    def on_jump():
        if win.highlight.data.max() == 0:
            return
        centre = atlas_reference.mask_centre_index(win.highlight.data)
        if centre is not None:
            win.centre_on(centre)

    search.textChanged.connect(lambda _t: refresh_filter())
    hide_empty.toggled.connect(lambda _c: refresh_filter())
    tree.itemSelectionChanged.connect(on_select)
    jump_btn.clicked.connect(on_jump)

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(QLabel("Atlas ontology -- selecting a node highlights it and every "
                            "descendant in all three views; Ctrl/Shift-click to highlight "
                            "several regions at once."))
    layout.addWidget(search)
    layout.addWidget(hide_empty)
    layout.addWidget(tree)
    layout.addWidget(ontology_tree_ui.scrollable(status, 56))
    layout.addWidget(jump_btn)
    viewer.window.add_dock_widget(dock, area="left", name="Region selection")

    refresh_filter()


def _run_view(atlas_cfg):
    _import_gui()
    print("[atlas] loading atlas reference...")
    atlas = atlas_reference.load_atlas_reference(atlas_cfg, include_template=True)
    win = _open_atlas_window(atlas, atlas_cfg.resolution_um, ortho=atlas_cfg.ortho)
    _add_region_panel(win.viewer, atlas, win)


# =====================================================================================
# selftests -- synthetic arrays only, no GUI, no config, no atlas files on disk
# =====================================================================================
def selftest_ortho_panes_geometry():
    print("1. atlas ortho panes: every axis is sliced exactly once")
    # Every axis is the slider axis in exactly one pane, and is drawn in the
    # other two -- i.e. the three panes really are the three orthogonal views
    # and not two copies of one.
    orders = [order for order, _title in _ORTHO_PANES]
    assert len(orders) == 3, orders
    sliced = [order[0] for order in orders]
    assert sorted(sliced) == [0, 1, 2], sliced
    for order in orders:
        assert sorted(order) == [0, 1, 2], order
    print("   ok")


def selftest_crosshair_vectors():
    print("2. atlas crosshair: exactly two lines per pane, crossing at the current slice")
    shape = (6, 8, 10)
    vectors = crosshair_vectors((4, 5, 7), shape)
    assert vectors.shape == (3, 2, 3), vectors.shape

    # Each line starts at 0 on its own axis and at the crosshair on the others,
    # and is exactly as long as the volume, so it spans the whole pane.
    assert np.array_equal(vectors[:, 0], [[0, 5, 7], [4, 0, 7], [4, 5, 0]]), vectors[:, 0]
    assert np.array_equal(vectors[:, 1], np.diag(shape)), vectors[:, 1]

    # A pane slicing axis k shows a line iff that line's start sits in the
    # current slice -- true for the two lines lying in the plane, false for the
    # one pointing at the viewer. Two lines per pane is the whole trick.
    for sliced_axis in range(3):
        in_slice = [i for i in range(3) if vectors[i, 0, sliced_axis] == (4, 5, 7)[sliced_axis]]
        assert len(in_slice) == 2, (sliced_axis, in_slice)
        assert sliced_axis not in in_slice, (sliced_axis, in_slice)
    print("   ok")


def run_selftests():
    print("=== atlas_view.py selftests (synthetic data only, no GUI) ===")
    selftest_ortho_panes_geometry()
    selftest_crosshair_vectors()
    print("=== all selftests passed ===")
    print("(atlas_reference.py --selftest covers atlas loading / ontology maths)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Browse an atlas annotation against its ontology")
    _local_config.add_config_arg(parser, "atlas_view")
    parser.add_argument("--selftest", action="store_true",
                        help="run the built-in synthetic tests (no GUI, no config) and exit")
    args_cli = parser.parse_args()

    if args_cli.selftest:
        return run_selftests()

    atlas_cfg = _load_local_config(args_cli.config)
    _run_view(atlas_cfg)

    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
