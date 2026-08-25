"""Standalone tool: browse an atlas's annotation volume against its ontology.

A read-only reference viewer -- three synced ortho panes (grayscale template,
full annotation in colour, and whatever the ontology tree currently selects),
plus an ontology panel to pick one or several regions at once (their union
highlighted together -- see _add_region_panel) and a hover panel to read off
the full ancestor chain of whatever the mouse is over. Nothing here writes
anything or registers to anything; it exists purely so you can look at an
atlas and understand its ontology.

The three panes are not restricted to the atlas's own voxel axes: they are
three mutually orthogonal PLANES that can be tilted as a rigid frame (see
_add_plane_panel and plane_slice), so a sample cut at an angle can be matched
by reslicing the atlas at that angle instead of eyeballing it between two
axis-aligned slices. Shift+drag inside any pane rotates the frame about that
pane's own normal, which is exactly "turn the picture in this view and watch
the other two views reslice".

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

The plane geometry (frames, bounds, resampling, crosshairs) is separately
runnable with no display and no config, on purely synthetic data:

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
QDoubleSpinBox = QGridLayout = QSlider = None
ViewerModel = QtViewer = None


def _import_gui():
    global napari, QLabel, QPushButton, QVBoxLayout, QWidget
    global QAbstractItemView, QCheckBox, QLineEdit, QSplitter, QTreeWidget, Qt
    global QDoubleSpinBox, QGridLayout, QSlider
    global ViewerModel, QtViewer
    import napari as _napari
    from napari.components import ViewerModel as _ViewerModel
    from napari.qt import QtViewer as _QtViewer
    from PyQt5.QtCore import Qt as _Qt
    from PyQt5.QtWidgets import (QAbstractItemView as _QAbstractItemView,
                                 QCheckBox as _QCheckBox, QDoubleSpinBox as _QDoubleSpinBox,
                                 QGridLayout as _QGridLayout, QLabel as _QLabel,
                                 QLineEdit as _QLineEdit, QPushButton as _QPushButton,
                                 QSlider as _QSlider, QSplitter as _QSplitter,
                                 QTreeWidget as _QTreeWidget, QVBoxLayout as _QVBoxLayout,
                                 QWidget as _QWidget)
    napari, QLabel, QPushButton = _napari, _QLabel, _QPushButton
    QVBoxLayout, QWidget = _QVBoxLayout, _QWidget
    QAbstractItemView = _QAbstractItemView
    QCheckBox, QLineEdit, QSplitter = _QCheckBox, _QLineEdit, _QSplitter
    QTreeWidget, Qt = _QTreeWidget, _Qt
    QDoubleSpinBox, QGridLayout, QSlider = _QDoubleSpinBox, _QGridLayout, _QSlider
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


# (axis order, pane title). order[0] is the axis the pane's plane is
# perpendicular to when nothing is tilted; order[1]/order[2] are the axes it
# draws down the screen and across it. The panes no longer hand these to
# napari's dims.order -- the plane frame below is indexed by them instead --
# but the meaning, and so the on-screen layout, is unchanged from when they
# did.
_ORTHO_PANES = (
    ((0, 1, 2), "Plane 0 (normally axis 0)"),
    ((1, 0, 2), "Plane 1 (normally axis 1)"),
    ((2, 0, 1), "Plane 2 (normally axis 2)"),
)

# One colour per plane, used for BOTH the line a pane draws where another
# plane cuts through it and that plane's row in the plane panel. With three
# tiltable planes, "which plane is this line?" stops being obvious from the
# line's direction, so it has to be readable from its colour.
_PANE_COLOURS = ("#ff6060", "#48d16a", "#5b9dff")

# Layer attributes mirrored from the main pane onto the ortho panes. Only the
# main pane has a layer list, so these are the knobs a user can actually reach;
# everything else about the sub-panes is fixed at construction.
_MIRRORED_LAYER_ATTRS = ("visible", "opacity", "contrast_limits", "gamma", "colormap")

# Starting height in px of the bottom dock holding the two reconstructed
# views, leaving the main canvas the rest. Draggable afterwards.
_ORTHO_DOCK_HEIGHT = 300

# How far the mouse may travel between press and release and still count as a
# click rather than the start of a camera pan (see _pane_mouse_callback).
_CLICK_SLOP_PX = 4

# Sampling step used WHILE a drag is in flight. Every pane reslices on every
# mouse move, and three oblique planes through an 800^3 annotation is a few
# hundred ms at full resolution -- enough to feel like the window is fighting
# back. Halving the grid quarters that, and the full-resolution pass runs on
# release, so the coarse version is only ever on screen while the mouse is
# actually moving.
_DRAG_STRIDE = 2


# =====================================================================================
# plane geometry -- pure numpy, no napari, no Qt (selftested at the bottom)
# =====================================================================================
def volume_origin(shape):
    """The voxel the plane frame pivots on: the middle voxel of the volume.

    Deliberately an integer voxel index rather than the exact geometric centre
    (shape-1)/2: with an even-sized axis the geometric centre sits at x.5, so
    an un-tilted plane would sample every coordinate exactly halfway between
    two voxels and the rounding would duplicate one row and drop the next.
    Rounding down to a real voxel makes the un-tilted case exact.
    """
    return np.array([(int(n) - 1) // 2 for n in shape], dtype=float)


def identity_frame():
    """The un-tilted frame: plane k's normal is data axis k.

    A frame is a 3x3 rotation whose ROW k is the unit normal of plane k, in
    voxel coordinates. Rows, not columns, so `frame[k]` reads as "plane k's
    normal" everywhere below.
    """
    return np.eye(3)


def rotation_about(axis, angle):
    """Rodrigues rotation matrix: `angle` radians about the (unit) `axis`."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def orthonormal_frame(frame):
    """Gram-Schmidt, to stop a long drag from drifting.

    Every rotation is one more float matrix product on top of the last, and a
    few thousand mouse-move events of that leaves the rows measurably
    non-perpendicular -- which shows up as three panes that are no longer
    quite orthogonal views of each other.
    """
    rows = []
    for row in np.asarray(frame, dtype=float):
        for done in rows:
            row = row - np.dot(row, done) * done
        rows.append(row / np.linalg.norm(row))
    return np.array(rows)


def rotated_frame(frame, pane_axis, angle):
    """`frame` turned by `angle` radians about plane `pane_axis`'s own normal.

    The frame stays rigid: the pane you are dragging in keeps its plane
    exactly (its normal is the rotation axis, so it is fixed), and the OTHER
    two planes swing round it. That is the whole interaction -- one pane's
    picture turns, the other two panes reslice.
    """
    rotation = rotation_about(np.asarray(frame, dtype=float)[pane_axis], angle)
    return orthonormal_frame(np.asarray(frame, dtype=float) @ rotation.T)


def frame_from_euler(angles_deg):
    """Frame from three rotations about the DATA axes, applied 0, then 1, then 2.

    What the plane panel's three spin boxes set. Euler angles are a poor way
    to *drive* a rotation by mouse (they gimbal-lock, and the same frame has
    several spellings) but a good way to type an exact one in and to read the
    current tilt back off -- which is what an atlas user actually wants to
    record ("resliced 12 degrees off coronal").
    """
    a0, a1, a2 = (np.deg2rad(float(a)) for a in angles_deg)
    rotation = (rotation_about((1, 0, 0), a0)
                @ rotation_about((0, 1, 0), a1)
                @ rotation_about((0, 0, 1), a2))
    # Rows are plane normals, i.e. the images of the data axes under the
    # rotation -- the columns of `rotation`.
    return rotation.T


def euler_from_frame(frame):
    """Inverse of frame_from_euler, in degrees.

    Only used to keep the spin boxes showing the frame the mouse has dragged
    the panes to, so the degenerate (gimbal-locked) case just has to be
    stable and round-trip, not to pick any particular one of the infinitely
    many equivalent spellings.
    """
    rotation = np.asarray(frame, dtype=float).T
    sin_a1 = float(np.clip(rotation[0, 2], -1.0, 1.0))
    a1 = np.arcsin(sin_a1)
    if abs(sin_a1) < 1.0 - 1e-9:
        a0 = np.arctan2(-rotation[1, 2], rotation[2, 2])
        a2 = np.arctan2(-rotation[0, 1], rotation[0, 0])
    else:
        a0 = np.arctan2(rotation[1, 0], rotation[1, 1])
        a2 = 0.0
    return tuple(float(np.rad2deg(a)) for a in (a0, a1, a2))


def plane_bounds(shape, origin, direction):
    """(lo, hi): how far along `direction` the volume reaches from `origin`.

    Projecting all eight corners of the box and taking the extremes, in the
    closed form -- the projected half-width of a box is just the sum of its
    half-sides times |direction| componentwise. A pane's sampling grid is
    sized from this, so a tilted plane grows its grid exactly enough to keep
    the whole volume on screen instead of cropping its corners off.
    """
    extent = np.asarray(shape, dtype=float) - 1.0
    direction = np.asarray(direction, dtype=float)
    centre = float(np.dot(extent / 2.0 - np.asarray(origin, dtype=float), direction))
    half = 0.5 * float(np.sum(np.abs(direction) * extent))
    return int(np.floor(centre - half)), int(np.ceil(centre + half))


def resample_plane(volumes, shape, point0, row_dir, col_dir, s_vals, t_vals):
    """Nearest-neighbour resample of every volume in `volumes` on one plane.

    The plane is {point0 + s*row_dir + t*col_dir}, sampled at the given s and
    t (voxel units). All the volumes are read at the SAME rounded coordinates
    -- computed once, shared -- so the annotation, the highlight and the
    template can never disagree about which voxel a screen pixel is, and the
    per-volume cost is one gather rather than a second coordinate pass.

    Nearest, not linear, for all three: the annotation is a label image where
    interpolation would invent structures that are not in the atlas, and
    sampling the template anywhere else than the annotation is sampled would
    put the grayscale half a voxel away from the outline drawn over it.

    Anything falling outside the volume reads back as 0, which is background
    for a label image and black for the template.
    """
    shape = tuple(int(n) for n in shape)
    s_vals = np.asarray(s_vals, dtype=float)[:, None]
    t_vals = np.asarray(t_vals, dtype=float)[None, :]
    inside = None
    coords = []
    for axis in range(3):
        # floor(x + 0.5), not rint: rint is round-half-to-EVEN, so a plane
        # whose coordinates land on exact halves (any even-sized axis) would
        # sample one voxel twice and skip its neighbour, in stripes.
        coord = np.floor(point0[axis] + s_vals * row_dir[axis] + t_vals * col_dir[axis] + 0.5)
        ok = (coord >= 0) & (coord < shape[axis])
        inside = ok if inside is None else (inside & ok)
        coords.append(np.clip(coord, 0, shape[axis] - 1).astype(np.int64))
    flat = (coords[0] * shape[1] + coords[1]) * shape[2] + coords[2]

    sampled = []
    for volume in volumes:
        if not volume.flags.c_contiguous:
            volume = np.ascontiguousarray(volume)
        plane = volume.reshape(-1)[flat]
        plane[~inside] = 0
        sampled.append(plane)
    return sampled


def plane_slice(volumes, shape, origin, frame, centre, order, stride=1):
    """Everything one pane draws, for the current frame and crosshair.

    `order` is one _ORTHO_PANES entry: order[0] picks the frame row that is
    this pane's normal, order[1] the row drawn down the screen and order[2]
    the row drawn across it. At the identity frame that reproduces the plain
    axis-aligned slices this viewer used to show, voxel for voxel.

    In-plane coordinates (s, t) are measured from `origin`, NOT from the
    crosshair, so a given anatomical point keeps the same coordinate as the
    crosshair moves -- clicking a new point must not slide the picture out
    from under the mouse. `stride` samples every nth voxel for a cheaper
    (coarser) pass during drags; the returned s_min/t_min still say where the
    grid starts, so the caller can place the image identically either way.

    Returns images in the order `volumes` were given, plus the grid bounds and
    the crosshair's own (s, t) -- which is where the other two planes cut
    through this one.
    """
    frame = np.asarray(frame, dtype=float)
    normal, row_dir, col_dir = frame[order[0]], frame[order[1]], frame[order[2]]
    rel = np.asarray(centre, dtype=float) - np.asarray(origin, dtype=float)
    offset = float(np.dot(rel, normal))

    s_min, s_max = plane_bounds(shape, origin, row_dir)
    t_min, t_max = plane_bounds(shape, origin, col_dir)
    s_vals = np.arange(s_min, s_max + 1, stride, dtype=float)
    t_vals = np.arange(t_min, t_max + 1, stride, dtype=float)

    point0 = np.asarray(origin, dtype=float) + offset * normal
    images = resample_plane(volumes, shape, point0, row_dir, col_dir, s_vals, t_vals)
    return SimpleNamespace(images=images, offset=offset,
                           s_min=s_min, s_max=s_max, t_min=t_min, t_max=t_max,
                           s_centre=float(np.dot(rel, row_dir)),
                           t_centre=float(np.dot(rel, col_dir)))


def expand_coarse(image, rows, cols, stride):
    """A stride-sampled plane blown back up to the full grid.

    So that a coarse (drag-time) pass lands in exactly the same place on
    screen as the full one: the picture gets blockier and nothing moves. The
    alternative -- leaving it small and scaling the layer up by `stride`
    instead -- means writing a napari layer's `scale` on every mouse move,
    which is both more work for vispy and (on at least one driver here) a
    crash on teardown.
    """
    if stride == 1:
        return image
    return image.repeat(stride, axis=0).repeat(stride, axis=1)[:rows, :cols]


def plane_point(origin, frame, order, offset, s, t):
    """The voxel coordinate of in-plane point (s, t) on the plane `offset`
    away from `origin` along plane `order[0]`'s normal. Inverse of the
    projection plane_slice reports -- what a click in a pane means in 3D."""
    frame = np.asarray(frame, dtype=float)
    return (np.asarray(origin, dtype=float) + offset * frame[order[0]]
            + s * frame[order[1]] + t * frame[order[2]])


def plane_offsets(centre, frame, origin):
    """(d0, d1, d2): how far the crosshair sits along each plane's normal.

    The three numbers the plane panel's sliders hold; together with the frame
    they are just another spelling of the crosshair position, which is why
    centre_from_offsets can invert them exactly.
    """
    rel = np.asarray(centre, dtype=float) - np.asarray(origin, dtype=float)
    return tuple(float(np.dot(rel, np.asarray(frame, dtype=float)[k])) for k in range(3))


def centre_from_offsets(offsets, frame, origin):
    """Crosshair voxel coordinate from the three per-plane offsets."""
    frame = np.asarray(frame, dtype=float)
    return np.asarray(origin, dtype=float) + np.asarray(offsets, dtype=float) @ frame


def crosshair_vectors(sl, shift=(0.0, 0.0)):
    """napari Vectors data (2, 2, 2) for one pane: the two lines where the
    other two planes cut through this one.

    Two lines, not three: the third plane is this pane's own, and it meets the
    pane everywhere rather than along a line. Because the frame stays rigid,
    those two cuts are always perpendicular and always run along the pane's
    own axes -- what changes when you tilt is the picture underneath them, not
    the cross. The line at constant s belongs to plane order[1] and the one at
    constant t to plane order[2], which is what _PANE_COLOURS colours them by.
    """
    s_shift, t_shift = float(shift[0]), float(shift[1])
    s_c, t_c = sl.s_centre + s_shift, sl.t_centre + t_shift
    s_lo, t_lo = sl.s_min + s_shift, sl.t_min + t_shift
    return np.array([[[s_c, t_lo], [0.0, float(sl.t_max - sl.t_min)]],
                     [[s_lo, t_c], [float(sl.s_max - sl.s_min), 0.0]]], dtype=float)


def pane_handedness(frame, order):
    """+1 or -1: whether (normal, row, col) is right- or left-handed.

    Pane 1 draws axes (0, 2) down and across, which with axis 1 pointing at
    the viewer is a LEFT-handed triple, while panes 0 and 2 are right-handed.
    Without this sign a rotation drag would turn the picture the wrong way in
    exactly one of the three panes.
    """
    frame = np.asarray(frame, dtype=float)
    return float(np.sign(np.dot(np.cross(frame[order[0]], frame[order[1]]), frame[order[2]])))


# =====================================================================================
# the window
# =====================================================================================
def _add_atlas_layers(model, atlas, voxel, features, order, first):
    """The atlas's four layers, added to one ViewerModel in draw order.

    Called once per ortho pane. Unlike the axis-aligned viewer this grew out
    of, the image layers hold a 2D RESLICE of the volume rather than the
    volume itself -- napari slices along array axes only, and the whole point
    here is planes that are not array axes. The reslicing is ours (plane_slice
    -> the window's refresh()); what napari still owns is the display:
    colormaps, contrast, opacity, the layer list and the camera.

    Every layer's `scale` is set HERE and never written again -- the reslice
    always fills the same world-sized grid (see expand_coarse), so only `data`
    and `translate` move afterwards.
    """
    scale = (voxel, voxel)
    template = None
    if atlas.template is not None:
        # Contrast from the WHOLE volume (subsampled), not from the slice the
        # window happens to open on: napari would otherwise pick limits from
        # one plane and every other plane would be wrongly stretched.
        sample = atlas.template[::4, ::4, ::4]
        lo, hi = float(sample.min()), float(sample.max())
        template = model.add_image(first.images[2], name="reference (template grayscale)",
                                   colormap="gray", scale=scale,
                                   contrast_limits=[lo, hi if hi > lo else lo + 1.0])
    annotation = model.add_labels(first.images[0], name="annotation (all regions)",
                                  opacity=0.45, features=features, scale=scale)
    annotation.editable = False
    highlight = model.add_labels(first.images[1], name="selection (selected regions)",
                                 opacity=0.85, scale=scale,
                                 colormap=napari.utils.DirectLabelColormap(
                                     color_dict={None: "transparent", 1: "red"}))
    highlight.editable = False              # a reference; painting here would mean nothing
    cross = model.add_vectors(crosshair_vectors(first), name="crosshair",
                              edge_color=[_PANE_COLOURS[order[1]], _PANE_COLOURS[order[2]]],
                              edge_width=1.5, vector_style="line", opacity=0.9, scale=scale)
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

    THREE PANES (`ortho`), one per plane of a shared, tiltable orthogonal
    FRAME: a region's shape in the other two planes is often what tells you
    whether the tree selection is the structure you meant, and a sample cut
    obliquely needs the atlas resliced obliquely to be comparable at all.
    napari has neither an ortho mode nor oblique slicing, so each pane is its
    own ViewerModel holding a 2D image that this file reslices out of the
    volume (plane_slice) whenever the frame or the crosshair moves. Pane 0 is
    the real napari Viewer (it owns the window, the layer list and the layer
    controls) and the other two are bare QtViewer canvases in a dock beneath
    it.

    ONE CROSSHAIR over all of them, and it is now two PLANE CUTS rather than
    two lines through a voxel grid: left-clicking any pane recentres all three
    planes on the clicked point, and Shift+left-drag turns the whole frame
    about the dragged pane's normal, so that pane's picture follows the mouse
    while the other two reslice live. Both go through one piece of state
    (`state.frame`, `state.centre`), so the sliders, the angle boxes, clicks,
    drags and "jump to region" cannot disagree about where the planes are.

    A HOVER PANEL (_add_ancestry_panel) on the right, because the one structure
    a voxel is labelled with is usually a leaf ("layer 5 of primary motor
    area") and the level you are actually picking is one of its ancestors. A
    PLANE PANEL (_add_plane_panel) above it holds the frame itself: one
    position slider per plane and one angle box per data axis, for the times
    you want an exact 12 degrees rather than a dragged one. A separate
    REGION-SELECTION panel (_add_region_panel) on the left is what drives the
    tree click -> highlight path -- see that function for why it gets its own
    dedicated side rather than sharing one with the hover panel.

    The atlas grid here is INDEPENDENT of any sample grid -- nothing in this
    window is registered to anything, it is a reference only. What the atlas's
    own extent, orientation and downsampling affect is exactly this display,
    nothing else.
    """
    voxel = float(resolution_um) * atlas.downsample if resolution_um else 1.0
    shape = atlas.compact.shape
    origin = volume_origin(shape)
    features = atlas_reference.annotation_features(atlas)
    panes = tuple(_ORTHO_PANES if ortho else _ORTHO_PANES[:1])

    # The one place the planes live. `highlight` is kept here as a full volume
    # (not just as the panes' 2D slices) because the region panel jumps to the
    # centre of the selection, which is a 3D question.
    state = SimpleNamespace(frame=identity_frame(),
                            centre=origin.copy(),
                            highlight=np.zeros(shape, dtype=np.uint8),
                            stride=1,
                            listeners=[])

    def volumes():
        """The volumes every pane resamples, in the order _add_atlas_layers
        and refresh() both read them back out."""
        vols = [atlas.compact, state.highlight]
        if atlas.template is not None:
            vols.append(atlas.template)
        return vols

    def slice_for(pane):
        return plane_slice(volumes(), shape, origin, state.frame, state.centre,
                           pane.order, stride=state.stride)

    viewer = napari.Viewer(title="Atlas viewer")
    main = SimpleNamespace(model=viewer, qt=None, order=panes[0][0], shift=np.zeros(2))
    main.layers = _add_atlas_layers(viewer, atlas, voxel, features, main.order,
                                    slice_for(main))
    built = [main]

    for order, _title in panes[1:]:
        model = ViewerModel(ndisplay=2)
        # Canvas BEFORE layers: napari 0.8.0's QtViewer.__init__ walks the
        # layers already in the model and reorders them against a visual map
        # it has not filled in yet, which raises KeyError on the second layer.
        # Adding to an empty model routes through the same code path one layer
        # at a time, where the map is always current.
        qt = QtViewer(model)
        pane = SimpleNamespace(model=model, qt=qt, order=order, shift=np.zeros(2))
        pane.layers = _add_atlas_layers(model, atlas, voxel, features, order, slice_for(pane))
        built.append(pane)
        model.reset_view()

    if len(built) > 1:
        splitter = QSplitter(Qt.Horizontal)
        for pane, (order, title) in zip(built[1:], panes[1:]):
            wrapper = QWidget()
            layout = QVBoxLayout(wrapper)
            layout.setContentsMargins(0, 0, 0, 0)
            heading = QLabel(title)
            heading.setStyleSheet(f"color: {_PANE_COLOURS[order[0]]};")
            layout.addWidget(heading)
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
        _mirror_layer_attrs(built)
        # QtViewer holds a vispy canvas and its GL context; napari only cleans
        # up the ones its own Window created, so these two have to be closed by
        # hand or the context outlives the window it was drawn in.
        viewer.window._qt_window.destroyed.connect(
            lambda *_a: [pane.qt.close() for pane in built[1:]])

    def refresh():
        """Reslice every pane for the current frame/crosshair and place the
        result. Called on every change, from one place, for the same reason
        the old viewer derived its crosshair from the slider instead of
        storing it: there is then no second copy of where the planes are."""
        for pane in built:
            sl = slice_for(pane)
            rows, cols = sl.s_max - sl.s_min + 1, sl.t_max - sl.t_min + 1
            translate = ((sl.s_min + pane.shift[0]) * voxel,
                         (sl.t_min + pane.shift[1]) * voxel)
            roles = ["annotation", "highlight"] + (
                ["template"] if pane.layers.template is not None else [])
            for role, image in zip(roles, sl.images):
                layer = getattr(pane.layers, role)
                layer.data = expand_coarse(image, rows, cols, state.stride)
                layer.translate = translate
            pane.layers.cross.data = crosshair_vectors(sl, pane.shift)

    def apply_state():
        refresh()
        for listener in state.listeners:
            listener()

    def crosshair_in_pane(pane):
        """The crosshair's position in `pane`'s own drawn coordinates, shift
        included -- i.e. where on screen the cross is."""
        rel = state.centre - origin
        return np.array([np.dot(rel, state.frame[pane.order[1]]) + pane.shift[0],
                         np.dot(rel, state.frame[pane.order[2]]) + pane.shift[1]])

    def set_frame(frame):
        """Turn the planes, pivoting each pane's picture on ITS CROSSHAIR.

        In-plane coordinates are measured from the volume's middle voxel, so
        left alone a rotation would swing the picture about the middle of the
        atlas -- away from whatever the user centred on before reaching for
        the rotation, which is invariably the thing they are trying to line
        up. Each pane keeps a `shift` that absorbs the difference, so the
        crosshair stays put on screen and the picture turns around it, while
        clicks (which do not touch the frame) still leave the picture alone.
        """
        before = [crosshair_in_pane(pane) for pane in built]
        state.frame = orthonormal_frame(frame)
        for pane, was in zip(built, before):
            pane.shift += was - crosshair_in_pane(pane)
        apply_state()

    def centre_on(index):
        """Point every plane at voxel `index` -- so the two reconstructions
        land on it too, not just the pane that was clicked."""
        state.centre = np.clip(np.asarray(index, dtype=float), 0,
                               np.asarray(shape, dtype=float) - 1)
        apply_state()

    def set_offsets(offsets):
        centre_on(centre_from_offsets(offsets, state.frame, origin))

    def rotate(pane_axis, angle):
        set_frame(rotated_frame(state.frame, pane_axis, angle))

    def set_euler(angles_deg):
        set_frame(frame_from_euler(angles_deg))

    def reset_planes():
        state.centre = origin.copy()
        state.frame = identity_frame()
        for pane in built:
            pane.shift[:] = 0.0
        apply_state()
        for pane in built:
            pane.model.reset_view()

    def set_drag(active):
        """Coarse sampling while the mouse is down, full resolution once it
        comes back up."""
        state.stride = _DRAG_STRIDE if active else 1
        refresh()

    def set_highlight(mask, name=None):
        """The selection layer's data, on every pane at once.

        The 3D mask is what is stored; each pane's 2D version is resliced out
        of it alongside the annotation, at exactly the same coordinates, so
        the highlight cannot land half a voxel off the region it highlights.
        """
        state.highlight = np.ascontiguousarray(mask)
        if name:
            for pane in built:
                pane.layers.highlight.name = name
        refresh()

    def pane_position(pane, world):
        """A napari world position in `pane` -> the voxel it points at."""
        s = float(world[0]) / voxel - pane.shift[0]
        t = float(world[1]) / voxel - pane.shift[1]
        offset = float(np.dot(state.centre - origin, state.frame[pane.order[0]]))
        return plane_point(origin, state.frame, pane.order, offset, s, t)

    win = SimpleNamespace(viewer=viewer, panes=built, state=state, origin=origin,
                          voxel=voxel, shape=shape,
                          set_highlight=set_highlight, centre_on=centre_on,
                          set_offsets=set_offsets, set_euler=set_euler,
                          reset_planes=reset_planes, rotate=rotate, set_drag=set_drag,
                          crosshair_in_pane=crosshair_in_pane, pane_position=pane_position)

    for pane in built:
        pane.model.mouse_drag_callbacks.append(_pane_mouse_callback(pane, win))
    main.model.reset_view()

    win.hover = _add_ancestry_panel(viewer, atlas, built)
    _add_plane_panel(viewer, win)
    return win


def _pane_mouse_callback(pane, win):
    """Left-click anywhere in a pane -> crosshair there, in every pane;
    Shift+left-drag -> turn the frame about this pane's normal.

    A generator callback, which is napari's way of telling a click from the
    start of a drag: napari resumes it on every mouse_move and once more on
    release, so the decision can be made after the fact. A plain press handler
    cannot -- the same press that begins a camera pan would jump the crosshair
    on its way, and the atlas would slide out from under the pan.

    Shift is what separates "turn the planes" from "pan the camera"; an
    unmodified drag still belongs to napari, so panning and zooming are
    untouched.
    """
    def callback(_model, event):
        if event.button != 1:
            return                          # right-click menus stay napari's
        if "Shift" in event.modifiers:
            yield from _rotation_drag(pane, win, event)
            return
        if event.modifiers:
            return                          # other modified drags stay napari's
        origin_px = np.asarray(event.pos, dtype=float)
        dragged = False
        yield
        while event.type == "mouse_move":
            if np.abs(np.asarray(event.pos, dtype=float) - origin_px).max() > _CLICK_SLOP_PX:
                dragged = True
            yield
        if not dragged:
            # event.position is this pane's 2D world position, which together
            # with the pane's own plane offset is a full 3D point -- the click
            # lands ON the plane being clicked, so that pane does not move and
            # the other two do.
            win.centre_on(win.pane_position(pane, event.position))
    return callback


def _rotation_drag(pane, win, event):
    """Shift+drag: the angle the cursor sweeps around the crosshair becomes
    the angle the frame turns through, about this pane's normal.

    Measured around the crosshair because that is what set_frame pins the
    picture to; the sign flips with the pane's handedness so that the picture
    follows the mouse in all three panes rather than in two of them (see
    pane_handedness).
    """
    axis = pane.order[0]
    sign = pane_handedness(win.state.frame, pane.order)

    def cursor_angle():
        rel = np.asarray(event.position, dtype=float)[:2] / win.voxel - win.crosshair_in_pane(pane)
        if np.hypot(*rel) < 1.0:
            return None                     # on top of the pivot: no meaningful angle
        return float(np.arctan2(rel[0], rel[1]))   # s down, t across: screen angle

    last = cursor_angle()
    win.set_drag(True)
    # A left-drag is also how napari pans, and it does that from vispy rather
    # than through these callbacks -- so without muting it for the duration the
    # canvas would slide sideways underneath the rotation.
    camera = pane.model.camera
    camera.mouse_pan = False
    try:
        yield
        while event.type == "mouse_move":
            now = cursor_angle()
            if now is not None and last is not None:
                delta = now - last
                # atan2 wraps at +-pi; a drag straight through the wrap would
                # otherwise read as a near-full turn the other way.
                delta = float(np.arctan2(np.sin(delta), np.cos(delta)))
                win.rotate(axis, sign * delta)
            if now is not None:
                last = now
            yield
    finally:
        camera.mouse_pan = True
        win.set_drag(False)


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


def _add_plane_panel(viewer, win):
    """The frame itself, as numbers: where each plane sits and how far the
    whole frame is turned.

    The mouse is the fast way to place the planes and a poor way to record
    what it did, which for an oblique atlas reslice is half the point -- "12
    degrees off coronal" is the thing you write down and reproduce on the next
    sample. So the same state the drags write is exposed here as three
    position sliders (one per plane, along that plane's own normal) and three
    angle boxes (one per DATA axis, applied 0 then 1 then 2), each of which
    both follows a drag and can be typed into.

    Angles are per data axis rather than per pane because a pane's normal is
    itself a function of the angles -- three numbers that each move the frame
    they are measured in would never settle.
    """
    state, origin, voxel = win.state, win.origin, win.voxel
    busy = {"in": False}
    sliders, readouts, angle_boxes = [], [], []

    positions = QGridLayout()
    for axis in range(3):
        swatch = QLabel()
        swatch.setFixedWidth(10)
        swatch.setStyleSheet(f"background: {_PANE_COLOURS[axis]};")
        slider = QSlider(Qt.Horizontal)
        readout = QLabel()
        readout.setMinimumWidth(90)
        positions.addWidget(swatch, axis, 0)
        positions.addWidget(QLabel(f"plane {axis}"), axis, 1)
        positions.addWidget(slider, axis, 2)
        positions.addWidget(readout, axis, 3)
        sliders.append(slider)
        readouts.append(readout)

    angles = QGridLayout()
    for axis in range(3):
        box = QDoubleSpinBox()
        box.setRange(-180.0, 180.0)
        box.setDecimals(1)
        box.setSingleStep(1.0)
        box.setSuffix(" deg")
        box.setWrapping(True)
        # Otherwise every keystroke of "-12.5" reslices three planes, twice on
        # the way to a number the user has not finished typing.
        box.setKeyboardTracking(False)
        angles.addWidget(QLabel(f"about axis {axis}"), axis, 0)
        angles.addWidget(box, axis, 1)
        angle_boxes.append(box)

    status = QLabel()
    reset = QPushButton("Reset planes (axis-aligned, centred)")

    def on_position(*_args):
        if busy["in"]:
            return
        win.set_offsets([float(s.value()) for s in sliders])

    def on_angle(*_args):
        if busy["in"]:
            return
        win.set_euler([b.value() for b in angle_boxes])

    def sync():
        """Follow the state, whatever moved it.

        `busy` is what stops the obvious loop: writing a slider emits its own
        valueChanged, which would drive the state that is currently writing
        the slider.
        """
        busy["in"] = True
        try:
            offsets = plane_offsets(state.centre, state.frame, origin)
            for axis, (slider, readout) in enumerate(zip(sliders, readouts)):
                lo, hi = plane_bounds(win.shape, origin, state.frame[axis])
                slider.setRange(lo, hi)
                slider.setValue(int(round(offsets[axis])))
                readout.setText(f"{offsets[axis] * voxel:,.0f} um"
                                if voxel != 1.0 else f"{offsets[axis]:.0f} vox")
            for box, angle in zip(angle_boxes, euler_from_frame(state.frame)):
                box.setValue(angle)
            centre = ", ".join(f"{int(round(v))}" for v in state.centre)
            # The angle between a plane's normal and the data axis it started
            # on: the one number that says "these panes are not axis-aligned
            # any more" without having to read three Euler angles.
            tilt = max(np.rad2deg(np.arccos(np.clip(abs(state.frame[k][k]), -1.0, 1.0)))
                       for k in range(3))
            status.setText(f"crosshair voxel ({centre}); planes tilted up to {tilt:.1f} deg "
                           f"off the atlas axes")
        finally:
            busy["in"] = False

    for slider in sliders:
        slider.valueChanged.connect(on_position)
        slider.sliderPressed.connect(lambda: win.set_drag(True))
        slider.sliderReleased.connect(lambda: win.set_drag(False))
    for box in angle_boxes:
        box.valueChanged.connect(on_angle)
    reset.clicked.connect(lambda *_a: win.reset_planes())

    dock = QWidget()
    layout = QVBoxLayout(dock)
    layout.addWidget(QLabel(
        "Three orthogonal planes, tiltable together.\n"
        "Click a view to move all three; Shift+drag inside a view to turn the\n"
        "frame about that view's own normal (its picture follows the mouse,\n"
        "the other two views reslice). Coloured lines mark where each other\n"
        "plane cuts through the view."))
    layout.addWidget(QLabel("Position along each plane's normal"))
    layout.addLayout(positions)
    layout.addWidget(QLabel("Orientation (rotations about the atlas axes)"))
    layout.addLayout(angles)
    layout.addWidget(ontology_tree_ui.scrollable(status, 56))
    layout.addWidget(reset)
    viewer.window.add_dock_widget(dock, area="right", name="Planes")

    state.listeners.append(sync)
    sync()
    return SimpleNamespace(sync=sync, sliders=sliders, angle_boxes=angle_boxes)


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
            # The annotation layer is a 2D reslice now, so the value under the
            # cursor is a plain lookup in it -- no view direction, no slider
            # axis to reconstruct.
            show(pane.layers.annotation.get_value(event.position, world=True))
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
        centre = atlas_reference.mask_centre_index(win.state.highlight)
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
def _ramp_volume(shape):
    """A volume whose every voxel holds its own flat index, so a resampled
    plane can be checked coordinate by coordinate."""
    return np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)


def selftest_ortho_panes_geometry():
    print("1. atlas ortho panes: every axis is the normal exactly once")
    # Every axis is one pane's normal, and is drawn by the other two -- i.e.
    # the three panes really are three orthogonal views and not two copies of
    # one.
    orders = [order for order, _title in _ORTHO_PANES]
    assert len(orders) == 3, orders
    normals = [order[0] for order in orders]
    assert sorted(normals) == [0, 1, 2], normals
    for order in orders:
        assert sorted(order) == [0, 1, 2], order
    # The sign a rotation drag has to be corrected by is a property of the
    # pane layout, so it is worth pinning down here: pane 1 draws (0, 2),
    # which with axis 1 towards the viewer is left-handed.
    signs = [pane_handedness(identity_frame(), order) for order in orders]
    assert signs == [1.0, -1.0, 1.0], signs
    print("   ok")


def selftest_frame_algebra():
    print("2. plane frames: orthonormal under rotation, and Euler round-trips")
    frame = identity_frame()
    for axis, angle in ((0, 0.4), (2, -1.1), (1, 0.9), (0, 2.5)):
        frame = rotated_frame(frame, axis, angle)
        assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-9), frame
        assert np.isclose(np.linalg.det(frame), 1.0), np.linalg.det(frame)

    # Rotating about a pane's own normal leaves that pane's plane alone and
    # swings the other two -- the property the whole interaction rests on.
    turned = rotated_frame(identity_frame(), 1, np.pi / 2)
    assert np.allclose(turned[1], [0, 1, 0]), turned
    assert not np.allclose(turned[0], [1, 0, 0]), turned

    for angles in ((0, 0, 0), (12.5, 0, 0), (0, -30, 0), (5, 10, -20), (170, 40, 90)):
        back = euler_from_frame(frame_from_euler(angles))
        assert np.allclose(back, angles, atol=1e-6), (angles, back)
    # A frame reached by dragging must decompose too, or the angle boxes could
    # not follow the mouse.
    assert np.allclose(frame_from_euler(euler_from_frame(frame)), frame, atol=1e-9)
    print("   ok")


def selftest_plane_bounds():
    print("3. plane bounds: the grid covers every corner of the volume")
    shape = (6, 8, 10)
    origin = volume_origin(shape)
    assert np.array_equal(origin, [2, 3, 4]), origin
    corners = np.array([[i, j, k] for i in (0, shape[0] - 1)
                        for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)], dtype=float)
    rng = np.random.default_rng(0)
    for _ in range(20):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        lo, hi = plane_bounds(shape, origin, direction)
        projected = (corners - origin) @ direction
        assert lo <= projected.min() and hi >= projected.max(), (lo, hi, projected)
        assert lo >= projected.min() - 1 and hi <= projected.max() + 1, (lo, hi, projected)
    print("   ok")


def selftest_identity_slices():
    print("4. un-tilted planes reproduce the plain axis-aligned slices exactly")
    shape = (6, 8, 10)
    volume = _ramp_volume(shape)
    origin = volume_origin(shape)
    frame = identity_frame()
    for order, _title in _ORTHO_PANES:
        for offset in (-2, 0, 3):
            centre = origin.copy()
            centre[order[0]] += offset
            sl = plane_slice([volume], shape, origin, frame, centre, order)
            image = sl.images[0]
            index = int(origin[order[0]]) + offset
            expected = np.take(volume, index, axis=order[0])
            if order[1] > order[2]:
                expected = expected.T
            # The grid is sized to the box, which for an even-sized axis is one
            # sample wider than the volume; that extra line reads as background.
            assert image.shape[0] >= expected.shape[0], (image.shape, expected.shape)
            rows, cols = expected.shape
            assert np.array_equal(image[:rows, :cols], expected), order
            assert np.all(image[rows:] == 0) and np.all(image[:, cols:] == 0), order
            assert np.isclose(sl.offset, offset), (sl.offset, offset)
    print("   ok")


def selftest_oblique_sampling():
    print("5. tilted planes: every sample is the voxel the geometry says it is")
    shape = (9, 11, 13)
    volume = _ramp_volume(shape)
    origin = volume_origin(shape)
    frame = rotated_frame(rotated_frame(identity_frame(), 0, np.deg2rad(25)),
                          2, np.deg2rad(-15))
    order = (1, 0, 2)
    centre = origin + np.array([1.0, -2.0, 0.5])
    sl = plane_slice([volume], shape, origin, frame, centre, order)
    image = sl.images[0]

    rng = np.random.default_rng(1)
    for _ in range(200):
        i = int(rng.integers(image.shape[0]))
        j = int(rng.integers(image.shape[1]))
        point = plane_point(origin, frame, order, sl.offset, sl.s_min + i, sl.t_min + j)
        voxel = np.floor(point + 0.5).astype(int)
        if np.any(voxel < 0) or np.any(voxel >= np.array(shape)):
            assert image[i, j] == 0, (i, j, voxel, image[i, j])
        else:
            flat = (voxel[0] * shape[1] + voxel[1]) * shape[2] + voxel[2]
            assert image[i, j] == flat, (i, j, voxel, image[i, j], flat)

    # A tilted plane still holds the point it is centred on, to well under a
    # voxel -- otherwise clicking would drift the crosshair off its own plane.
    back = plane_point(origin, frame, order, sl.offset, sl.s_centre, sl.t_centre)
    assert np.allclose(back, centre, atol=1e-9), (back, centre)

    # Every volume handed to one call is read at the same coordinates: a mask
    # built from the volume must resample to the mask of the resampled volume.
    mask = (volume % 7 == 0).astype(np.uint8)
    both = plane_slice([volume, mask], shape, origin, frame, centre, order)
    inside = both.images[0] != 0
    assert np.array_equal(both.images[1][inside], (both.images[0][inside] % 7 == 0), ), "mask"
    print("   ok")


def selftest_offsets_roundtrip():
    print("6. plane offsets: sliders and the crosshair are one state, not two")
    shape = (6, 8, 10)
    origin = volume_origin(shape)
    frame = rotated_frame(identity_frame(), 1, np.deg2rad(37))
    for centre in (origin, origin + np.array([1.0, -2.0, 3.5])):
        offsets = plane_offsets(centre, frame, origin)
        assert np.allclose(centre_from_offsets(offsets, frame, origin), centre), offsets
    # Un-tilted, an offset IS the voxel index minus the middle voxel, which is
    # what makes the sliders readable as slice positions.
    offsets = plane_offsets(origin + np.array([2.0, 0.0, 0.0]), identity_frame(), origin)
    assert np.allclose(offsets, (2, 0, 0)), offsets
    print("   ok")


def selftest_crosshair_vectors():
    print("7. crosshair: two full-width lines crossing where the other planes cut")
    shape = (6, 8, 10)
    volume = np.zeros(shape, dtype=np.uint8)
    origin = volume_origin(shape)
    order = (0, 1, 2)
    centre = origin + np.array([0.0, 2.0, -3.0])
    sl = plane_slice([volume], shape, origin, identity_frame(), centre, order)
    vectors = crosshair_vectors(sl)
    assert vectors.shape == (2, 2, 2), vectors.shape
    # Line 0 is the plane-order[1] cut: constant s, spanning the full width.
    assert np.isclose(vectors[0, 0, 0], sl.s_centre) and vectors[0, 1, 0] == 0
    assert np.isclose(vectors[0, 1, 1], sl.t_max - sl.t_min)
    # Line 1 is the plane-order[2] cut: constant t, spanning the full height.
    assert np.isclose(vectors[1, 0, 1], sl.t_centre) and vectors[1, 1, 1] == 0
    assert np.isclose(vectors[1, 1, 0], sl.s_max - sl.s_min)
    # A pane's shift moves the cross and its grid together -- it is what keeps
    # the crosshair still on screen while a rotation turns the picture.
    shifted = crosshair_vectors(sl, (2.5, -1.5))
    assert np.isclose(shifted[0, 0, 0], sl.s_centre + 2.5), shifted
    assert np.isclose(shifted[1, 0, 1], sl.t_centre - 1.5), shifted
    print("   ok")


def selftest_coarse_expansion():
    print("8. coarse drag passes land exactly where the full-resolution one does")
    shape = (24, 30, 36)
    volume = _ramp_volume(shape)
    origin = volume_origin(shape)
    frame = frame_from_euler((13, 21, -7))
    centre = origin + np.array([1.0, 2.0, -3.0])
    for order, _title in _ORTHO_PANES:
        full = plane_slice([volume], shape, origin, frame, centre, order)
        coarse = plane_slice([volume], shape, origin, frame, centre, order,
                             stride=_DRAG_STRIDE)
        rows, cols = full.s_max - full.s_min + 1, full.t_max - full.t_min + 1
        # Same grid origin, so the same `translate` places both...
        assert (coarse.s_min, coarse.t_min) == (full.s_min, full.t_min), order
        grown = expand_coarse(coarse.images[0], rows, cols, _DRAG_STRIDE)
        # ...and the same shape, so nothing on screen moves or resizes.
        assert grown.shape == full.images[0].shape == (rows, cols), (grown.shape, order)
        # Every stride-th sample is the real one; the rest are its neighbours.
        assert np.array_equal(grown[::_DRAG_STRIDE, ::_DRAG_STRIDE],
                              full.images[0][::_DRAG_STRIDE, ::_DRAG_STRIDE]), order
    print("   ok")


def selftest_rotation_drag():
    print("9. rotation drags: the picture follows the mouse, the crosshair holds still")

    def on_screen(frame, shift, origin, order, point):
        """Where a voxel lands in a pane's drawn (s, t), the way _refresh
        places the image and crosshair_vectors places the cross."""
        rel = np.asarray(point, dtype=float) - origin
        return np.array([np.dot(rel, frame[order[1]]) + shift[0],
                         np.dot(rel, frame[order[2]]) + shift[1]])

    shape = (24, 30, 36)
    origin = volume_origin(shape)
    swept = np.deg2rad(11.0)                # the angle the cursor sweeps
    for order, _title in _ORTHO_PANES:
        frame = frame_from_euler((7, -13, 21))      # start off-axis; nothing special
        centre = origin + np.array([2.0, -3.0, 1.5])
        probe = origin + np.array([5.0, 4.0, -6.0])
        cross_before = on_screen(frame, (0, 0), origin, order, centre)
        rel_before = on_screen(frame, (0, 0), origin, order, probe) - cross_before

        # Exactly what _rotation_drag and set_frame do between two mouse moves.
        turned = rotated_frame(frame, order[0], pane_handedness(frame, order) * swept)
        shift = cross_before - on_screen(turned, (0, 0), origin, order, centre)
        cross_after = on_screen(turned, shift, origin, order, centre)
        rel_after = on_screen(turned, shift, origin, order, probe) - cross_after

        # The crosshair is the pivot: it must not move on screen at all...
        assert np.allclose(cross_before, cross_after, atol=1e-9), (order, cross_before, cross_after)
        # ...and everything else turns rigidly around it, by the angle swept,
        # in the SAME direction in all three panes (the handedness correction).
        assert np.isclose(np.linalg.norm(rel_before), np.linalg.norm(rel_after)), order
        angle = (np.arctan2(rel_after[0], rel_after[1])
                 - np.arctan2(rel_before[0], rel_before[1]))
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        assert np.isclose(angle, swept), (order, np.rad2deg(angle))
    print("   ok")


def run_selftests():
    print("=== atlas_view.py selftests (synthetic data only, no GUI) ===")
    selftest_ortho_panes_geometry()
    selftest_frame_algebra()
    selftest_plane_bounds()
    selftest_identity_slices()
    selftest_oblique_sampling()
    selftest_offsets_roundtrip()
    selftest_crosshair_vectors()
    selftest_coarse_expansion()
    selftest_rotation_drag()
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
