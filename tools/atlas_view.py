"""Standalone tool: browse an atlas's annotation volume against its ontology.

A read-only reference viewer -- three synced ortho panes (grayscale template,
full annotation in colour, and whatever the ontology tree currently selects),
plus an ontology panel to pick one or several regions at once (their union
highlighted together -- see _add_region_panel) and a wide hover bar along the
bottom (shared/hover_bar.py, wired up by _add_hover_bar) reading off the
deepest levels of the ancestor chain of whatever the mouse is over, in that
region's own atlas colour. Nothing here writes
anything or registers to anything; it exists purely so you can look at an
atlas and understand its ontology.

The three panes are not restricted to the atlas's own voxel axes: they are
three mutually orthogonal PLANES that can be tilted as a rigid frame (see
_add_plane_panel and plane_slice), so a sample cut at an angle can be matched
by reslicing the atlas at that angle instead of eyeballing it between two
axis-aligned slices. The ATLAS ITSELF never moves: what you drag is the pair
of coloured lines lying across a pane -- where the other two planes cut
through it -- and those two panes reslice to the angle you aim them at. Each
pane also carries its own slider, under its own canvas, for scrolling that
plane along its normal.

Point it at a SAMPLE as well (sample_path in the config, see
_add_sample_panel) and the reading changes: the three planes then stand for
the sample's own axes -- the light-sheet frame, which is fixed, because the
stack was cut at whatever angle the brain happened to be lying at -- and the
frame's three angles become how far the ATLAS has to be turned to meet them.
The sample itself is drawn in ONE pane, as the plane the microscope really
acquired, scrolled by a slider of its own that moves nothing else; the atlas
is drawn beside it, or superimposed on it in additive green and magenta, with
an in-plane offset and one scale to get the two brains the same size and in
the same place first. Nothing is registered, resampled or written: what you
take away is three angles, read off the Planes panel.

This used to be a second window paint_mask.py opened from its own ontology
picker, kept in sync with paint_mask's brush-label assignment. It is
standalone now: paint_mask.py's own ontology panel (for assigning tree nodes
to brush labels, feeding the guide-outline export) no longer opens or drives
any atlas display, and this viewer no longer knows paint_mask exists. Point
both at the same atlas_annotation_path / ontology_path if you want to browse
the atlas while painting -- just as two independent windows, not two views of
one state. The atlas loading + ontology math both tools share lives in
shared/atlas_reference.py; this file is what turns that data into a window.

Usage (needs a display; runs in the antsreg conda env -- same napari+PyQt5+
SimpleITK requirement as paint_mask.py): edit configs/atlas_view.yaml
(gitignored -- copy it from configs/atlas_view.example.yaml the first time),
then just run the file -- no command-line arguments.

    conda activate antsreg
    python tools/atlas_view.py
    python tools/atlas_view.py configs/atlas_view.devccf.yaml   # or point at another config

The plane geometry (frames, bounds, resampling, crosshairs) is separately
runnable with no display and no config, on purely synthetic data:

    python tools/atlas_view.py --selftest
"""

import argparse
from types import SimpleNamespace

import numpy as np

# Run as `python tools/<name>.py`, so sys.path[0] is tools/, not the repo
# root -- put the root on it before importing anything from shared/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import atlas_reference   # GUI-free atlas loading + ontology math
from shared import local_config      # configs/<tool>.yaml
from shared import ontology_tree_ui  # the shared Qt ontology tree widget
from shared import hover_bar         # the shared bottom 'region under cursor' bar

# napari/PyQt5 are imported lazily by _import_gui(), for the same reason
# paint_mask.py does it: --selftest (pure numpy, no window) should run
# without a display or even PyQt5 installed.
napari = QLabel = QPushButton = QVBoxLayout = QWidget = None
QAbstractItemView = QCheckBox = QLineEdit = QSplitter = QTreeWidget = Qt = None
QDoubleSpinBox = QGridLayout = QSlider = QFont = QFontMetrics = QSize = None
ViewerModel = QtViewer = None


def _import_gui():
    global napari, QLabel, QPushButton, QVBoxLayout, QWidget
    global QAbstractItemView, QCheckBox, QLineEdit, QSplitter, QTreeWidget, Qt
    global QDoubleSpinBox, QGridLayout, QSlider, QFont, QFontMetrics, QSize
    global ViewerModel, QtViewer
    import napari as _napari
    from napari.components import ViewerModel as _ViewerModel
    from napari.qt import QtViewer as _QtViewer
    from PyQt5.QtCore import QSize as _QSize, Qt as _Qt
    from PyQt5.QtGui import QFont as _QFont, QFontMetrics as _QFontMetrics
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
    QFont, QFontMetrics, QSize = _QFont, _QFontMetrics, _QSize
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
    cfg = local_config.load_config(
        "atlas_view", cli_path=cli_path,
        required=("atlas_annotation_path", "ontology_path"),
        legacy_paths=_LEGACY_CONFIG_PATHS)
    atlas_cfg = atlas_reference.atlas_reference_config(cfg)
    # Optional, and attached rather than passed separately: the sample is one
    # more thing this window can be pointed at, not a second configuration.
    atlas_cfg.sample = atlas_reference.sample_volume_config(cfg)
    return atlas_cfg


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

# The sample's own colour, for its slider and its half of the overlay. Not one
# of the three plane colours: the sample is not a plane of the atlas, and the
# one control in this window that moves it must not look like the three that
# move the atlas.
_SAMPLE_COLOUR = "#b9f27a"

# Layer attributes mirrored from the main pane onto the ortho panes. Only the
# main pane has a layer list, so these are the knobs a user can actually reach;
# everything else about the sub-panes is fixed at construction.
_MIRRORED_LAYER_ATTRS = ("visible", "opacity", "contrast_limits", "gamma", "colormap",
                         "blending")

# Starting height in px of the bottom dock holding the two reconstructed
# views, leaving the main canvas the rest. Draggable afterwards.
_ORTHO_DOCK_HEIGHT = 300

# Starting width in px of the left-hand ontology dock, and a cap on how wide
# its "Region" column opens. Both are STARTING points, not fixed sizes: every
# widget in that panel is allowed to shrink (see _shrinkable), so the dock
# edge can be dragged to any width afterwards. Without the cap the column
# grows to the longest region name in the atlas ("Anterior cingulate area,
# dorsal part, layer 6a" and friends), which is most of a screen wide.
_REGION_DOCK_WIDTH = 340
_REGION_NAME_COLUMN_WIDTH = 260

# How near a cut line a press has to land to grab it and start aiming that
# plane instead of panning the camera (see grabbed_cut_line). Screen pixels,
# converted through the pane's zoom, so it is the same target on screen at
# any magnification.
_LINE_GRAB_PX = 8

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
    two planes swing round it. That is the whole interaction -- the dragged
    pane's picture does not move at all (see plane_basis: its drawing axes do
    not turn with the frame either), its two cut lines swing across it, and
    the other two panes reslice to the angle those lines now describe.
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


def axes_from_frame(frame, order):
    """The 3x3 view matrix one pane starts from: its normal, then the axis
    drawn down the screen and the one drawn across it, straight out of
    `frame` in `order`.

    Only the STARTING point. From here each pane carries its own two drawing
    axes (plane_basis) instead of re-reading them off the frame, which is
    what keeps the sample still while the planes turn.
    """
    frame = np.asarray(frame, dtype=float)
    return np.array([frame[order[0]], frame[order[1]], frame[order[2]]])


def pane_handedness(axes):
    """+1 or -1: whether a pane's (normal, row, col) is right- or left-handed.

    Pane 1 draws axes (0, 2) down and across, which with axis 1 pointing at
    the viewer is a LEFT-handed triple, while panes 0 and 2 are right-handed.
    Without this sign a rotation drag would turn the cut lines the wrong way
    in exactly one of the three panes, and plane_basis would mirror its
    picture. It is a property of the pane, fixed once at construction: a
    rotation cannot change it.
    """
    axes = np.asarray(axes, dtype=float)
    return float(np.sign(np.dot(np.cross(axes[0], axes[1]), axes[2])))


def plane_basis(normal, row_hint, col_hint, handedness=1.0):
    """An orthonormal (row, col) pair spanning the plane with `normal`, kept
    as close to (row_hint, col_hint) as that plane allows.

    THIS IS WHAT HOLDS THE SAMPLE STILL. A pane's picture is sampled along
    its two drawing axes, and this viewer used to take them straight off the
    global frame -- so turning the frame about a pane's own normal spun that
    pane's picture on screen even though its plane had not moved at all. What
    the user sees then is the atlas rotating under a fixed cut, i.e. exactly
    backwards from "hold the sample still and tilt the plane".

    Carrying the pane's PREVIOUS axes across instead makes its picture
    invariant whenever its own plane is unchanged (the rotation the drag
    applies is about this pane's normal, which leaves both hints already in
    the plane, so they come back untouched), and moves it as little as the
    geometry allows when the plane really does tilt.

    Gram-Schmidt against the new normal, with the sign of the pair preserved
    (`handedness`) so a picture can never mirror, plus a fallback for the one
    degenerate case -- a plane tilted until the old row axis points along the
    new normal, where the old COLUMN axis is still a perfectly good hint.
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    row = np.asarray(row_hint, dtype=float)
    row = row - np.dot(row, normal) * normal
    if np.linalg.norm(row) < 1e-3:
        # cross(col, normal) is perpendicular to both, and with this sign the
        # column axis below comes back out as col_hint itself.
        row = handedness * np.cross(np.asarray(col_hint, dtype=float), normal)
    row = row / np.linalg.norm(row)
    col = handedness * np.cross(normal, row)
    return np.array([row, col])


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


def plane_slice(volumes, shape, origin, axes, centre, stride=1):
    """Everything one pane draws, for the current plane and crosshair.

    `axes` is the pane's 3x3 view matrix (axes_from_frame / plane_basis): row
    0 is the plane's normal, which comes from the global frame; rows 1 and 2
    are the pane's OWN drawing axes, down the screen and across it, which do
    not turn with the frame. At the identity frame that reproduces the plain
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
    normal, row_dir, col_dir = np.asarray(axes, dtype=float)
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


def plane_point(origin, axes, offset, s, t):
    """The voxel coordinate of in-plane point (s, t) on the plane `offset`
    away from `origin` along `axes`'s normal. Inverse of the projection
    plane_slice reports -- what a click in a pane means in 3D."""
    normal, row_dir, col_dir = np.asarray(axes, dtype=float)
    return (np.asarray(origin, dtype=float) + offset * normal
            + s * row_dir + t * col_dir)


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


def cut_directions(axes, frame, order):
    """The two in-plane directions, in this pane's own (s, t) coordinates,
    along which the other two planes cut it.

    Plane k meets this pane along cross(normal, frame[k]) -- the one
    direction perpendicular to both normals -- read off in the pane's drawing
    axes. Returned in the order (plane order[1], plane order[2]), which is
    what _PANE_COLOURS colours the two lines by.

    While the pane's axes came straight off the frame these were always
    (0, 1) and (1, 0), which is why the cross used to be drawn axis-aligned.
    Now that a pane holds its own axes (plane_basis) they are what MOVES when
    you drag: the picture stays, the lines swing.
    """
    normal, row_dir, col_dir = np.asarray(axes, dtype=float)
    frame = np.asarray(frame, dtype=float)
    directions = []
    for k in (order[1], order[2]):
        along = np.cross(normal, frame[k])
        length = np.linalg.norm(along)
        if length < 1e-12:              # a plane parallel to this one: no line
            directions.append((0.0, 0.0))
            continue
        along = along / length
        directions.append((float(np.dot(along, row_dir)), float(np.dot(along, col_dir))))
    return directions


def line_in_box(point, direction, lo, hi):
    """(a, b): the range of `a*direction` over which point + a*direction stays
    inside the box [lo, hi], or None if the line misses it.

    What keeps an obliquely drawn cut line exactly as long as the picture is
    wide. The lines are no longer parallel to the grid, so neither "the full
    width" nor "the full height" is their length any more, and a line simply
    made long enough to always cross would stick out past the image and drag
    napari's reset_view zoom out with it.
    """
    near, far = -np.inf, np.inf
    for value, step, low, high in zip(point, direction, lo, hi):
        if abs(step) < 1e-12:
            if value < low or value > high:
                return None             # parallel to this edge, and outside it
            continue
        first, second = (low - value) / step, (high - value) / step
        near = max(near, min(first, second))
        far = min(far, max(first, second))
    return (near, far) if far > near else None


def crosshair_vectors(sl, directions, shift=(0.0, 0.0)):
    """napari Vectors data (2, 2, 2) for one pane: the two lines where the
    other two planes cut through this one, clipped to the drawn grid.

    Two lines, not three: the third plane is this pane's own, and it meets the
    pane everywhere rather than along a line. The frame is rigid, so the two
    are always perpendicular to each other and always cross at the crosshair
    -- but they are no longer parallel to the pane's own axes. That is the
    point: turning the frame about this pane's normal leaves its picture
    exactly where it was and swings these two lines across it, so the lines
    ARE the handle for "how are the other two planes angled".

    `directions` is the pair from cut_directions, in the same order.
    """
    s_shift, t_shift = float(shift[0]), float(shift[1])
    centre = np.array([sl.s_centre + s_shift, sl.t_centre + t_shift])
    lo = np.array([sl.s_min + s_shift, sl.t_min + t_shift])
    hi = np.array([sl.s_max + s_shift, sl.t_max + t_shift])

    lines = []
    for direction in directions:
        direction = np.asarray(direction, dtype=float)
        length = np.linalg.norm(direction)
        span = None if length < 1e-12 else line_in_box(centre, direction / length, lo, hi)
        if span is None:
            lines.append([centre, np.zeros(2)])      # nothing to draw
            continue
        near, far = span
        direction = direction / length
        lines.append([centre + near * direction, (far - near) * direction])
    return np.array(lines, dtype=float)


# =====================================================================================
# the sample -- a second volume, never rotated, that the atlas is aimed at
# =====================================================================================
# Everything above describes ONE volume and three planes cut through it. With a
# sample loaded the reading changes: the three planes are the SAMPLE's own axes
# (the light-sheet frame, fixed, because that is how the stack came off the
# microscope), and the frame is how the atlas is turned to meet them. Plane k
# is perpendicular to sample axis k; state.frame[k] is the atlas-voxel
# direction that sample axis k corresponds to. So the same three Euler angles
# that used to say "reslice the atlas 12 degrees off coronal" now say "this
# sample was cut 12 degrees off the atlas's horizontal", which is the number a
# registration needs and the one a light sheet cannot be aimed accurately
# enough to assume is zero.
#
# The sample is shown in ONE pane and scrolled by ONE slider of its own
# (_SAMPLE_PANE_AXIS): the plane the microscope actually acquired, and nothing
# else. It is deliberately NOT tied to where the atlas's planes sit. Deriving
# the sample slice from the atlas plane offset -- which this did at first --
# means every tilt and every scroll of the atlas drags the sample picture
# around too, and then there is nothing fixed left on screen to judge the tilt
# against. Two independent sliders, one per volume, is the whole interaction:
# park the sample on a plane you know, and turn and scroll the atlas until it
# matches.
#
# What still ties the two together is only how the sample is DRAWN in that
# pane: a position and a size, so the two brains are comparable at all. In the
# pane's own (in-plane) axes, atlas world microns -> sample microns is
#
#     u_a = offset_um[a] + scale * w_a
#
# with u measured from the sample's own middle voxel along sample axis a, and
# w the atlas world coordinate the pane already draws in. The two functions
# below are that line read sideways (where does the sample slice belong on
# screen) and backwards (what offset and scale put the two brains on top of
# each other).

# Which of the sample's axes the acquisition planes lie across, i.e. which
# pane shows the sample. Axis 0 is the array's slowest axis, which for a stack
# read page by page (tifffile, and SimpleITK for anything else) is the sheet
# index -- the plane the microscope really did image, the only one of the
# three that is not an interpolation between sheets and the only one worth
# comparing an atlas against.
_SAMPLE_PANE_AXIS = 0


def sample_plane_image(volume, order, index):
    """(image, index): one acquisition plane of the sample, for its pane.

    A plain take() along the pane's normal, not a reslice: the sample is the
    fixed thing here and is never turned, resampled or interpolated -- it is
    shown exactly as acquired. The pane's other two axes are always ascending
    (see _ORTHO_PANES), so what take() leaves behind is already in the pane's
    own (down the screen, across it) order.

    The index is clipped and returned, so the caller can show what it actually
    landed on rather than what was asked for.
    """
    axis = int(order[0])
    index = int(np.clip(int(index), 0, volume.shape[axis] - 1))
    return np.take(volume, index, axis=axis), index


def sample_placement(order, sample_shape, sample_voxel, scale, offset_um):
    """(scale, translate) laying the sample plane into the pane's world.

    A pane's world coordinates are the ATLAS's microns (every atlas layer is
    drawn with scale=atlas voxel size), so the sample -- a different voxel
    size on a different grid -- is placed by inverting the atlas-to-sample map
    rather than by resampling it. Nothing about the sample image is ever
    interpolated: it is the reference.

    Two numbers per in-plane axis, both from the same line: a sample voxel is
    `sample_voxel / scale` atlas microns wide, and index 0 sits where the
    sample's own middle voxel, offset by offset_um, says it does.
    """
    origin = volume_origin(sample_shape)
    scales, translates = [], []
    for axis in (int(order[1]), int(order[2])):
        step = float(sample_voxel[axis]) / float(scale)
        scales.append(step)
        translates.append((-float(origin[axis]) * float(sample_voxel[axis])
                           - float(offset_um[axis])) / float(scale))
    return np.array(scales, dtype=float), np.array(translates, dtype=float)


def coarse_step(shape, target=128):
    """Subsampling step that gets `shape` down to about `target` voxels a side.

    For the auto-fit only, which asks a question about the outline of a brain
    -- np.argwhere over a full-resolution 800^3 volume would cost several
    times the volume itself to answer it no better.
    """
    return max(1, int(max(int(n) for n in shape) // int(target)))


def foreground_threshold(volume, step=1, fraction=0.05):
    """A rough tissue/background level for a grayscale sample.

    A twentieth of the way from the 1st to the 99th percentile. Percentiles
    because a light-sheet stack has hot specks that would drag min/max
    anywhere, and a low fraction because what the auto-fit wants is the
    OUTLINE of the brain, dim edges included, not a clean segmentation --
    on a real 20 um stack here the box stops growing somewhere below a
    twentieth and starts swallowing background haze somewhere below a
    fiftieth, so this sits in the middle of the flat part.
    """
    coarse = volume[::step, ::step, ::step]
    lo, hi = np.percentile(coarse, (1.0, 99.0))
    return float(lo) + float(fraction) * float(hi - lo)


def foreground_box(volume, threshold, step=1):
    """((centre, size) in voxels) of everything above `threshold`, or None."""
    coarse = volume[::step, ::step, ::step]
    found = np.argwhere(coarse > threshold)
    if found.size == 0:
        return None
    lo = found.min(axis=0) * float(step)
    hi = found.max(axis=0) * float(step)
    return (lo + hi) / 2.0, (hi - lo + 1.0)


def fit_sample_placement(atlas_box, sample_box, atlas_origin, atlas_voxel,
                         sample_shape, sample_voxel, order):
    """(offset_um, scale) putting the atlas's brain on top of the sample's, in
    the pane the sample is drawn in.

    Centre on centre, size to size, from the two bounding boxes -- a starting
    point for the eye, not a registration. It exists because the angle is the
    only thing here that is genuinely hard to guess: two brains half a brain
    apart and 20% apart in size cannot be compared for angle at all, and
    nudging three numbers by hand before you can start on the fourth is how a
    tool like this gets abandoned.

    IN-PLANE ONLY, both the offsets and the ratios the scale comes from. The
    third axis is the one the two sliders scroll independently, so it has
    nothing to place; and including its ratio in the size would be actively
    wrong here -- it is the axis most likely to be cropped differently in the
    two volumes (a light sheet stops where the stack stops), and one axis
    cropped short would shrink the atlas in the two axes you can actually see.

    ONE scale rather than one per axis: the remaining two ratios differ mostly
    because the boxes are cropped differently, not because the brains are, and
    letting each axis stretch on its own bakes that difference in as
    anisotropy. `offset_um` comes back as a full 3-vector with 0 in the
    normal's slot, since that is the shape everything downstream indexes.

    Both boxes are measured axis-aligned in their own grids while the atlas
    may be turned by a few degrees, which widens its box slightly -- true of
    any tilt, small at the angles this tool is for, and this is a starting
    point either way.
    """
    (atlas_centre, atlas_size), (sample_centre, sample_size) = atlas_box, sample_box
    atlas_origin = np.asarray(atlas_origin, dtype=float)
    sample_origin = volume_origin(sample_shape)
    in_plane = (int(order[1]), int(order[2]))

    ratios = [(float(sample_size[a]) * float(sample_voxel[a]))
              / (float(atlas_size[a]) * float(atlas_voxel)) for a in in_plane]
    scale = float(np.exp(np.mean(np.log(ratios))))

    offset_um = np.zeros(3)
    for axis in in_plane:
        atlas_reach = (float(atlas_centre[axis]) - atlas_origin[axis]) * float(atlas_voxel)
        offset_um[axis] = ((float(sample_centre[axis]) - sample_origin[axis])
                           * float(sample_voxel[axis]) - atlas_reach * scale)
    return offset_um, scale


# =====================================================================================
# the window
# =====================================================================================
def _add_atlas_layers(model, atlas, voxel, features, order, first, cross):
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
    # One colour per LINE. A pane draws the crosshair twice when the sample is
    # beside the atlas (see pane_cross), so the colours are repeated to match:
    # napari wants exactly as many as there are vectors, and the count is fixed
    # for the life of the layer -- which is why the second pair is drawn as
    # zero-length vectors when there is nothing beside the atlas, rather than
    # dropped.
    colours = [_PANE_COLOURS[order[1]], _PANE_COLOURS[order[2]]] * (len(cross) // 2)
    cross = model.add_vectors(cross, name="crosshair", edge_color=colours,
                              edge_width=1.5, vector_style="line", opacity=0.9, scale=scale)
    return SimpleNamespace(template=template, annotation=annotation,
                           highlight=highlight, cross=cross)


def _open_atlas_window(atlas, resolution_um, ortho=True, sample=None):
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
    planes on the clicked point, and dragging one of the two coloured cut
    lines (or Shift+left-dragging anywhere in the pane) swings that cut round,
    reslicing the other two panes live to the angle it now describes. The
    dragged pane's OWN picture does not move while you do it -- see
    plane_basis: each pane draws on axes it carries itself rather than on the
    frame's other two rows, so the gesture looks like "the sample holds still
    and the plane turns" instead of the atlas spinning under a fixed cut.
    Both go through one piece of state (`state.frame`, `state.centre`), so the
    sliders, the angle boxes, clicks, drags and "jump to region" cannot
    disagree about where the planes are.

    A SLIDER UNDER EVERY PANE (_add_pane_slider) scrolls that pane's plane
    along its own normal, so the view you are looking at is the one you
    scroll; the plane panel's three sliders hold the same three numbers. With
    a sample loaded, its pane carries a second slider (_add_sample_slider) for
    the sample's own stack -- two volumes, two sliders, no link between them.

    A HOVER BAR (_add_hover_bar) along the bottom, because the one structure
    a voxel is labelled with is usually a leaf ("layer 5 of primary motor
    area") and the level you are actually picking is one of its ancestors: it
    reads out the deepest few levels of that chain in large type, painted in
    the region's own annotation colour. A PLANE PANEL (_add_plane_panel) on the
    right holds the frame itself: one position slider per plane and one angle
    box per data axis, for the times you want an exact 12 degrees rather than
    a dragged one (and the panes' own sliders for the times you do not). A separate
    REGION-SELECTION panel (_add_region_panel) on the left is what drives the
    tree click -> highlight path -- see that function for why it gets its own
    dedicated side.

    A SAMPLE VOLUME (`sample`, optional -- see _add_sample_panel) turns the
    same window into a comparison. The three planes then stand for the
    SAMPLE's own axes, which is what a light sheet actually fixes: the stack
    was cut at whatever angle the brain happened to be lying at, and the
    question is what angle the atlas has to be turned through to match it.
    ONE pane (_SAMPLE_PANE_AXIS) gains one more layer at the bottom of its
    stack -- the sample's own acquisition plane, drawn beside that pane's
    atlas reslice or superimposed on it, and scrolled by a second slider under
    that canvas which moves the sample and nothing else. The panes' drawing
    axes also switch from "carried across" to "the frame's own" (see
    set_frame), so a rotation turns the atlas under a sample that stays
    exactly where it is. Without a sample nothing about the window changes.

    The atlas grid here is INDEPENDENT of the sample grid -- nothing in this
    window is registered to anything, and the sample-to-atlas map is only ever
    the rigid rotation + offset + single scale the panel holds. What the
    atlas's own extent, orientation and downsampling affect is exactly this
    display, nothing else.
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
                            listeners=[],
                            # The atlas is the thing that moves as soon as
                            # there is a sample to move it against, and the
                            # panes' axes have to follow the frame for that to
                            # be visible at all -- see set_frame.
                            sample=None,
                            lock_axes=sample is not None)
    if sample is not None:
        state.sample = SimpleNamespace(
            volume=sample.volume,
            voxel=np.asarray(sample.voxel_um, dtype=float),
            path=sample.path,
            contrast=_sample_contrast(sample.volume),
            # Which acquisition plane is on screen. Its own number, moved by
            # its own slider and by nothing else -- see _SAMPLE_PANE_AXIS on
            # why it is not derived from where the atlas's planes sit.
            index=int(volume_origin(sample.volume.shape)[_SAMPLE_PANE_AXIS]),
            # Where the atlas's middle voxel sits relative to the sample's,
            # in microns along the two IN-PLANE sample axes (the third entry
            # is unused -- that axis has two independent sliders instead), and
            # how much bigger the atlas has to be drawn to match the sample.
            # Both start neutral; the panel's auto-fit is what sets them.
            offset_um=np.zeros(3),
            scale=1.0,
            side_by_side=True)

    def volumes():
        """The volumes every pane resamples, in the order _add_atlas_layers
        and refresh() both read them back out."""
        vols = [atlas.compact, state.highlight]
        if atlas.template is not None:
            vols.append(atlas.template)
        return vols

    def slice_for(pane):
        return plane_slice(volumes(), shape, origin, pane.axes, state.centre,
                           stride=state.stride)

    def new_pane(model, qt, order):
        """One pane's own state: which plane it draws, the two axes it draws
        that plane on (its own, see plane_basis), its handedness (fixed) and
        the screen shift that keeps its crosshair still."""
        axes = axes_from_frame(state.frame, order)
        return SimpleNamespace(model=model, qt=qt, order=order, shift=np.zeros(2),
                               axes=axes, handedness=pane_handedness(axes), column=None)

    def pane_t_bounds(pane):
        """(lo, hi) of this pane's atlas picture across the screen, in voxels.

        The same bounds plane_slice sizes its grid from -- read separately
        here because the sample's placement and a click's meaning both need
        to know where the atlas picture ends, and neither has a slice in hand.
        """
        return plane_bounds(shape, origin, pane.axes[2])

    def atlas_width(pane):
        """How wide this pane's atlas picture is, in world microns."""
        t_min, t_max = pane_t_bounds(pane)
        return (t_max - t_min + 1) * voxel

    def sample_gap(pane):
        """How far to the side of the atlas reslice this pane draws the sample.

        Half of each picture's width, so the two sit next to each other
        without touching whichever of them is wider -- and zero when the mode
        is overlay, which is the whole of what "overlay" means here: the same
        two layers, drawn in the same place instead of beside each other.
        """
        smp = state.sample
        # By pane order, not by layer: this is called while the pane's layers
        # are still being built (pane_cross feeds _add_atlas_layers).
        if smp is None or not smp.side_by_side or int(pane.order[0]) != _SAMPLE_PANE_AXIS:
            return 0.0
        axis = int(pane.order[2])
        sample_width = smp.volume.shape[axis] * smp.voxel[axis] / smp.scale
        return 0.53 * (atlas_width(pane) + sample_width)

    def sample_slice_for(pane):
        """The sample plane its slider is currently parked on."""
        smp = state.sample
        image, smp.index = sample_plane_image(smp.volume, pane.order, smp.index)
        return image

    def add_sample_layer(model, pane):
        """The sample's own layer, in the ONE pane that shows it
        (_SAMPLE_PANE_AXIS), added BEFORE the atlas's so it sits at the bottom
        of the stack -- in overlay mode the atlas is what is drawn over the
        sample, not the other way round."""
        if state.sample is None or int(pane.order[0]) != _SAMPLE_PANE_AXIS:
            return None
        return model.add_image(sample_slice_for(pane), name="sample (fixed, never rotated)",
                               colormap="gray", contrast_limits=list(state.sample.contrast))

    def place_sample(pane):
        """Draw this pane's sample plane: which slice, how big, and where."""
        smp = state.sample
        layer = pane.layers.sample if smp is not None else None
        if layer is None:
            return
        layer.data = sample_slice_for(pane)
        scale, translate = sample_placement(pane.order, smp.volume.shape, smp.voxel,
                                            smp.scale, smp.offset_um)
        # Only when it actually changed: see _add_atlas_layers on why writing a
        # layer's scale on every mouse move is worth avoiding. Here it depends
        # on nothing but the size ratio, which a drag never touches.
        if not np.allclose(np.asarray(layer.scale, dtype=float), scale):
            layer.scale = scale
        layer.translate = translate + pane.shift * voxel + np.array([0.0, sample_gap(pane)])

    def pane_cross(pane, sl):
        """The two cut lines -- twice over when the sample is drawn beside the
        atlas, so the pair over there marks the same point in the sample as
        the pair over here does in the atlas. (Clipped to the atlas picture's
        width in both places: they are a position marker, not a measurement of
        the sample's extent.)"""
        directions = cut_directions(pane.axes, state.frame, pane.order)
        lines = crosshair_vectors(sl, directions, pane.shift)
        if state.sample is None or int(pane.order[0]) != _SAMPLE_PANE_AXIS:
            return lines
        gap = sample_gap(pane)
        beside = (crosshair_vectors(sl, directions, (pane.shift[0], pane.shift[1] + gap / voxel))
                  if gap else np.zeros_like(lines))
        return np.concatenate([lines, beside])

    viewer = napari.Viewer(title="Atlas viewer")
    # napari's own layer controls stop shrinking while they still own a good
    # part of the left column -- with the region panel sharing that column,
    # the space cannot be traded back. See free_layer_controls_height.
    ontology_tree_ui.free_layer_controls_height(viewer)
    main = new_pane(viewer, None, panes[0][0])
    main.column = _main_pane_column(viewer)
    first = slice_for(main)
    sample_layer = add_sample_layer(viewer, main)
    main.layers = _add_atlas_layers(viewer, atlas, voxel, features, main.order,
                                    first, pane_cross(main, first))
    main.layers.sample = sample_layer
    place_sample(main)
    built = [main]

    for order, _title in panes[1:]:
        model = ViewerModel(ndisplay=2)
        # Canvas BEFORE layers: napari 0.8.0's QtViewer.__init__ walks the
        # layers already in the model and reorders them against a visual map
        # it has not filled in yet, which raises KeyError on the second layer.
        # Adding to an empty model routes through the same code path one layer
        # at a time, where the map is always current.
        qt = QtViewer(model)
        pane = new_pane(model, qt, order)
        first = slice_for(pane)
        sample_layer = add_sample_layer(model, pane)
        pane.layers = _add_atlas_layers(model, atlas, voxel, features, order,
                                        first, pane_cross(pane, first))
        pane.layers.sample = sample_layer
        place_sample(pane)
        built.append(pane)
        model.reset_view()

    # The one pane the sample is drawn in, or None with no sample loaded.
    sample_pane = next((pane for pane in built
                        if pane.layers.sample is not None), None)

    ortho_dock = None
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
            pane.column = layout          # the pane's own slider goes here
            # A bare QtViewer asks for 800x626 and Qt would honour both of
            # them: two side by side in a bottom dock open a ~1600px-wide
            # window whose main canvas is a letterbox. A small floor plus the
            # explicit resizeDocks below makes the ortho row a strip that the
            # user can then drag to whatever they actually want.
            pane.qt.setMinimumSize(160, 120)
            splitter.addWidget(wrapper)
        ortho_dock = viewer.window.add_dock_widget(splitter, area="bottom",
                                                   name="Ortho views (synced)")
        viewer.window._qt_window.resizeDocks([ortho_dock], [_ORTHO_DOCK_HEIGHT], Qt.Vertical)
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
            pane.layers.cross.data = pane_cross(pane, sl)
            place_sample(pane)

    def apply_state():
        refresh()
        for listener in state.listeners:
            listener()

    def crosshair_in_pane(pane):
        """The crosshair's position in `pane`'s own drawn coordinates, shift
        included -- i.e. where on screen the cross is."""
        rel = state.centre - origin
        return np.array([np.dot(rel, pane.axes[1]) + pane.shift[0],
                         np.dot(rel, pane.axes[2]) + pane.shift[1]])

    def set_frame(frame):
        """Turn the planes, pivoting each pane's picture on ITS CROSSHAIR.

        In-plane coordinates are measured from the volume's middle voxel, so
        left alone a tilt would swing each reslice about the middle of the
        atlas -- away from whatever the user centred on before reaching for
        the rotation, which is invariably the thing they are trying to line
        up. Each pane keeps a `shift` that absorbs the difference, so the
        crosshair stays put on screen and the new cut appears around it, while
        clicks (which do not touch the frame) still leave the picture alone.

        Each pane's DRAWING AXES are carried across here too, from its old
        plane to its new one, rather than re-read off the frame (plane_basis).
        A pane whose own plane did not move -- the one being dragged, whose
        normal is the rotation axis -- therefore keeps its axes exactly, and
        its picture does not budge: what turns is the pair of cut lines drawn
        over it, and the other two panes' pictures.

        UNLESS there is a sample (`state.lock_axes`), where that is exactly
        backwards. The sample is then the thing that holds still -- it is the
        stack as acquired, the fixed frame the panes stand for -- and the atlas
        is what the user is turning. Carrying the axes across would hide the
        one component of the rotation that matters most for lining two brains
        up: the in-plane one, which would leave the dragged pane's atlas
        picture untouched while the sample beside it stayed put too, i.e.
        looking like nothing happened. Read off the frame instead, and the
        atlas turns under a stationary sample in every pane.
        """
        before = [crosshair_in_pane(pane) for pane in built]
        state.frame = orthonormal_frame(frame)
        for pane, was in zip(built, before):
            normal = state.frame[pane.order[0]]
            if state.lock_axes:
                pane.axes = axes_from_frame(state.frame, pane.order)
            else:
                pane.axes = np.array([normal, *plane_basis(normal, pane.axes[1], pane.axes[2],
                                                           pane.handedness)])
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
            pane.axes = axes_from_frame(state.frame, pane.order)
        apply_state()
        for pane in built:
            pane.model.reset_view()

    def set_sample(index=None, offset_um=None, scale=None, side_by_side=None):
        """Everything about the sample: which plane of it is on screen, and
        how that plane is drawn against the atlas.

        `index` is the sample's own slider and moves nothing else -- the atlas
        stays exactly where it is. Rotation is not here at all: that is the
        frame, the same one the panes, the plane sliders and the drags have
        always shared.
        """
        smp = state.sample
        if smp is None:
            return
        if index is not None:
            smp.index = int(np.clip(int(index), 0,
                                    smp.volume.shape[_SAMPLE_PANE_AXIS] - 1))
        if offset_um is not None:
            smp.offset_um = np.asarray(offset_um, dtype=float)
        if scale is not None:
            smp.scale = max(float(scale), 1e-3)
        if side_by_side is not None:
            smp.side_by_side = bool(side_by_side)
        apply_state()

    def fit_sample():
        """Auto-fit: the two brains the same size, in the same place, and both
        sliders parked in the middle of their own tissue.

        The atlas's own foreground is exact and free -- every voxel the
        annotation labels -- while the sample's is a threshold guess
        (foreground_threshold), which is why this only seeds numbers the user
        then adjusts. The two SLICE positions are seeded too but stay
        independent afterwards: each volume's slider is parked at the middle
        of its own bounding box, which is a comparable starting pair without
        tying one to the other. Returns None if either volume turns out to be
        empty at that threshold.
        """
        smp = state.sample
        if smp is None:
            return None
        atlas_box = foreground_box(atlas.compact, 0, coarse_step(shape))
        step = coarse_step(smp.volume.shape)
        sample_box = foreground_box(smp.volume, foreground_threshold(smp.volume, step), step)
        if atlas_box is None or sample_box is None:
            return None
        offset_um, scale = fit_sample_placement(atlas_box, sample_box, origin, voxel,
                                                smp.volume.shape, smp.voxel, sample_pane.order)
        set_sample(offset_um=offset_um, scale=scale,
                   index=int(round(float(sample_box[0][_SAMPLE_PANE_AXIS]))))
        centre_on(atlas_box[0])
        return offset_um, scale, smp.index

    def reset_views():
        """Re-fit every pane's camera to what it is now drawing -- what makes
        switching between side by side and overlay land on the pictures
        instead of on the empty half of where they used to be."""
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
        """A napari world position in `pane` -> the voxel it points at.

        A click on the SAMPLE, while the sample is drawn beside the atlas,
        means the same as a click on the atlas at the matching point: the two
        are one picture with a gap down the middle, so the gap comes off
        first. Without it, clicking the very thing you are comparing against
        would send the crosshair a brain's width outside the atlas.
        """
        s = float(world[0]) / voxel - pane.shift[0]
        t = float(world[1]) / voxel - pane.shift[1]
        gap = sample_gap(pane)
        # Everything past the atlas picture's own right edge is the sample:
        # sample_gap leaves a margin between the two, so nothing in between is
        # ambiguous.
        if gap and t > pane_t_bounds(pane)[1]:
            t -= gap / voxel
        offset = float(np.dot(state.centre - origin, pane.axes[0]))
        return plane_point(origin, pane.axes, offset, s, t)

    win = SimpleNamespace(viewer=viewer, panes=built, state=state, origin=origin,
                          voxel=voxel, shape=shape,
                          set_highlight=set_highlight, centre_on=centre_on,
                          set_offsets=set_offsets, set_euler=set_euler,
                          reset_planes=reset_planes, rotate=rotate, set_drag=set_drag,
                          set_sample=set_sample, fit_sample=fit_sample, reset_views=reset_views,
                          sample_pane=sample_pane,
                          crosshair_in_pane=crosshair_in_pane, pane_position=pane_position)

    for pane in built:
        pane.model.mouse_drag_callbacks.append(_pane_mouse_callback(pane, win))
        if pane.column is not None:
            pane.slider = _add_pane_slider(pane, win, pane.column)
    if sample_pane is not None and sample_pane.column is not None:
        # Under that pane's atlas slider: one volume per slider, stacked in
        # the order they are drawn in.
        win.sample_slider = _add_sample_slider(win, sample_pane.column)

    win.hover = _add_hover_bar(viewer, atlas, built, below=ortho_dock)
    planes_panel = _add_plane_panel(viewer, win)
    if state.sample is not None:
        sample_panel = _add_sample_panel(viewer, win)
        # Tabbed rather than stacked, and the sample panel in front: two panels
        # sharing one column arrive as a pair of slivers otherwise, and with a
        # sample loaded the sample panel is where the session starts.
        ontology_tree_ui.tabify(viewer, [planes_panel.dock, sample_panel.dock],
                                current=sample_panel.dock)
    # After the panels, not before: the sample panel's opening auto-fit resizes
    # the atlas, and a camera fitted to the un-fitted picture would open zoomed
    # to nothing much.
    reset_views()
    return win


def _pane_mouse_callback(pane, win):
    """Left-click anywhere in a pane -> crosshair there, in every pane;
    left-drag ON one of the two cut lines (or Shift+left-drag anywhere) ->
    swing those lines, i.e. tilt the other two planes.

    The sample never moves. Dragging in a pane does not turn its picture --
    that pane's plane is the rotation axis and its drawing axes are its own
    (plane_basis) -- it turns the two coloured lines lying across it, which
    are where the other two planes cut through. Those panes then reslice to
    the new angle. So the gesture reads as "aim this cut", not "spin the
    atlas".

    Grabbing a line needs no modifier, because that is the natural handle and
    a line is a small target; Shift+drag does the same thing from anywhere in
    the pane, for when the lines are off screen or awkward to hit. Every
    other unmodified drag is still napari's camera pan, and a press that
    never moves is still a click that recentres -- including one that landed
    on a line.

    A generator callback, which is napari's way of telling a click from the
    start of a drag: napari resumes it on every mouse_move and once more on
    release, so the decision can be made after the fact. A plain press handler
    cannot -- the same press that begins a camera pan would jump the crosshair
    on its way, and the atlas would slide out from under the pan.
    """
    def callback(_model, event):
        if event.button != 1:
            return                          # right-click menus stay napari's
        if "Shift" in event.modifiers:
            yield from _rotation_drag(pane, win, event)
            return
        if event.modifiers:
            return                          # other modified drags stay napari's
        if grabbed_cut_line(pane, win, event.position) is not None:
            # A press on a line: a drag aims it, a click still recentres.
            if not (yield from _rotation_drag(pane, win, event)):
                win.centre_on(win.pane_position(pane, event.position))
            return
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


def point_line_distance(point, centre, direction):
    """Perpendicular distance from `point` to the infinite line through
    `centre` along `direction` (2D). Zero-length directions read as
    infinitely far, so a pane parallel to another cannot be grabbed."""
    direction = np.asarray(direction, dtype=float)
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return np.inf
    direction = direction / length
    rel = np.asarray(point, dtype=float) - np.asarray(centre, dtype=float)
    return float(abs(rel[0] * direction[1] - rel[1] * direction[0]))


def grabbed_cut_line(pane, win, world_position, tolerance_px=None):
    """Which cut line (0, 1 or None) a press at `world_position` grabs.

    The tolerance is a screen distance, converted through the pane's zoom, so
    a line is equally easy to grab however far in or out the view is zoomed.
    """
    tolerance_px = _LINE_GRAB_PX if tolerance_px is None else tolerance_px
    zoom = float(getattr(pane.model.camera, "zoom", 1.0) or 1.0)
    tolerance = tolerance_px / zoom / win.voxel
    point = np.asarray(world_position, dtype=float)[:2] / win.voxel
    centre = win.crosshair_in_pane(pane)
    directions = cut_directions(pane.axes, win.state.frame, pane.order)
    distances = [point_line_distance(point, centre, direction) for direction in directions]
    nearest = int(np.argmin(distances))
    return nearest if distances[nearest] <= tolerance else None


def _rotation_drag(pane, win, event):
    """The angle the cursor sweeps around the crosshair becomes the angle the
    two cut lines swing through -- and with them the two planes they belong
    to, about this pane's own normal.

    Measured around the crosshair because that is where the lines cross and
    what set_frame pins the picture to. The sign is the pane's handedness
    NEGATED: rotating the frame by +a about a right-handed pane's normal
    carries its in-plane vectors -- and so the cut lines drawn along them --
    through -a on screen, so following the cursor means turning the frame the
    other way. (While the drag turned the picture instead of the lines, the
    sign was the other one, which is exactly the interaction this replaced.)

    Returns True if it actually rotated anything, so the caller can treat a
    press that never moved as a plain click.
    """
    axis = pane.order[0]
    # Two opposite gestures, because with a sample loaded the two volumes swap
    # roles. Atlas alone: the pane's picture is pinned (plane_basis) and what
    # follows the cursor are the two cut lines, which turn against the frame.
    # With a sample: the SAMPLE is what is pinned, the pane's axes are the
    # frame's own (see set_frame), and what follows the cursor is the atlas
    # picture itself, which turns with the frame. Same drag, same sign of
    # rotation on screen, opposite sign on the frame.
    sign = pane.handedness if win.state.lock_axes else -pane.handedness

    def cursor_angle():
        rel = np.asarray(event.position, dtype=float)[:2] / win.voxel - win.crosshair_in_pane(pane)
        if np.hypot(*rel) < 1.0:
            return None                     # on top of the pivot: no meaningful angle
        return float(np.arctan2(rel[0], rel[1]))   # s down, t across: screen angle

    last = cursor_angle()
    turned = False
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
                if delta:
                    win.rotate(axis, sign * delta)
                    turned = True
            if now is not None:
                last = now
            yield
    finally:
        camera.mouse_pan = True
        win.set_drag(False)
    return turned


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
    for role in ("sample", "template", "annotation", "highlight", "cross"):
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


def _shrinkable(widget):
    """Let `widget` be squeezed to nothing by the dock it lives in.

    Qt gives a layout the larger of a widget's explicit minimum and its
    minimumSizeHint, and for a plain QLabel/QCheckBox/QPushButton that hint
    is THE WHOLE OF ITS TEXT on one line. One long caption in a panel is
    therefore a hard floor on the dock's width -- and on the main window's,
    since a dock cannot be dragged below its contents' minimum. That is what
    made the ontology dock open enormous and refuse to be narrowed: nothing
    in it was allowed to be narrow.

    Setting an explicit minimum overrides the hint, so the caption clips (or,
    for a wrapped label inside ontology_tree_ui.scrollable, scrolls) instead
    of dictating the layout, and the dock edge becomes draggable to any
    width.
    """
    widget.setMinimumWidth(1)
    return widget


def _caption(text, height=52):
    """A panel's explanatory paragraph: word-wrapped, scrollable, and free to
    be as narrow as the dock is. See ontology_tree_ui.scrollable for why a
    bare wrapped label cannot be used -- its height would then grow every
    time the dock is narrowed, and drag the window's minimum size with it."""
    return _shrinkable(ontology_tree_ui.scrollable(QLabel(text), height))


def _add_pane_slider(pane, win, layout):
    """One slider per view, directly under that view's own canvas: where this
    pane's plane sits along its own normal.

    The plane panel on the right holds the same three numbers, but reaching
    across the window to scroll the view you are looking at is precisely the
    interaction this tool exists for -- so each pane gets its own. Both write
    the same state (win.set_offsets) and both follow it (state.listeners), so
    they cannot disagree about where a plane is; the slider is coloured like
    the pane's heading and its cut lines, so which plane it scrolls is
    readable without a label.

    `layout` is the box the slider is appended to -- the wrapper around an
    ortho pane, or the main viewer's own canvas column.
    """
    axis = pane.order[0]
    slider = _shrinkable(QSlider(Qt.Horizontal))
    slider.setToolTip(f"Scroll the ATLAS's plane {axis} along its own normal")
    colour = _PANE_COLOURS[axis]
    slider.setStyleSheet(f"QSlider::handle:horizontal {{ background: {colour}; "
                         f"border: 1px solid {colour}; width: 10px; "
                         f"margin: -4px 0; border-radius: 3px; }}"
                         "QSlider::groove:horizontal { height: 3px; background: #555; }")
    busy = {"in": False}

    def sync():
        """Follow the state, whatever moved it -- a click in another pane, a
        rotation, the panel's own slider. `busy` stops the write-back loop
        that would otherwise start here (see the plane panel's sync)."""
        offsets = plane_offsets(win.state.centre, win.state.frame, win.origin)
        lo, hi = plane_bounds(win.shape, win.origin, win.state.frame[axis])
        busy["in"] = True
        try:
            slider.setRange(lo, hi)
            slider.setValue(int(round(offsets[axis])))
        finally:
            busy["in"] = False

    def on_move(*_args):
        if busy["in"]:
            return
        # Only THIS plane's offset changes; the other two keep the values the
        # crosshair already has, so scrolling one view leaves the other two
        # cuts where they were.
        offsets = list(plane_offsets(win.state.centre, win.state.frame, win.origin))
        offsets[axis] = float(slider.value())
        win.set_offsets(offsets)

    slider.valueChanged.connect(on_move)
    slider.sliderPressed.connect(lambda: win.set_drag(True))
    slider.sliderReleased.connect(lambda: win.set_drag(False))
    win.state.listeners.append(sync)
    sync()
    layout.addWidget(slider)
    return slider


def _add_sample_slider(win, layout):
    """The sample's own slider: which acquisition plane is on screen.

    Deliberately a SECOND slider under the same canvas as that pane's atlas
    slider, in the sample's own colour rather than a plane colour. The two
    volumes are scrolled independently -- that is the whole layout (see
    _SAMPLE_PANE_AXIS) -- and two sliders stacked under one canvas, one per
    volume, is the shortest way to say so: park the sample on a plane you
    know, then scroll and turn the atlas until it matches.

    `layout` is the box to append to, so the same widget can be built under
    the canvas and again in the panel; both write win.set_sample(index=...)
    and both follow the state, so they cannot disagree.
    """
    smp = win.state.sample
    slider = _shrinkable(QSlider(Qt.Horizontal))
    slider.setToolTip("Scroll the SAMPLE through its own acquisition planes "
                      "(the atlas stays where it is)")
    slider.setStyleSheet(f"QSlider::handle:horizontal {{ background: {_SAMPLE_COLOUR}; "
                         f"border: 1px solid {_SAMPLE_COLOUR}; width: 10px; "
                         f"margin: -4px 0; border-radius: 3px; }}"
                         "QSlider::groove:horizontal { height: 3px; background: #555; }")
    slider.setRange(0, int(smp.volume.shape[_SAMPLE_PANE_AXIS]) - 1)
    busy = {"in": False}

    def sync():
        busy["in"] = True
        try:
            slider.setValue(int(smp.index))
        finally:
            busy["in"] = False

    def on_move(*_args):
        if busy["in"]:
            return
        win.set_sample(index=slider.value())

    slider.valueChanged.connect(on_move)
    slider.sliderPressed.connect(lambda: win.set_drag(True))
    slider.sliderReleased.connect(lambda: win.set_drag(False))
    win.state.listeners.append(sync)
    sync()
    layout.addWidget(slider)
    return slider


def _main_pane_column(viewer):
    """The layout the main viewer's canvas lives in, so a pane slider can be
    put under it exactly like the ortho panes' own.

    napari's QtViewer is a splitter whose first widget is a column holding the
    canvas and the dims sliders; appending to that column puts the slider
    under the canvas. Private API, hence the guard: if a napari release moves
    it, the main pane simply goes without its own slider (the plane panel
    still has all three) rather than the window failing to open.
    """
    try:
        return viewer.window._qt_viewer.canvas.native.parentWidget().layout()
    except Exception:
        return None


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
        readout.setMinimumWidth(60)
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
    reset = _shrinkable(QPushButton("Reset planes (axis-aligned, centred)"))

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

    atlas_only = (
        "Three orthogonal planes, tiltable together. The atlas itself never "
        "moves: click a view to point all three planes at that voxel, and "
        "drag one of the coloured lines lying across a view (or Shift+drag "
        "anywhere in it) to swing that cut round -- the other two views "
        "reslice to the angle you aim. Each view also has its own slider "
        "underneath for scrolling that plane along its normal.")
    with_sample = (
        "The three planes are the SAMPLE's own axes, and these angles are how "
        "far the atlas is turned to meet them -- what you write down as \"this "
        "brain was cut N degrees off the atlas's horizontal\". Type an exact "
        "angle here, or Shift+drag inside a view (or drag one of its coloured "
        "lines) to turn the atlas about that view's own normal: the sample "
        "stays put and the atlas swings under it. Everything here moves the "
        "ATLAS only -- these three angles, the three position sliders and the "
        "one under each view. The sample has its own slider, in the Sample "
        "panel and under the view it is drawn in.")

    dock = QWidget()
    dock.setMinimumWidth(1)
    layout = QVBoxLayout(dock)
    layout.addWidget(_caption(with_sample if win.state.sample is not None else atlas_only, 96))
    layout.addWidget(QLabel("Position along each plane's normal"))
    layout.addLayout(positions)
    layout.addWidget(QLabel("Orientation (rotations about the atlas axes)"))
    layout.addLayout(angles)
    layout.addWidget(ontology_tree_ui.scrollable(status, 56))
    layout.addWidget(reset)
    dock_widget = viewer.window.add_dock_widget(dock, area="right", name="Planes")

    state.listeners.append(sync)
    sync()
    return SimpleNamespace(sync=sync, sliders=sliders, angle_boxes=angle_boxes,
                           dock=dock_widget)


def _sample_contrast(volume):
    """Contrast limits for the sample layer, from the WHOLE volume.

    Same reason the atlas template takes its limits from the volume rather
    than from one plane (see _add_atlas_layers), plus one of its own:
    percentiles instead of min/max, because a light-sheet stack reliably
    carries a few saturated specks and one of them is enough to leave the
    brain looking black.
    """
    coarse = volume[::4, ::4, ::4]
    lo, hi = np.percentile(coarse, (0.5, 99.5))
    lo, hi = float(lo), float(hi)
    return lo, (hi if hi > lo else lo + 1.0)


def _add_sample_panel(viewer, win):
    """The sample: which plane of it is on screen, and how it is drawn against
    the atlas.

    The angle is what this panel exists to let you find -- a light sheet
    cannot be aimed to the degree, so a stack is cut at whatever angle the
    brain was lying at, and an atlas registered as though that angle were zero
    pulls structures across boundaries no amount of deformable registration
    afterwards puts back. It is also the one number that is hopeless to guess
    and easy to SEE, provided the two brains are first the same size and in
    the same place.

    So: the sample shows ONE plane -- the one the microscope acquired
    (_SAMPLE_PANE_AXIS) -- scrolled by ONE slider of its own that moves
    nothing else. Everything else in the window belongs to the atlas: its
    three planes, their angles, their positions. You park the sample on a
    plane you can read, and then turn and scroll the atlas until it matches.

    The two numbers that are not the angle are in-plane only (the third axis
    has two independent sliders instead): where the atlas sits relative to the
    sample, and how big it is drawn. An auto-fit sets both from the two
    bounding boxes.

    SIDE BY SIDE is for reading the sample -- the two pictures never touch, so
    an oblique olfactory bulb or a midbrain riding up into cortex is visible
    as itself rather than as a colour clash. OVERLAY is for judging: the same
    two layers drawn in the same place, in additive green and magenta, where
    what you are looking for is white.

    What the panel cannot do is register anything. Rotation, an in-plane
    offset and a single isotropic scale is the whole model; there is no shear,
    no per-axis stretch and no deformation, on purpose -- the output is three
    angles to reproduce a cut with, not a transform to resample with.
    """
    state, voxel = win.state, win.voxel
    smp = state.sample
    order = win.sample_pane.order
    in_plane = (int(order[1]), int(order[2]))
    busy = {"in": False}

    overlay = _shrinkable(QPushButton("Overlay the two (superimpose)"))
    overlay.setCheckable(True)

    plane_row = QGridLayout()
    plane_label = QLabel()
    swatch = QLabel()
    swatch.setFixedWidth(10)
    swatch.setStyleSheet(f"background: {_SAMPLE_COLOUR};")
    plane_row.addWidget(swatch, 0, 0)
    plane_row.addWidget(_shrinkable(plane_label), 0, 1)

    offsets = QGridLayout()
    offset_boxes = {}
    for row, axis in enumerate(in_plane):
        box = QDoubleSpinBox()
        box.setRange(-100000.0, 100000.0)
        box.setDecimals(0)
        box.setSingleStep(round(float(smp.voxel[axis]) * 5) or 10.0)
        box.setSuffix(" um")
        box.setKeyboardTracking(False)      # see the plane panel's angle boxes
        offsets.addWidget(QLabel(f"along axis {axis}"), row, 0)
        offsets.addWidget(box, row, 1)
        offset_boxes[axis] = box

    scale_box = QDoubleSpinBox()
    scale_box.setRange(10.0, 1000.0)
    scale_box.setDecimals(1)
    scale_box.setSingleStep(0.5)
    scale_box.setSuffix(" %")
    scale_box.setKeyboardTracking(False)
    scale_row = QGridLayout()
    scale_row.addWidget(QLabel("atlas size"), 0, 0)
    scale_row.addWidget(scale_box, 0, 1)

    fit = _shrinkable(QPushButton("Auto-fit size + position (bounding boxes)"))
    clear = _shrinkable(QPushButton("Reset (centred, 100%)"))
    status = QLabel()
    summary = QLineEdit()
    summary.setReadOnly(True)               # a line to copy into a notebook
    _shrinkable(summary)

    def apply_look(overlaid):
        """Grayscale side by side, additive green/magenta superimposed.

        Two grays drawn on top of each other is not a comparison -- whichever
        is on top is simply the picture. Additive complementary colours make
        agreement read as white and disagreement as a coloured fringe, which
        is the judgement this mode is for. Both are ordinary layer settings,
        so the layer list can override either of them afterwards.
        """
        for pane in win.panes:
            for layer, colour in ((pane.layers.sample, "green"),
                                  (pane.layers.template, "magenta")):
                if layer is None:
                    continue
                layer.colormap = colour if overlaid else "gray"
                layer.blending = "additive" if overlaid else "translucent"

    def on_mode(*_args):
        overlaid = overlay.isChecked()
        overlay.setText("Side by side (separate)" if overlaid
                        else "Overlay the two (superimpose)")
        apply_look(overlaid)
        win.set_sample(side_by_side=not overlaid)
        # The pictures just moved half a brain sideways; the cameras follow.
        win.reset_views()

    def on_numbers(*_args):
        if busy["in"]:
            return
        offset_um = np.zeros(3)
        for axis, box in offset_boxes.items():
            offset_um[axis] = box.value()
        win.set_sample(offset_um=offset_um, scale=scale_box.value() / 100.0)

    def on_fit(*_args):
        fitted = win.fit_sample()
        if fitted is None:
            status.setText("auto-fit found no foreground in one of the two volumes")
            return
        offset_um, scale, index = fitted
        print(f"[sample] auto-fit: offset "
              f"{tuple(round(float(offset_um[a])) for a in in_plane)} um, "
              f"scale {scale * 100:.1f}%, sample plane {index}")
        win.reset_views()

    def on_clear(*_args):
        win.set_sample(offset_um=np.zeros(3), scale=1.0)
        win.reset_views()

    def sync():
        """Follow the state, whatever moved it -- the same busy-flag dance as
        every other panel here (see _add_plane_panel.sync)."""
        busy["in"] = True
        try:
            for axis, box in offset_boxes.items():
                box.setValue(float(smp.offset_um[axis]))
            scale_box.setValue(smp.scale * 100.0)
            depth = int(smp.volume.shape[_SAMPLE_PANE_AXIS])
            plane_label.setText(f"plane {smp.index} of {depth} "
                                f"({smp.index * smp.voxel[_SAMPLE_PANE_AXIS] / 1000:.2f} mm "
                                f"into the stack)")
            status.setText(
                f"sample grid {tuple(smp.volume.shape)}, voxels "
                f"{tuple(round(float(v), 2) for v in smp.voxel)} um"
                f"\natlas {voxel:g} um; the sample slider moves the sample only")
            summary.setText(
                "angles_deg: [" + ", ".join(f"{a:.1f}" for a in euler_from_frame(state.frame))
                + "]   offset_um: [" + ", ".join(f"{smp.offset_um[a]:.0f}" for a in in_plane)
                + f"]   scale: {smp.scale:.3f}")
        finally:
            busy["in"] = False

    overlay.clicked.connect(on_mode)
    for box in offset_boxes.values():
        box.valueChanged.connect(on_numbers)
    scale_box.valueChanged.connect(on_numbers)
    fit.clicked.connect(on_fit)
    clear.clicked.connect(on_clear)

    dock = QWidget()
    dock.setMinimumWidth(1)
    layout = QVBoxLayout(dock)
    layout.addWidget(_caption(
        "Your sample, exactly as acquired -- one plane of it, never rotated, "
        "never resampled, and scrolled by its own slider alone. The ATLAS is "
        "what moves: turn it (Planes panel, or Shift+drag in a view) and "
        "scroll its planes until the two agree, then superimpose to check. "
        "Size and position first, angle second -- two brains that are not the "
        "same size cannot be compared for angle at all.", 110))
    layout.addWidget(overlay)
    layout.addWidget(QLabel("Sample plane (its own slider -- moves nothing else)"))
    layout.addLayout(plane_row)
    _add_sample_slider(win, layout)
    layout.addWidget(QLabel("Atlas centre, relative to the sample's (in plane)"))
    layout.addLayout(offsets)
    layout.addLayout(scale_row)
    layout.addWidget(fit)
    layout.addWidget(clear)
    layout.addWidget(_shrinkable(ontology_tree_ui.scrollable(status, 56)))
    layout.addWidget(summary)
    dock_widget = viewer.window.add_dock_widget(dock, area="right", name="Sample")

    state.listeners.append(sync)
    apply_look(False)
    # Opening on the neutral transform would mean opening on a half-brain
    # sample beside a whole-brain atlas at two different sizes, i.e. on the
    # one view of them that says nothing. The fit is a guess, and it is the
    # first thing anyone would press.
    on_fit()
    sync()
    return SimpleNamespace(sync=sync, dock=dock_widget, overlay=overlay,
                           offset_boxes=offset_boxes, scale_box=scale_box)


def _add_hover_bar(viewer, atlas, panes, below=None):
    """The shared bottom bar (shared/hover_bar.py), wired to these panes.

    This replaced the old right-hand "Region hierarchy" dock; the module's
    docstring says why a strip along the bottom beats a column on the right.
    All that is left here is the wiring: what the cursor is over, and what
    colour this window is drawing it in.

    `below` is the ortho-views dock, so the bar is split beneath it and spans
    the whole width at the very bottom rather than being parked beside the
    ortho panes.

    Returns hover_bar.add_hover_bar's handle -- SimpleNamespace(label=,
    show=, dock=) -- so the behaviour is testable without synthesising Qt
    mouse events: `show` takes an ontology id and is the whole of what the
    mouse callbacks do.
    """
    def colour_of(structure_id):
        """The RGBA napari paints this region with, straight out of the
        annotation layer's own colormap -- so the bar cannot drift out of
        step with the picture, whatever colormap the layer ends up on."""
        try:
            layer = panes[0].layers.annotation
            index = atlas.index_of_id.get(int(structure_id), 0)
            return list(layer.colormap.map(np.array([index]))[0])
        except Exception:               # any colormap napari might grow later
            return [0.5, 0.5, 0.5, 1.0]

    bar = hover_bar.add_hover_bar(
        viewer, atlas.structures, colour_of, below=below,
        resting="Hover over the atlas to read the region under the cursor.")

    def watcher(pane):
        def on_move(_model, event):
            # The annotation layer is a 2D reslice now, so the value under the
            # cursor is a plain lookup in it -- no view direction, no slider
            # axis to reconstruct. It holds COMPACT indices (see
            # atlas_reference._compact_annotation), which the bar knows
            # nothing about, so they are mapped back to ontology ids here.
            index = pane.layers.annotation.get_value(event.position, world=True)
            index = 0 if index is None else int(index)
            bar.show(int(atlas.present_ids[index]) if index > 0 else 0)
        return on_move

    for pane in panes:
        pane.model.mouse_move_callbacks.append(watcher(pane))
    return bar


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

    A large, DEDICATED dock on the LEFT, with the plane panel on the right:
    the ontology sits 2-12 levels deep, so a tree squeezed into a fraction of
    a shared column leaves most of it scrolled out of view. Each side gets the
    window's full height instead. Its WIDTH is a starting size only
    (_REGION_DOCK_WIDTH) -- see _shrinkable for what used to nail it open.
    """
    search = _shrinkable(QLineEdit())
    search.setPlaceholderText("Filter by name / acronym / id, any order...")
    hide_empty = _shrinkable(QCheckBox("Only regions with voxels here"))
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
    # A tree scrolls, so nothing inside it needs to dictate the dock's width.
    _shrinkable(tree)
    items = ontology_tree_ui.populate_ontology_tree(tree, atlas.structures, atlas.node_voxels)
    # Content-driven, but CAPPED. Sizing the name column to its contents alone
    # opens it at the longest region name in the atlas -- "Anterior cingulate
    # area, dorsal part, layer 6a" and friends, times a deep indent -- which
    # is most of a screen wide before the two number columns even start. The
    # cap keeps the usual name readable and lets the rest scroll. Both this
    # and the dock width below are only starting points: Interactive is the
    # header's default resize mode, so the column and the dock edge can both
    # be dragged to whatever width you want.
    tree.resizeColumnToContents(0)
    tree.header().setStretchLastSection(False)
    tree.setColumnWidth(0, min(tree.columnWidth(0), _REGION_NAME_COLUMN_WIDTH))

    status = QLabel()
    jump_btn = _shrinkable(QPushButton("Jump to selection centre"))

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
    dock.setMinimumWidth(1)
    layout = QVBoxLayout(dock)
    layout.addWidget(_caption("Atlas ontology -- selecting a node highlights it and every "
                              "descendant in all three views; Ctrl/Shift-click to highlight "
                              "several regions at once."))
    layout.addWidget(search)
    layout.addWidget(hide_empty)
    layout.addWidget(tree)
    layout.addWidget(_shrinkable(ontology_tree_ui.scrollable(status, 56)))
    layout.addWidget(jump_btn)
    dock_widget = viewer.window.add_dock_widget(dock, area="left", name="Region selection")
    # A starting width, not a fixed one: with everything above allowed to
    # shrink, the dock's own edge is now draggable from here to either
    # extreme.
    viewer.window._qt_window.resizeDocks([dock_widget], [_REGION_DOCK_WIDTH], Qt.Horizontal)

    refresh_filter()


def _run_view(atlas_cfg):
    _import_gui()
    print("[atlas] loading atlas reference...")
    atlas = atlas_reference.load_atlas_reference(atlas_cfg, include_template=True)
    sample = None
    if getattr(atlas_cfg, "sample", None) is not None:
        # The atlas's own voxel size is what decides how much of the sample is
        # worth reading when the config does not say -- see _sample_steps.
        sample = atlas_reference.load_sample_volume(atlas_cfg.sample,
                                                    target_um=atlas_cfg.resolution_um)
    win = _open_atlas_window(atlas, atlas_cfg.resolution_um, ortho=atlas_cfg.ortho,
                             sample=sample)
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
    signs = [pane_handedness(axes_from_frame(identity_frame(), order)) for order in orders]
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
            sl = plane_slice([volume], shape, origin, axes_from_frame(frame, order), centre)
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
    axes = axes_from_frame(frame, order)
    centre = origin + np.array([1.0, -2.0, 0.5])
    sl = plane_slice([volume], shape, origin, axes, centre)
    image = sl.images[0]

    rng = np.random.default_rng(1)
    for _ in range(200):
        i = int(rng.integers(image.shape[0]))
        j = int(rng.integers(image.shape[1]))
        point = plane_point(origin, axes, sl.offset, sl.s_min + i, sl.t_min + j)
        voxel = np.floor(point + 0.5).astype(int)
        if np.any(voxel < 0) or np.any(voxel >= np.array(shape)):
            assert image[i, j] == 0, (i, j, voxel, image[i, j])
        else:
            flat = (voxel[0] * shape[1] + voxel[1]) * shape[2] + voxel[2]
            assert image[i, j] == flat, (i, j, voxel, image[i, j], flat)

    # A tilted plane still holds the point it is centred on, to well under a
    # voxel -- otherwise clicking would drift the crosshair off its own plane.
    back = plane_point(origin, axes, sl.offset, sl.s_centre, sl.t_centre)
    assert np.allclose(back, centre, atol=1e-9), (back, centre)

    # Every volume handed to one call is read at the same coordinates: a mask
    # built from the volume must resample to the mask of the resampled volume.
    mask = (volume % 7 == 0).astype(np.uint8)
    both = plane_slice([volume, mask], shape, origin, axes, centre)
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
    print("7. crosshair: the cut lines cross at the crosshair and stop at the grid")
    shape = (6, 8, 10)
    volume = np.zeros(shape, dtype=np.uint8)
    origin = volume_origin(shape)
    order = (0, 1, 2)
    frame = identity_frame()
    axes = axes_from_frame(frame, order)
    centre = origin + np.array([0.0, 2.0, -3.0])
    sl = plane_slice([volume], shape, origin, axes, centre)

    # Un-tilted, the two cuts run along the pane's own axes, which is the
    # plain axis-aligned cross this viewer has always drawn.
    directions = cut_directions(axes, frame, order)
    assert np.allclose(directions[0], (0, 1)), directions
    assert np.allclose(np.abs(directions[1]), (1, 0)), directions
    vectors = crosshair_vectors(sl, directions)
    assert vectors.shape == (2, 2, 2), vectors.shape
    # Line 0 is the plane-order[1] cut: constant s, spanning the full width.
    assert np.isclose(vectors[0, 0, 0], sl.s_centre) and np.isclose(vectors[0, 1, 0], 0)
    assert np.isclose(abs(vectors[0, 1, 1]), sl.t_max - sl.t_min)
    # Line 1 is the plane-order[2] cut: constant t, spanning the full height.
    assert np.isclose(vectors[1, 0, 1], sl.t_centre) and np.isclose(vectors[1, 1, 1], 0)
    assert np.isclose(abs(vectors[1, 1, 0]), sl.s_max - sl.s_min)

    # Tilted, they are oblique -- still crossing at the crosshair, still
    # perpendicular to each other, and still stopping at the grid's edge
    # rather than running off past the picture.
    turned = rotated_frame(frame, order[0], np.deg2rad(31))
    oblique = cut_directions(axes, turned, order)
    assert not np.allclose(oblique[0], (0, 1)), oblique
    dots = np.dot(oblique[0], oblique[1])
    assert abs(dots) < 1e-9, oblique
    vectors = crosshair_vectors(sl, oblique)
    lo = np.array([sl.s_min, sl.t_min])
    hi = np.array([sl.s_max, sl.t_max])
    for line, direction in zip(vectors, oblique):
        start, span = line[0], line[1]
        assert np.all(start >= lo - 1e-6) and np.all(start <= hi + 1e-6), line
        assert np.all(start + span >= lo - 1e-6) and np.all(start + span <= hi + 1e-6), line
        # The crosshair lies on the line, between its two ends.
        cross = np.array([sl.s_centre, sl.t_centre])
        along = np.dot(cross - start, span) / np.dot(span, span)
        assert -1e-6 <= along <= 1 + 1e-6, (along, line)
        assert np.isclose(point_line_distance(cross, start, span), 0, atol=1e-9), line
        assert np.allclose(span / np.linalg.norm(span),
                           np.array(direction) * np.sign(np.dot(span, direction))), line

    # A pane's shift moves the cross and its grid together -- it is what keeps
    # the crosshair still on screen while the planes turn.
    shifted = crosshair_vectors(sl, directions, (2.5, -1.5))
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
        axes = axes_from_frame(frame, order)
        full = plane_slice([volume], shape, origin, axes, centre)
        coarse = plane_slice([volume], shape, origin, axes, centre, stride=_DRAG_STRIDE)
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
    print("9. drags aim the cut lines; the dragged pane's own picture holds still")

    def screen(axes, shift, origin, point):
        """Where a voxel lands in a pane's drawn (s, t), the way refresh()
        places the image and crosshair_vectors places the cross."""
        rel = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
        return np.array([np.dot(rel, axes[1]) + shift[0],
                         np.dot(rel, axes[2]) + shift[1]])

    def screen_angle(direction):
        """The convention _rotation_drag measures the cursor in: s down,
        t across."""
        return float(np.arctan2(direction[0], direction[1]))

    shape = (24, 30, 36)
    volume = _ramp_volume(shape)
    origin = volume_origin(shape)
    swept = np.deg2rad(11.0)                # the angle the cursor sweeps
    for dragged, _title in _ORTHO_PANES:
        frame = frame_from_euler((7, -13, 21))      # start off-axis; nothing special
        centre = origin + np.array([2.0, -3.0, 1.5])
        panes = []
        for order, _t in _ORTHO_PANES:
            axes = axes_from_frame(frame, order)
            panes.append(SimpleNamespace(order=order, axes=axes, shift=np.zeros(2),
                                         handedness=pane_handedness(axes)))
        pane = next(p for p in panes if p.order == dragged)

        before_image = plane_slice([volume], shape, origin, pane.axes, centre).images[0]
        before_axes = pane.axes.copy()
        before_lines = cut_directions(pane.axes, frame, pane.order)
        crosses = [screen(p.axes, p.shift, origin, centre) for p in panes]

        # Exactly what _rotation_drag -> win.rotate -> set_frame do between
        # two mouse moves, including the sign the drag applies.
        turned = rotated_frame(frame, dragged[0], -pane.handedness * swept)
        for p, was in zip(panes, crosses):
            normal = turned[p.order[0]]
            p.axes = np.array([normal, *plane_basis(normal, p.axes[1], p.axes[2],
                                                    p.handedness)])
            p.shift += was - screen(p.axes, p.shift, origin, centre)

        # The dragged pane draws exactly the same picture as before: same
        # plane, same axes, same pixels, no shift. THIS is "the sample does
        # not move" -- it used to spin with the mouse.
        assert np.allclose(pane.axes, before_axes, atol=1e-12), dragged
        assert np.allclose(pane.shift, 0, atol=1e-12), pane.shift
        after_image = plane_slice([volume], shape, origin, pane.axes, centre).images[0]
        assert np.array_equal(before_image, after_image), dragged

        # What DOES move is the pair of cut lines lying across it: they swing
        # by the angle the cursor swept, the way the cursor went, in all three
        # panes alike (that is what the handedness sign is for).
        after_lines = cut_directions(pane.axes, turned, pane.order)
        for was, now in zip(before_lines, after_lines):
            delta = screen_angle(now) - screen_angle(was)
            delta = float(np.arctan2(np.sin(delta), np.cos(delta)))
            assert np.isclose(delta, swept, atol=1e-9), (dragged, np.rad2deg(delta))
        # ...and they stay perpendicular to each other, i.e. still three
        # orthogonal planes.
        assert abs(np.dot(after_lines[0], after_lines[1])) < 1e-9, after_lines

        # The other two planes really did tilt, by that same angle.
        for p in panes:
            if p.order is dragged:
                continue
            was_normal, now_normal = frame[p.order[0]], turned[p.order[0]]
            angle = np.arccos(np.clip(np.dot(was_normal, now_normal), -1.0, 1.0))
            assert np.isclose(angle, swept, atol=1e-9), (p.order, np.rad2deg(angle))

        # And in every pane the crosshair is the pivot: it must not move on
        # screen at all, or the picture would slide out from under the drag.
        for p, was in zip(panes, crosses):
            assert np.allclose(screen(p.axes, p.shift, origin, centre), was, atol=1e-9), p.order

    # A press only grabs a line when it lands on one, whatever the zoom.
    axes = axes_from_frame(identity_frame(), (0, 1, 2))
    directions = cut_directions(axes, identity_frame(), (0, 1, 2))
    origin2d = np.zeros(2)
    assert point_line_distance((0.0, 5.0), origin2d, directions[0]) == 0.0
    assert point_line_distance((3.0, 5.0), origin2d, directions[0]) == 3.0
    assert point_line_distance((0.0, 0.0), origin2d, (0.0, 0.0)) == np.inf
    print("   ok")


def selftest_pane_axes():
    print("10. pane axes: carried across a tilt, never re-read off the frame")
    for order, _title in _ORTHO_PANES:
        axes = axes_from_frame(identity_frame(), order)
        handedness = pane_handedness(axes)
        # A pane's axes stay an orthonormal, correctly handed basis of its own
        # plane through any number of tilts, without drifting.
        frame = identity_frame()
        rng = np.random.default_rng(3)
        for _ in range(200):
            axis = int(rng.integers(3))
            frame = rotated_frame(frame, axis, float(rng.normal(scale=0.4)))
            normal = frame[order[0]]
            axes = np.array([normal, *plane_basis(normal, axes[1], axes[2], handedness)])
            assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-9), (order, axes)
            assert np.isclose(pane_handedness(axes), handedness), order

            # Turning the frame about THIS pane's own normal must leave its
            # axes alone -- that is the whole "the sample does not move".
            held = rotated_frame(frame, order[0], float(rng.normal(scale=0.5)))
            after = plane_basis(held[order[0]], axes[1], axes[2], handedness)
            assert np.allclose(after, axes[1:], atol=1e-9), (order, after, axes)

        # The degenerate case: a plane tilted until the old ROW axis points
        # along the new normal. The column hint carries it instead of blowing
        # up, and comes back unchanged.
        normal = axes[1].copy()
        basis = plane_basis(normal, axes[1], axes[2], handedness)
        assert np.allclose(basis[1], axes[2], atol=1e-9), basis
        assert np.isclose(np.dot(basis[0], normal), 0, atol=1e-9), basis
        assert np.isclose(pane_handedness(np.array([normal, *basis])), handedness)
    print("   ok")


def selftest_hover_bar_line():
    print("11. hover bar: keeps the deepest levels, folds the shallow end away")
    # The bar itself lives in shared/hover_bar.py now (paint_mask's labels
    # mode grew the same strip); this keeps it in THIS tool's selftest run,
    # because this is the tool whose window it was built for.
    hover_bar.selftest_ancestry_line()
    hover_bar.selftest_ancestry_labels()
    print("   ok")


def selftest_sample_mapping():
    print("12. sample: placed against the atlas in plane, by one offset and one scale")
    shape = (40, 50, 60)
    voxel = 20.0
    isotropic = np.full(3, voxel)

    # Neutral case: same grid, same voxel size, nothing offset or rescaled.
    # A sample voxel must then land exactly where the atlas draws the voxel of
    # the same index.
    for order, _title in _ORTHO_PANES:
        scale, translate = sample_placement(order, shape, isotropic, 1.0, np.zeros(3))
        for dim, in_plane in enumerate((order[1], order[2])):
            world = 7.0 * scale[dim] + translate[dim]
            expected = (7.0 - volume_origin(shape)[in_plane]) * voxel
            assert np.isclose(world, expected), (order, dim, world, expected)

    # General case: a different grid, anisotropic voxels, an offset and a size
    # ratio. The sample index meaning the same point as atlas coordinate s has
    # to land on the same world position -- an origin or a sign wrong here
    # shows up as a sample that slides out from under the atlas as soon as
    # either is rescaled.
    sample_shape = (30, 44, 55)
    sample_voxel = np.array([25.0, 10.0, 10.0])
    sample_origin = volume_origin(sample_shape)
    size_ratio, offset_um = 1.3, np.array([120.0, -75.0, 40.0])

    for order, _title in _ORTHO_PANES:
        scale, translate = sample_placement(order, sample_shape, sample_voxel,
                                            size_ratio, offset_um)
        for dim, in_plane in enumerate((order[1], order[2])):
            for s in (-9.0, 0.0, 12.0):
                index = (sample_origin[in_plane]
                         + (s * voxel * size_ratio + offset_um[in_plane])
                         / sample_voxel[in_plane])
                world = index * scale[dim] + translate[dim]
                assert np.isclose(world, s * voxel), (order, dim, s, world)

    # Only the two in-plane axes are read: the third has its own slider, and
    # anything written in its slot must not move the picture.
    for order, _title in _ORTHO_PANES:
        offset = np.zeros(3)
        offset[int(order[0])] = 9999.0
        assert all(np.allclose(a, b) for a, b in
                   zip(sample_placement(order, sample_shape, sample_voxel, 1.0, offset),
                       sample_placement(order, sample_shape, sample_voxel, 1.0, np.zeros(3))))
    print("   ok")


def selftest_sample_slices():
    print("13. sample: one acquisition plane, taken whole, clipped to the stack")
    volume = _ramp_volume((6, 7, 8))
    for order, _title in _ORTHO_PANES:
        image, index = sample_plane_image(volume, order, 2)
        assert index == 2, (order, index)
        # take() leaves the remaining axes in ascending order, which for every
        # pane is already (down the screen, across it).
        assert np.array_equal(image, np.take(volume, 2, axis=order[0]))
        assert image.shape == (volume.shape[order[1]], volume.shape[order[2]]), image.shape

        # A slider cannot ask for a plane outside the stack, but a stale index
        # after a reload could: clip, and say which plane that was.
        for outside, expected in ((-3, 0), (volume.shape[order[0]] + 3, volume.shape[order[0]] - 1)):
            edge, index = sample_plane_image(volume, order, outside)
            assert index == expected, (order, outside, index)
            assert np.array_equal(edge, np.take(volume, expected, axis=order[0]))
    print("   ok")


def selftest_sample_autofit():
    print("14. sample auto-fit: same size and same place IN PLANE, third axis left alone")
    atlas = np.zeros((60, 60, 60), dtype=np.uint8)
    atlas[10:30, 20:40, 25:55] = 3               # a labelled blob, off centre
    sample = np.zeros((80, 70, 90), dtype=np.uint16)
    sample[20:60, 10:40, 30:75] = 900            # a different size, elsewhere
    voxel, sample_voxel = 20.0, np.array([10.0, 25.0, 12.0])
    order = _ORTHO_PANES[_SAMPLE_PANE_AXIS][0]
    in_plane = (order[1], order[2])

    atlas_box = foreground_box(atlas, 0)
    sample_box = foreground_box(sample, foreground_threshold(sample))
    assert np.allclose(atlas_box[1], (20, 20, 30)), atlas_box
    assert np.allclose(sample_box[1], (40, 30, 45)), sample_box

    offset_um, scale = fit_sample_placement(atlas_box, sample_box, volume_origin(atlas.shape),
                                            voxel, sample.shape, sample_voxel, order)

    # The two brains' centres have to land on each other on screen: the atlas
    # box centre and the sample box centre must come out at the same world
    # position, read through sample_placement -- i.e. through the arithmetic
    # the pane really uses rather than a restatement of the fit.
    layer_scale, translate = sample_placement(order, sample.shape, sample_voxel,
                                              scale, offset_um)
    for dim, axis in enumerate(in_plane):
        sample_world = sample_box[0][axis] * layer_scale[dim] + translate[dim]
        atlas_world = (atlas_box[0][axis] - volume_origin(atlas.shape)[axis]) * voxel
        assert np.isclose(sample_world, atlas_world), (axis, sample_world, atlas_world)

    # In-plane ratios only. The third axis here is deliberately the odd one
    # out (0.5 against 0.9 and 0.9); a scale that included it would come out
    # visibly smaller and shrink the atlas in the two axes you can see.
    ratios = [(sample_box[1][a] * sample_voxel[a]) / (atlas_box[1][a] * voxel) for a in in_plane]
    assert np.isclose(scale, float(np.exp(np.mean(np.log(ratios))))), scale
    assert min(ratios) <= scale <= max(ratios), (scale, ratios)
    all_three = [(sample_box[1][a] * sample_voxel[a]) / (atlas_box[1][a] * voxel)
                 for a in range(3)]
    assert not np.isclose(scale, float(np.exp(np.mean(np.log(all_three))))), scale
    # Nothing is placed along the third axis: that is the sliders' business.
    assert offset_um[int(order[0])] == 0.0, offset_um

    assert foreground_box(np.zeros((4, 4, 4)), 0) is None
    print("   ok")


def selftest_locked_pane_axes():
    print("15. sample mode: the atlas turns under the sample instead of holding still")
    order = (0, 1, 2)
    frame = identity_frame()
    axes = axes_from_frame(frame, order)
    turned = rotated_frame(frame, 0, np.deg2rad(10.0))

    locked = axes_from_frame(turned, order)                 # what a sample forces
    carried = np.array([turned[0], *plane_basis(turned[0], axes[1], axes[2],
                                                pane_handedness(axes))])
    # Same plane either way: the rotation was about this pane's own normal.
    assert np.allclose(locked[0], carried[0]) and np.allclose(locked[0], axes[0])
    # Carried: the picture is pinned and the cut lines are what move.
    assert np.allclose(carried[1:], axes[1:])
    # Locked: the picture turns, by exactly the angle asked for -- which is
    # the in-plane component of the tilt, and invisible without this.
    angle = np.rad2deg(np.arccos(np.clip(float(np.dot(locked[1], axes[1])), -1.0, 1.0)))
    assert np.isclose(angle, 10.0), angle
    print("   ok")


def run_selftests():
    print("=== tools/atlas_view.py selftests (synthetic data only, no GUI) ===")
    selftest_ortho_panes_geometry()
    selftest_frame_algebra()
    selftest_plane_bounds()
    selftest_identity_slices()
    selftest_oblique_sampling()
    selftest_offsets_roundtrip()
    selftest_crosshair_vectors()
    selftest_coarse_expansion()
    selftest_rotation_drag()
    selftest_pane_axes()
    selftest_hover_bar_line()
    selftest_sample_mapping()
    selftest_sample_slices()
    selftest_sample_autofit()
    selftest_locked_pane_axes()
    print("=== all selftests passed ===")
    print("(shared/atlas_reference.py --selftest covers atlas loading / ontology maths)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Browse an atlas annotation against its ontology")
    local_config.add_config_arg(parser, "atlas_view")
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
