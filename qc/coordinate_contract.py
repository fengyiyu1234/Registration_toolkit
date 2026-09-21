"""Coordinate contract for the full-chain QC viewer.

Public coordinates are always ``(x, y, z)``. Conversion to napari's
``(z, y, x)`` order belongs at the display boundary.
"""
from __future__ import annotations

import numpy as np


def _xyz(value, name):
    out = np.asarray(value, dtype=float)
    if out.ndim == 0 or out.shape[-1] != 3 or not np.isfinite(out).all():
        raise ValueError(f"{name} must end in three finite values in (x, y, z) order")
    return out


def global_pixel_to_physical(global_xyz, voxel_um):
    return _xyz(global_xyz, "global_xyz") * _xyz(voxel_um, "voxel_um")


def physical_to_global_pixel(physical_um, voxel_um):
    return _xyz(physical_um, "physical_um") / _xyz(voxel_um, "voxel_um")


def physical_to_label_voxel(physical_um, origin_um, spacing_um, direction=None):
    p = _xyz(physical_um, "physical_um")
    origin = _xyz(origin_um, "origin_um")
    spacing = _xyz(spacing_um, "spacing_um")
    d = np.eye(3) if direction is None else np.asarray(direction, dtype=float)
    if d.shape != (3, 3) or not np.isfinite(d).all():
        raise ValueError("direction must be a finite 3x3 matrix")
    return ((p - origin) @ np.linalg.inv(d).T) / spacing


def label_voxel_to_physical(voxel_xyz, origin_um, spacing_um, direction=None):
    v = _xyz(voxel_xyz, "voxel_xyz")
    origin = _xyz(origin_um, "origin_um")
    spacing = _xyz(spacing_um, "spacing_um")
    d = np.eye(3) if direction is None else np.asarray(direction, dtype=float)
    if d.shape != (3, 3) or not np.isfinite(d).all():
        raise ValueError("direction must be a finite 3x3 matrix")
    return origin + (v * spacing) @ d.T


def napari_xyz_to_zyx(xyz):
    return _xyz(xyz, "xyz")[::-1]
