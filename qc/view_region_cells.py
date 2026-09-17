#!/usr/bin/env python
"""Post-registration QC: pick registered cells by atlas region, go back to the
raw data around each one, and judge whether it really sits in that region.

The GUI-free half (selection, geometry, depth, verdict file) is
qc/region_cells.py -- read its docstring for the coordinate frames.

THREE WAYS TO RUN (antsreg env, from the repo root)

    python qc/view_region_cells.py                  # napari, one site at a time
    python qc/view_region_cells.py --summary        # numbers only, no images read
    python qc/view_region_cells.py --snapshot       # one PNG per site, headless
    python qc/view_region_cells.py --summary --frame-check   # + coordinate check

    python qc/view_region_cells.py configs/region_qc.s12t.yaml   # another config

`source: tiles` reads the 0.65 um tiles (the Windows machine that holds Y:);
`source: volume` reads the registration-grid tiff(s) and works anywhere the
run directory is.  Same config otherwise.

IN THE VIEWER

    n / p           next / previous site
    u               next site without a verdict
    1 2 3 4         区域正确 / 区域错误 / 不是细胞 / 看不清   (then moves on)
    hover           region under the cursor, and the cell under it

Verdicts go to <out_dir>/verdicts.csv on every click, keyed by run_dir and
cell id, so a session can be closed and resumed; re-judging overwrites.

WHAT TO LOOK AT
  * The ring with the crosshair is the site's cell.  Coloured rings are cells
    in the selected region set (colour = GFP / RFP / GFP_RFP, diamond =
    Sox9+); grey rings are every other cell in the box.
  * The yellow contour is the selected region set as labels_in_sample puts it
    in the sample.  Thin white lines are all other region boundaries.
  * The info panel prints both region answers (the cell table's and the
    label volume's) and the signed distance to the region edge.  A cell
    labelled L2/3 that is 15 um inside the edge is a registration-precision
    question, not a mistake; one 150 um outside it is.
  * The signal ratio is the qc/cut_crops check: mean intensity at the cells'
    centres over random positions in the box.  Around 1 means the frame is
    wrong.  On the 2.6 um volume it is weak (a 50 um shift barely moves it),
    so there run --frame-check once per run instead: it compares marker+ and
    marker- cells on the whole brain and reports the z offset per tile.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qc import region_cells as rc  # noqa: E402
from shared import local_config  # noqa: E402

TOOL = "region_qc"
REQUIRED = ("run_dir", "ontology_json", "regions", "out_dir")


# ── Shared drawing helpers ────────────────────────────────────────────────────

def _point_style(cells):
    """(border colours, symbols) for a cell table slice."""
    colors, symbols = [], []
    for name, selected in zip(cells["class_name"], cells["selected"]):
        fp, sox9 = rc.marker_key(name)
        colors.append(rc.MARKER_COLORS.get(fp, (1, 1, 1, 1)) if selected else rc.OTHER_COLOR)
        symbols.append("diamond" if sox9 else "disc")
    return np.array(colors, float).reshape(-1, 4), np.array(symbols)


def _contrast(arr, pct=(0.5, 99.8)):
    sample = arr[::max(1, arr.shape[0] // 8), ::4, ::4]
    lo, hi = np.percentile(sample, pct)
    return float(lo), float(max(hi, lo + 1))


def _channel_colormap(name, n_channels):
    if n_channels == 1:
        return "gray"
    return rc.CHANNEL_COLORMAPS.get(name, "gray")


def _boundaries_2d(lab2d):
    edge = np.zeros(lab2d.shape, bool)
    edge[:-1, :] |= lab2d[:-1, :] != lab2d[1:, :]
    edge[:, :-1] |= lab2d[:, :-1] != lab2d[:, 1:]
    return edge


# ── Headless PNG ──────────────────────────────────────────────────────────────

def render_snapshot(session, k, cut, path, mip_um=None):
    """One site -> one PNG: max projection of the box over +-mip_um around the
    cell's z, region contour on the slice through the cell, cells within the
    projected slab."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    box = cut["box"]
    vox = box.voxel_um
    centre_local = box.local(cut["center"])[0]
    zc = int(np.clip(np.rint(centre_local[2]), 0, box.shape_xyz[2] - 1))
    if mip_um is None:
        mip_um = max(vox[2], 16.0) if box.source == "tiles" else 0.0
    half = int(np.floor(mip_um / vox[2]))
    z0, z1 = max(0, zc - half), min(box.shape_xyz[2], zc + half + 1)

    rgb = np.zeros((box.shape_xyz[1], box.shape_xyz[0], 3), float)
    tints = {"red": (1, 0, 0), "green": (0, 1, 0), "cyan": (0, 1, 1),
             "magenta": (1, 0, 1), "gray": (1, 1, 1)}
    for name, arr in box.arrays.items():
        lo, hi = _contrast(arr)
        img = np.clip((arr[z0:z1].max(0).astype(float) - lo) / (hi - lo), 0, 1)
        rgb += img[..., None] * np.array(tints[_channel_colormap(name, len(box.arrays))])
    rgb = np.clip(rgb, 0, 1)

    # labels on the image grid, on the plane through the cell
    plane_origin = box.origin_um.copy()
    plane_origin[2] = box.origin_um[2] + zc * vox[2]
    lab = session.labels.resample_to(plane_origin, vox, [box.shape_xyz[0], box.shape_xyz[1], 1])[:, :, 0].T
    sel = np.isin(lab, list(session.region_ids))

    ext = [box.origin_um[0], box.origin_um[0] + box.shape_xyz[0] * vox[0],
           box.origin_um[1] + box.shape_xyz[1] * vox[1], box.origin_um[1]]
    fig, ax = plt.subplots(figsize=(9, 9 * rgb.shape[0] / rgb.shape[1] + 1.2), dpi=110)
    ax.imshow(rgb, extent=ext, interpolation="nearest")
    edge = _boundaries_2d(lab)
    overlay = np.zeros(lab.shape + (4,))
    overlay[edge] = (1, 1, 1, 0.35)
    ax.imshow(overlay, extent=ext, interpolation="nearest")
    if sel.any() and not sel.all():
        xs = box.origin_um[0] + (np.arange(lab.shape[1]) + 0.0) * vox[0]
        ys = box.origin_um[1] + (np.arange(lab.shape[0]) + 0.0) * vox[1]
        ax.contour(xs, ys, sel.astype(float), levels=[0.5], colors=["yellow"], linewidths=1.4)

    cells = cut["cells"]
    zlo = box.origin_um[2] + (z0 - 0.5) * vox[2]
    zhi = box.origin_um[2] + (z1 - 0.5) * vox[2]
    slab = cells[(cells["pz"] >= zlo) & (cells["pz"] < zhi)]
    if len(slab):
        colors, symbols = _point_style(slab)
        for sym, marker in (("disc", "o"), ("diamond", "D")):
            m = symbols == sym
            if m.any():
                ax.scatter(slab["px"][m], slab["py"][m], s=60, marker=marker,
                           facecolors="none", edgecolors=colors[m], linewidths=1.0)
    c = cut["center"]
    ax.scatter([c[0]], [c[1]], s=400, marker="o", facecolors="none",
               edgecolors="white", linewidths=1.8)
    ax.axhline(c[1], color="white", lw=0.4, alpha=0.5)
    ax.axvline(c[0], color="white", lw=0.4, alpha=0.5)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

    info = session.describe_site(k)
    ratio = cut["signal_ratio"]
    ax.set_title(
        f"site {k}  {info['cell_id']}   px ({info['xyz_px']})\n"
        f"table: {info['table_region']}\n"
        f"labels: {info['labels_region']}   depth {info['depth']}   "
        f"signal {ratio:.2f}   z slab {mip_um:.0f} µm",
        fontsize=8, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run_snapshots(session):
    snap_dir = session.out_dir / "snapshots"
    rows = []
    for k in range(len(session.sites)):
        cut = session.cut_site(k)
        info = session.describe_site(k)
        name = f"site{k:03d}_{info['cell_id'].replace(':', '_')}.png"
        render_snapshot(session, k, cut, snap_dir / name, session.cfg.get("snapshot_mip_um"))
        ratio = cut["signal_ratio"]
        flag = "" if np.isfinite(ratio) and ratio >= _signal_floor(session) else "   <-- CHECK"
        print(f"  {name}  cells {len(cut['cells']):4d}  signal {ratio:5.2f}{flag}")
        rows.append({**info, "signal_ratio": ratio, "n_cells_in_box": len(cut["cells"]),
                     "png": name})
    out = snap_dir / "sites.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n{len(rows)} 张 -> {snap_dir}\n清单 -> {out}")


# ── napari ────────────────────────────────────────────────────────────────────

class Viewer:
    def __init__(self, session, start=0, extra_lines=()):
        import napari
        from concurrent.futures import ThreadPoolExecutor
        from qtpy.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                                    QPushButton, QVBoxLayout, QWidget, QCheckBox)

        self.s = session
        self.store = rc.VerdictStore(session.out_dir / "verdicts.csv", session.run_dir)
        self.k = None
        self.cut = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.viewer = napari.Viewer(title=f"region QC — {session.run_dir.name}")

        panel = QWidget()
        lay = QVBoxLayout(panel)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(1)  # Qt.RichText
        lay.addWidget(self.info)

        nav = QHBoxLayout()
        for text, fn in (("◀ p", self.prev), ("n ▶", self.next), ("未判 u", self.next_open)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            nav.addWidget(b)
        lay.addLayout(nav)

        grid = QGridLayout()
        for i, (key, label) in enumerate(rc.VERDICTS.items()):
            b = QPushButton(f"{i + 1}  {label}")
            b.clicked.connect(lambda _=False, v=key: self.judge(v))
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid)
        self.note = QLineEdit()
        self.note.setPlaceholderText("备注（随判定一起保存）")
        lay.addWidget(self.note)

        self.show_other = QCheckBox("显示其它脑区的细胞（灰）")
        self.show_other.setChecked(True)
        self.show_other.stateChanged.connect(lambda _: self._set_other_visible())
        lay.addWidget(self.show_other)
        shot = QPushButton("保存截图")
        shot.clicked.connect(self.screenshot)
        lay.addWidget(shot)

        self.tally = QLabel()
        lay.addWidget(self.tally)
        summary = QLabel("<br>".join(_html(line) for line in
                                     list(session.summary_lines()) + list(extra_lines)))
        summary.setWordWrap(True)
        summary.setTextFormat(1)
        summary.setStyleSheet("font-size: 10px; color: #aaa;")
        lay.addWidget(summary)
        lay.addStretch(1)
        self.viewer.window.add_dock_widget(panel, name="region QC", area="right")

        v = self.viewer
        v.bind_key("n", lambda _v: self.next(), overwrite=True)
        v.bind_key("p", lambda _v: self.prev(), overwrite=True)
        v.bind_key("u", lambda _v: self.next_open(), overwrite=True)
        for i, key in enumerate(rc.VERDICTS):
            v.bind_key(str(i + 1), lambda _v, kk=key: self.judge(kk), overwrite=True)
        v.mouse_move_callbacks.append(self._on_move)
        v.text_overlay.visible = True
        v.text_overlay.position = "bottom_left"
        v.text_overlay.font_size = 11

        self.show(start)

    # -- navigation ----------------------------------------------------------

    def _fetch(self, k):
        fut = self.futures.pop(k, None)
        return fut.result() if fut is not None else self.s.cut_site(k)

    def _prefetch(self, k):
        if 0 <= k < len(self.s.sites) and k not in self.futures:
            self.futures = {kk: f for kk, f in self.futures.items() if abs(kk - k) <= 1}
            self.futures[k] = self.pool.submit(self.s.cut_site, k)

    def show(self, k):
        n = len(self.s.sites)
        if n == 0:
            self.info.setText("没有选中任何细胞 —— 检查 regions / classes。")
            return
        k = int(np.clip(k, 0, n - 1))
        self.viewer.status = f"loading site {k} ..."
        self.k, self.cut = k, self._fetch(k)
        self._draw()
        self._prefetch(k + 1)

    def next(self):
        if self.k is not None:
            self.show(self.k + 1)

    def prev(self):
        if self.k is not None:
            self.show(self.k - 1)

    def next_open(self):
        n = len(self.s.sites)
        for j in list(range(self.k + 1, n)) + list(range(0, self.k + 1)):
            if self.store.get(self.s.sites.iloc[j]["cell_id"])[0] is None:
                self.show(j)
                return
        self.viewer.status = "全部 site 都判过了"

    def judge(self, verdict):
        row = self.s.sites.iloc[self.k]
        self.store.set(row, verdict, self.note.text().strip(), self.s.source.kind)
        self.viewer.status = f"site {self.k}: {rc.VERDICTS[verdict]}"
        if self.k < len(self.s.sites) - 1:
            self.next()
        else:
            self._update_info()

    def screenshot(self):
        info = self.s.describe_site(self.k)
        path = self.s.out_dir / "screens" / f"site{self.k:03d}_{info['cell_id'].replace(':', '_')}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.viewer.screenshot(str(path), canvas_only=True)
        self.viewer.status = f"saved {path}"

    # -- drawing -------------------------------------------------------------

    def _draw(self):
        from napari.utils.colormaps import DirectLabelColormap

        v, cut, box = self.viewer, self.cut, self.cut["box"]
        v.layers.clear()
        scale = tuple(box.voxel_um[::-1])
        translate = tuple(box.origin_um[::-1])
        for name, arr in box.arrays.items():
            v.add_image(arr, name=name, scale=scale, translate=translate,
                        colormap=_channel_colormap(name, len(box.arrays)),
                        blending="additive", contrast_limits=_contrast(arr))

        lab = cut["labels"].transpose(2, 1, 0)                     # xyz -> zyx
        lab_scale = tuple(self.s.labels.spacing[::-1])
        lab_translate = tuple(cut["labels_origin"][::-1])
        self.lab_layer = v.add_labels(lab.astype(np.int32), name="all regions",
                                      scale=lab_scale, translate=lab_translate,
                                      opacity=0.35, visible=True)
        self.lab_layer.contour = 1
        sel = np.isin(lab, list(self.s.region_ids)).astype(np.uint8)
        sel_layer = v.add_labels(sel, name="selected region", scale=lab_scale,
                                 translate=lab_translate, opacity=0.9,
                                 colormap=DirectLabelColormap(
                                     color_dict={1: "yellow", None: "transparent"}))
        sel_layer.contour = 2

        cells = cut["cells"].reset_index(drop=True)
        local = box.local(cells[["px", "py", "pz"]].to_numpy()) if len(cells) else np.zeros((0, 3))
        local[:, 2] = np.rint(local[:, 2])      # snap to the image's own slices
        diam = (12.0 if box.source == "tiles" else 20.0) / box.voxel_um[0]
        self.cells = cells
        self.point_layers = {}
        for name, m in (("other cells", ~cells["selected"].to_numpy()),
                        ("region cells", cells["selected"].to_numpy())):
            sub = cells[m]
            colors, symbols = _point_style(sub) if len(sub) else (np.zeros((0, 4)), np.array([]))
            layer = v.add_points(local[m][:, ::-1], name=name, size=diam, symbol=symbols,
                                 face_color="transparent", border_color=colors,
                                 border_width=0.12, scale=scale, translate=translate,
                                 features=sub.reset_index(drop=True)[
                                     ["cell_id", "class_name", "region_id"]])
            self.point_layers[name] = (layer, sub.reset_index(drop=True))
        self._set_other_visible()

        c = box.local(cut["center"])[0]
        c[2] = np.rint(c[2])
        target = v.add_shapes(
            [np.array([[c[2], c[1] - 3 * diam, c[0]], [c[2], c[1] + 3 * diam, c[0]]]),
             np.array([[c[2], c[1], c[0] - 3 * diam], [c[2], c[1], c[0] + 3 * diam]])],
            shape_type="line", edge_color="white", edge_width=diam * 0.08,
            name="target", scale=scale, translate=translate)
        target.editable = False

        world = box.origin_um[::-1] + c[::-1] * box.voxel_um[::-1]
        v.dims.ndisplay = 2
        v.dims.set_point(0, world[0])
        v.camera.center = tuple(world)
        v.reset_view()
        v.camera.center = tuple(world)
        self._update_info()

    def _set_other_visible(self):
        if getattr(self, "point_layers", None):
            self.point_layers["other cells"][0].visible = self.show_other.isChecked()

    def _update_info(self):
        d = self.s.describe_site(self.k)
        verdict, note = self.store.get(d["cell_id"])
        self.note.setText(note)
        ratio = self.cut["signal_ratio"]
        floor = _signal_floor(self.s)
        warn = "" if np.isfinite(ratio) and ratio >= floor else \
            f" <b style='color:#f66'>← 低于 {floor:g}，先查坐标系</b>"
        agree = "一致" if d["agrees"] else "<b style='color:#fa0'>不一致</b>"
        n_sel = int(self.cut["cells"]["selected"].sum())
        self.info.setText(
            f"<b>site {d['site'] + 1} / {len(self.s.sites)}</b>　{_html(d['cell_id'])}<br>"
            f"原始像素 ({d['xyz_px']})　tile {_html(str(d['tile']))}<br>"
            f"细胞表：{_html(d['table_region'])}<br>"
            f"标签图：{_html(d['labels_region'])}（{agree}）<br>"
            f"离所选区域边界 <b>{d['depth']}</b>（正 = 区内）<br>"
            f"框内细胞 {len(self.cut['cells'])}，其中所选区域 {n_sel}<br>"
            f"signal ratio {ratio:.2f}{warn}<br>"
            f"判定：<b>{rc.VERDICTS.get(verdict, '—') if verdict else '—'}</b>")
        tally = self.store.tally()
        self.tally.setText("已判：" + ("，".join(f"{rc.VERDICTS[k]} {n}" for k, n in tally.items()
                                            if k in rc.VERDICTS) or "0"))

    def _on_move(self, viewer, event):
        pos = np.asarray(viewer.cursor.position, float)
        if pos.size != 3:
            return
        phys = pos[::-1]
        sid = int(self.s.labels.lookup(phys[None])[0])
        text = self.s.region_label(sid)
        if sid in self.s.region_ids:
            text += "   [所选区域内]"
        for name, (layer, table) in self.point_layers.items():
            if not layer.visible or len(table) == 0:
                continue
            idx = layer.get_value(pos, world=True)
            if idx is not None:
                r = table.iloc[int(idx)]
                text += f"\n● {r['cell_id']}  →  {self.s.region_label(r['region_id'])}"
                break
        viewer.text_overlay.text = text


def _signal_floor(session):
    default = 1.5 if session.source.kind == "tiles" else 1.2
    return float(session.cfg.get("signal_warn_below", default))


def _html(text):
    import html
    return html.escape(str(text))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    local_config.add_config_arg(parser, TOOL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summary", action="store_true",
                      help="只打印选区统计和 site 列表，不读图像")
    mode.add_argument("--snapshot", action="store_true",
                      help="不开窗口，每个 site 出一张 PNG 到 <out_dir>/snapshots/")
    parser.add_argument("--frame-check", action="store_true",
                        help="volume 模式下额外检查细胞坐标和原图的 z 偏移（整体 + 逐 tile，约半分钟）")
    parser.add_argument("--start", type=int, default=0, help="从第几个 site 开始（0 起）")
    args = parser.parse_args()

    cfg = local_config.load_config(TOOL, args.config, required=REQUIRED)
    session = rc.Session(cfg)
    print("\n".join(session.summary_lines()))
    frame_lines = []
    if args.frame_check:
        frame_lines = rc.frame_check_lines(session, cfg.get("frame_check_marker", "RFP"))
        print("\n".join(frame_lines))

    if args.summary:
        cols = ["cell_id", "region_id", "lookup_id", "depth_um"]
        with pd.option_context("display.width", 160, "display.max_rows", 200):
            print(session.sites[cols].assign(region=session.sites["region_id"].map(
                session.region_label)).to_string())
        return
    if args.snapshot:
        run_snapshots(session)
        return

    import napari
    Viewer(session, start=args.start, extra_lines=frame_lines)
    napari.run()


if __name__ == "__main__":
    main()
