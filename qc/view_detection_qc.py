#!/usr/bin/env python
"""全链路质控：按图谱脑区选位置，把原始 tile、**每一级检测框**、图谱边界和最终
细胞表画在同一个框里。

GUI-free 的一半是 qc/detection_boxes.py（读盘、坐标换算、对账），选点和归区那
一半复用 qc/region_cells.py —— 也就是 qc/view_region_cells.py 用的同一套。

和 qc/view_region_cells.py 的区别
--------------------------------
那个工具画的是 cell_registration.csv，一个细胞一个点：能回答"这个细胞归区对不
对"，回答不了"这个细胞是怎么被数出来的"。这个工具把 brain_detector 的中间产物
也铺开：

    [s2] <ch>          原始 2D 检测，一个框一层
    [s3] <ch>          z-link 去重之后，一个细胞一个框
    [coloc] GFP+Sox9   共定位判定之后的最终类别
    region cells       cell_registration.csv，统计真正数的那一批

所以一个 Sox9+ 的细胞在这里能一路看下来：730 通道上到底有没有核、2D 检出了没、
z-link 有没有把它并掉、共定位有没有把它贴到 soma 上、最后有没有进细胞表。

四种跑法（antsreg 环境，仓库根目录）
------------------------------------
    python qc/view_detection_qc.py --funnel      # 数字：分级漏斗 + s4/细胞表对账
    python qc/view_detection_qc.py --summary     # 选了哪些 site，不读图
    python qc/view_detection_qc.py --snapshot    # 每个 site 一张 PNG，不开窗口
    python qc/view_detection_qc.py               # napari 逐个看、逐个判定

    python qc/view_detection_qc.py configs/detection_qc.s8.yaml

**先跑 --funnel。** 它不读一张图，几分钟内回答"类别比例是在哪一级变的"，而逐个
看图只能回答"这一个细胞对不对"。看图是用来解释漏斗里那个跳变的，不是用来发现它的。

窗口里
------
    n / p      下一个 / 上一个 site
    u          跳到下一个没判过的
    1-5        判定（见 VERDICTS），判完自动下一个
    hover      光标下的脑区，以及光标下的检测框属于哪一级、哪个类

判定写进 <out_dir>/verdicts.csv，按 run_dir + 细胞 id 记，可以关掉再接着判。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qc import detection_boxes as db  # noqa: E402
from qc import region_cells as rc  # noqa: E402
from qc.view_region_cells import (_boundaries_2d, _channel_colormap, _contrast,  # noqa: E402
                                  _html, _point_style)
from shared import local_config  # noqa: E402

TOOL = "detection_qc"
REQUIRED = ("run_dir", "ontology_json", "regions", "out_dir", "detection_dir")

# 这个工具要判的东西和归区质控不一样，所以有自己的词表（rc.VerdictStore 接受它）。
VERDICTS = {
    "ok":          "都对",
    "over_merge":  "该分没分（z-link 过并）",
    "under_merge": "该并没并（一个细胞数成多个）",
    "sox9_wrong":  "Sox9 判错",
    "not_cell":    "不是细胞 / 看不清",
}

STAGE_VISIBLE = {"s1": False, "s2": False, "s3": True, "s4": True}


# ── Session ───────────────────────────────────────────────────────────────────

class DetectionSession(rc.Session):
    """rc.Session 再加上 brain_detector 的各级检测框。

    所有 site 的框是**一次**扫盘取完的（db.DetectionRun.collect）：整脑的原始
    2D 表有几千万行，逐个 site 去扫会把那几十秒乘上 site 个数。
    """

    def __init__(self, cfg, log=print, load_boxes=True):
        super().__init__(cfg, log=log)
        grids = getattr(self.source, "grids", {})     # tiles 模式下才有
        self.run = db.DetectionRun(
            cfg["detection_dir"],
            cfg.get("detection_channels") or list(grids) or ["RFP", "GFP", "Sox9"],
            self.cell_voxel_um,
            stages=tuple(cfg.get("stages", ("s2", "s3", "s4"))),
            tile_grids=grids, log=log)
        self.warnings += self.run.warnings
        self.boxes = {}
        self.groups = []
        if load_boxes and len(self.sites):
            self.load_boxes()

    def locate_cell(self, query):
        """Resolve an exact cell id or the nearest registered cell to global xyz.

        Distance is measured in physical microns, not anisotropic pixels.
        Searches all registered cells, independently of the region filter.
        """
        query = query.strip()
        exact = self.cells.index[self.cells["cell_id"] == query]
        distance = 0.0
        if len(exact):
            idx = int(exact[0])
        else:
            try:
                xyz = np.asarray([float(x) for x in query.replace(",", " ").split()])
            except ValueError:
                raise ValueError("请输入完整 cell_id，或全局像素 x, y, z") from None
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError("请输入完整 cell_id，或三个有限的全局像素坐标")
            distances = np.linalg.norm(self.phys - xyz * self.cell_voxel_um, axis=1)
            idx = int(np.argmin(distances))
            distance = float(distances[idx])
        row = self.cells.iloc[idx].copy()
        existing = np.flatnonzero(self.sites["cell_id"].to_numpy() == row["cell_id"])
        if len(existing):
            return int(existing[0]), distance
        phys = self.phys[idx:idx + 1]
        row["lookup_id"] = int(self.labels.lookup(phys)[0])
        row["depth_um"] = float(rc.signed_depth_um(self.labels, self.region_ids, phys)[0])
        row["lookup_agrees"] = row["lookup_id"] in self.region_ids
        k = len(self.sites)
        # Read only the new window; leave existing site caches intact.
        extra = self.run.collect([(phys[0] - self.half_um, phys[0] + self.half_um)])
        for key, frame in extra.items():
            frame = frame.assign(window=k)
            self.boxes[key] = pd.concat([self.boxes[key], frame], ignore_index=True) \
                if key in self.boxes else frame
        self.sites = pd.concat([self.sites, row.to_frame().T], ignore_index=True)
        s4 = self.boxes.get(("s4", None))
        self.groups = db.coloc_groups(s4["class"].unique()) if s4 is not None else []
        return k, distance

    def source_locations(self, k):
        xyz = self.sites.iloc[k][["x", "y", "z"]].to_numpy(float)
        return {ch: grid.locate(xyz)
                for ch, grid in getattr(self.source, "grids", {}).items()}

    def windows(self):
        return [(self.site_phys(k) - self.half_um, self.site_phys(k) + self.half_um)
                for k in range(len(self.sites))]

    def load_boxes(self):
        self.log(f"扫描 {self.run.dir} 的各级检测框（{len(self.sites)} 个 site 一次取完）...")
        self.boxes = self.run.collect(self.windows())
        s4 = self.boxes.get(("s4", None))
        self.groups = db.coloc_groups(s4["class"].unique()) if s4 is not None else []
        for (stage, ch), df in sorted(self.boxes.items(), key=lambda kv: str(kv[0])):
            self.log(f"  {stage} {ch or '-':6s} {len(df):8,} 个框")

    def site_boxes(self, k):
        """-> [(layer_name, DataFrame, color_of, dash_of, visible)]，这个 site 的。

        coloc 那一级按 marker 组合拆成几层，和 visualize.py 一样用**超集**匹配：
        GFP+RFP+Sox9 的细胞同时出现在 GFP+RFP 层和 GFP+Sox9 层里。几层叠起来是
        一张 Venn 图，不是互斥分区 —— 各层的框加起来会超过细胞数，这是对的。
        """
        out = []
        for (stage, ch), df in sorted(self.boxes.items(), key=lambda kv: str(kv[0])):
            sub = df[df["window"] == k]
            if not len(sub):
                continue
            if stage == "s4":
                for grp in self.groups:
                    keep = sub["class"].map(
                        lambda c, g=grp: g["markers"].issubset(
                            {m.lower() for m in db.class_markers(c)}))
                    if not keep.any():
                        continue
                    out.append((
                        f"[coloc] {grp['name']}", sub[keep],
                        lambda r, g=grp: (g["glia_color"]
                                          if db.split_class(r["class"])[0] == "glia"
                                          else g["neuron_color"]),
                        lambda r: 8 if db.split_class(r["class"])[0] == "glia" else 0,
                        STAGE_VISIBLE["s4"]))
            else:
                out.append((f"[{stage}] {ch}", sub, None, None, STAGE_VISIBLE[stage]))
        return out

    def stage_tally(self, k):
        return {f"{stage} {ch or ''}".strip(): int((df["window"] == k).sum())
                for (stage, ch), df in sorted(self.boxes.items(), key=lambda kv: str(kv[0]))}

    # -- 数字模式 ------------------------------------------------------------

    def funnel(self):
        """分级漏斗 + s4/细胞表对账，两趟扫盘，一张图都不读。"""
        name = ", ".join(f"{a} ({n})" for _, n, a in self.region_roots)
        counts = db.region_stage_counts(self.run, self.labels, self.region_ids, log=self.log)
        sel_all = rc.select_cells(self.cells, self.region_ids)   # 不按 classes 过滤
        lines = db.funnel_lines(counts, sel_all, name)
        lines += [""] + db.match_lines(db.match_coloc_to_cells(self.run, self.cells))
        return counts, lines


# ── Headless PNG ──────────────────────────────────────────────────────────────

_STAGE_STYLE = {"s1": (":", 0.8), "s2": (":", 0.8), "s3": ("-", 1.0), "s4": ("--", 1.6)}


def render_snapshot(session, k, cut, path, mip_um=None):
    """一个 site 一张 PNG：z 方向最大投影的图像 + 投影厚度内所有级别的框。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    box = cut["box"]
    vox = box.voxel_um
    centre = box.local(cut["center"])[0]
    zc = int(np.clip(np.rint(centre[2]), 0, box.shape_xyz[2] - 1))
    if mip_um is None:
        mip_um = max(vox[2], 16.0) if box.source == "tiles" else 0.0
    half = int(np.floor(mip_um / vox[2]))
    z0, z1 = max(0, zc - half), min(box.shape_xyz[2], zc + half + 1)

    tints = {"red": (1, 0, 0), "green": (0, 1, 0), "cyan": (0, 1, 1),
             "magenta": (1, 0, 1), "gray": (1, 1, 1)}
    rgb = np.zeros((box.shape_xyz[1], box.shape_xyz[0], 3), float)
    for name, arr in box.arrays.items():
        lo, hi = _contrast(arr)
        img = np.clip((arr[z0:z1].max(0).astype(float) - lo) / (hi - lo), 0, 1)
        rgb += img[..., None] * np.array(tints[_channel_colormap(name, len(box.arrays))])
    rgb = np.clip(rgb, 0, 1)

    plane_origin = box.origin_um.copy()
    plane_origin[2] = box.origin_um[2] + zc * vox[2]
    lab = session.labels.resample_to(plane_origin, vox,
                                     [box.shape_xyz[0], box.shape_xyz[1], 1])[:, :, 0].T
    sel = np.isin(lab, list(session.region_ids))

    ext = [box.origin_um[0], box.origin_um[0] + box.shape_xyz[0] * vox[0],
           box.origin_um[1] + box.shape_xyz[1] * vox[1], box.origin_um[1]]
    fig, ax = plt.subplots(figsize=(10, 10 * rgb.shape[0] / rgb.shape[1] + 1.6), dpi=110)
    ax.imshow(rgb, extent=ext, interpolation="nearest")
    overlay = np.zeros(lab.shape + (4,))
    overlay[_boundaries_2d(lab)] = (1, 1, 1, 0.30)
    ax.imshow(overlay, extent=ext, interpolation="nearest")
    if sel.any() and not sel.all():
        xs = box.origin_um[0] + np.arange(lab.shape[1]) * vox[0]
        ys = box.origin_um[1] + np.arange(lab.shape[0]) * vox[1]
        ax.contour(xs, ys, sel.astype(float), levels=[0.5], colors=["yellow"], linewidths=1.4)

    zlo = box.origin_um[2] + (z0 - 0.5) * vox[2]
    zhi = box.origin_um[2] + (z1 - 0.5) * vox[2]
    drawn = []
    for layer_name, sub, color_of, _dash, _vis in session.site_boxes(k):
        stage = layer_name.split("]")[0].strip("[")
        stage = "s4" if stage == "coloc" else stage
        style, lw = _STAGE_STYLE.get(stage, ("-", 1.0))
        slab = sub[(sub["zc"] >= zlo) & (sub["zc"] < zhi)]
        if not len(slab):
            continue
        for _, r in slab.iterrows():
            col = color_of(r) if color_of else db.class_color(r["class"])
            ax.add_patch(Rectangle((min(r.x1, r.x2), min(r.y1, r.y2)),
                                   abs(r.x2 - r.x1), abs(r.y2 - r.y1),
                                   fill=False, edgecolor=col, linewidth=lw,
                                   linestyle=style))
        drawn.append(f"{layer_name} {len(slab)}")

    cells = cut["cells"]
    slab = cells[(cells["pz"] >= zlo) & (cells["pz"] < zhi)]
    if len(slab):
        colors, symbols = _point_style(slab)
        for sym, marker in (("disc", "o"), ("diamond", "D")):
            m = symbols == sym
            if m.any():
                ax.scatter(slab["px"][m], slab["py"][m], s=70, marker=marker,
                           facecolors="none", edgecolors=colors[m], linewidths=1.0)
    c = cut["center"]
    ax.scatter([c[0]], [c[1]], s=420, marker="o", facecolors="none",
               edgecolors="white", linewidths=1.8)
    ax.axhline(c[1], color="white", lw=0.4, alpha=0.5)
    ax.axvline(c[0], color="white", lw=0.4, alpha=0.5)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

    info = session.describe_site(k)
    ax.set_title(
        f"site {k}  {info['cell_id']}   px ({info['xyz_px']})\n"
        f"table: {info['table_region']}\n"
        f"labels: {info['labels_region']}   depth {info['depth']}   "
        f"signal {cut['signal_ratio']:.2f}   z slab {mip_um:.0f} µm\n"
        # English here, not Chinese: matplotlib's default font has no CJK
        # glyphs, so a Chinese title renders as boxes on a machine without one
        # installed.  The Qt panel is fine and stays Chinese.
        f"boxes: {'   '.join(drawn) if drawn else '(none in this slab)'}   "
        f"[s2 dotted / s3 solid / coloc dashed]",
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
        tally = session.stage_tally(k)
        print(f"  {name}  signal {cut['signal_ratio']:5.2f}  "
              + "  ".join(f"{k2} {v}" for k2, v in tally.items()))
        rows.append({**info, "signal_ratio": cut["signal_ratio"],
                     "n_cells_in_box": len(cut["cells"]), **tally, "png": name})
    out = snap_dir / "sites.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n{len(rows)} 张 -> {snap_dir}\n清单 -> {out}")


# ── napari ────────────────────────────────────────────────────────────────────

class Viewer:
    def __init__(self, session, start=0, extra_lines=()):
        import napari
        from concurrent.futures import ThreadPoolExecutor
        from qtpy.QtWidgets import (QCheckBox, QGridLayout, QHBoxLayout, QLabel,
                                    QLineEdit, QPushButton, QVBoxLayout, QWidget)

        self.s = session
        self.store = rc.VerdictStore(session.out_dir / "verdicts.csv", session.run_dir,
                                     vocab=VERDICTS)
        self.k = None
        self.cut = None
        self.box_layers = {}
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.viewer = napari.Viewer(title=f"detection QC — {session.run_dir.name}")

        panel = QWidget()
        lay = QVBoxLayout(panel)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(1)
        lay.addWidget(self.info)

        nav = QHBoxLayout()
        for text, fn in (("◀ p", self.prev), ("n ▶", self.next), ("未判 u", self.next_open)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            nav.addWidget(b)
        lay.addLayout(nav)

        self.location = QLineEdit()
        self.location.setPlaceholderText("cell_id 或全局像素 x, y, z（定位最近细胞）")
        self.location.returnPressed.connect(self.locate)
        lay.addWidget(self.location)
        jump = QPushButton("定位细胞 / 回看原图")
        jump.clicked.connect(self.locate)
        lay.addWidget(jump)
        self.location_info = QLabel()
        self.location_info.setWordWrap(True)
        lay.addWidget(self.location_info)
        from qtpy.QtWidgets import QTextEdit
        self.sources = QTextEdit()
        self.sources.setReadOnly(True)
        self.sources.setMaximumHeight(140)
        lay.addWidget(self.sources)

        grid = QGridLayout()
        for i, (key, label) in enumerate(VERDICTS.items()):
            b = QPushButton(f"{i + 1}  {label}")
            b.clicked.connect(lambda _=False, v=key: self.judge(v))
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid)
        self.note = QLineEdit()
        self.note.setPlaceholderText("备注（随判定一起保存）")
        lay.addWidget(self.note)

        self.stage_boxes = {}
        for stage in ("s1", "s2", "s3", "s4"):
            cb = QCheckBox({"s1": "[s1] 逐 tile 2D", "s2": "[s2] 原始 2D",
                            "s3": "[s3] z-link 后", "s4": "[coloc] 共定位"}[stage])
            cb.setChecked(STAGE_VISIBLE[stage])
            cb.stateChanged.connect(lambda _=0: self._apply_visibility())
            lay.addWidget(cb)
            self.stage_boxes[stage] = cb
        self.show_other = QCheckBox("显示其它脑区的细胞（灰）")
        self.show_other.setChecked(True)
        self.show_other.stateChanged.connect(lambda _=0: self._apply_visibility())
        lay.addWidget(self.show_other)

        shot = QPushButton("保存截图")
        shot.clicked.connect(self.screenshot)
        lay.addWidget(shot)
        self.tally = QLabel()
        lay.addWidget(self.tally)
        summary = QLabel("<br>".join(_html(x) for x in
                                     list(session.summary_lines()) + list(extra_lines)))
        summary.setWordWrap(True)
        summary.setTextFormat(1)
        summary.setStyleSheet("font-size: 10px; color: #aaa;")
        lay.addWidget(summary)
        lay.addStretch(1)
        self.viewer.window.add_dock_widget(panel, name="detection QC", area="right")

        v = self.viewer
        v.bind_key("n", lambda _v: self.next(), overwrite=True)
        v.bind_key("p", lambda _v: self.prev(), overwrite=True)
        v.bind_key("u", lambda _v: self.next_open(), overwrite=True)
        for i, key in enumerate(VERDICTS):
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
        self.location_info.clear()
        self.viewer.status = f"loading site {k} ..."
        self.k, self.cut = k, self._fetch(k)
        self._draw()
        self._prefetch(k + 1)

    def locate(self):
        from qtpy.QtWidgets import QMessageBox
        try:
            k, distance = self.s.locate_cell(self.location.text())
            self.show(k)
            self.location_info.setText(
                f"定位到 {self.s.sites.iloc[k]['cell_id']}；距输入坐标 {distance:.2f} µm")
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self.viewer.window._qt_window, "定位失败", str(exc))

    def next(self):
        if self.k is not None:
            self.show(self.k + 1)

    def prev(self):
        if self.k is not None:
            self.show(self.k - 1)

    def next_open(self):
        n = len(self.s.sites)
        if self.k is None:
            return
        for j in list(range(self.k + 1, n)) + list(range(0, self.k + 1)):
            if self.store.get(self.s.sites.iloc[j]["cell_id"])[0] is None:
                self.show(j)
                return
        self.viewer.status = "全部 site 都判过了"

    def judge(self, verdict):
        if self.k is None:
            return
        self.store.set(self.s.sites.iloc[self.k], verdict, self.note.text().strip(),
                       self.s.source.kind)
        self.viewer.status = f"site {self.k}: {VERDICTS[verdict]}"
        if self.k < len(self.s.sites) - 1:
            self.next()
        else:
            self._update_info()

    def screenshot(self):
        if self.k is None:
            return
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
        self.box_layers = {}
        scale = tuple(box.voxel_um[::-1])
        translate = tuple(box.origin_um[::-1])
        for name, arr in box.arrays.items():
            v.add_image(arr, name=name, scale=scale, translate=translate,
                        colormap=_channel_colormap(name, len(box.arrays)),
                        blending="additive", contrast_limits=_contrast(arr))

        lab = cut["labels"].transpose(2, 1, 0)
        lab_scale = tuple(self.s.labels.spacing[::-1])
        lab_translate = tuple(cut["labels_origin"][::-1])
        all_lab = v.add_labels(lab.astype(np.int32), name="all regions",
                               scale=lab_scale, translate=lab_translate, opacity=0.35)
        all_lab.contour = 1
        sel = v.add_labels(np.isin(lab, list(self.s.region_ids)).astype(np.uint8),
                           name="selected region", scale=lab_scale, translate=lab_translate,
                           opacity=0.9,
                           colormap=DirectLabelColormap(
                               color_dict={1: "yellow", None: "transparent"}))
        sel.contour = 2

        # 检测框：每一级栅格化成一个 Labels 层，画在图像自己的网格上
        for layer_name, sub, color_of, dash_of, visible in self.s.site_boxes(self.k):
            vol, cmap = db.rasterize(sub, box.origin_um, box.voxel_um, box.shape_xyz,
                                     outline_width=self.s.cfg.get("outline_width", 2),
                                     color_of=color_of, dash_of=dash_of)
            if not cmap:
                continue
            colors = {int(i): tuple(c) for i, c in cmap.items()}
            colors[0] = (0, 0, 0, 0)
            layer = v.add_labels(vol, name=layer_name, scale=scale, translate=translate,
                                 opacity=0.9, visible=visible,
                                 colormap=DirectLabelColormap(
                                     color_dict={**colors, None: "transparent"}))
            stage = layer_name.split("]")[0].strip("[")
            self.box_layers[layer_name] = (layer, "s4" if stage == "coloc" else stage, sub)

        cells = cut["cells"].reset_index(drop=True)
        local = box.local(cells[["px", "py", "pz"]].to_numpy()) if len(cells) else np.zeros((0, 3))
        if len(cells):
            local[:, 2] = np.rint(local[:, 2])
        diam = (12.0 if box.source == "tiles" else 20.0) / box.voxel_um[0]
        self.point_layers = {}
        for name, m in (("other cells", ~cells["selected"].to_numpy()),
                        ("region cells", cells["selected"].to_numpy())):
            subc = cells[m]
            colors, symbols = _point_style(subc) if len(subc) else ("gray", "disc")
            layer = v.add_points(local[m][:, ::-1], name=name, size=diam, symbol=symbols,
                                 face_color="transparent", border_color=colors,
                                 border_width=0.12, scale=scale, translate=translate,
                                 features=subc.reset_index(drop=True)[
                                     ["cell_id", "class_name", "region_id"]])
            self.point_layers[name] = (layer, subc.reset_index(drop=True))

        c = box.local(cut["center"])[0]
        c[2] = np.rint(c[2])
        target = v.add_shapes(
            [np.array([[c[2], c[1] - 3 * diam, c[0]], [c[2], c[1] + 3 * diam, c[0]]]),
             np.array([[c[2], c[1], c[0] - 3 * diam], [c[2], c[1], c[0] + 3 * diam]])],
            shape_type="line", edge_color="white", edge_width=diam * 0.08,
            name="target", scale=scale, translate=translate)
        target.editable = False

        self._apply_visibility()
        world = box.origin_um[::-1] + c[::-1] * box.voxel_um[::-1]
        v.dims.ndisplay = 2
        v.dims.set_point(0, world[0])
        v.camera.center = tuple(world)
        v.reset_view()
        v.camera.center = tuple(world)
        self._update_info()

    def _apply_visibility(self):
        for layer, stage, _ in self.box_layers.values():
            cb = self.stage_boxes.get(stage)
            layer.visible = cb.isChecked() if cb else True
        if getattr(self, "point_layers", None):
            self.point_layers["other cells"][0].visible = self.show_other.isChecked()

    def _update_info(self):
        d = self.s.describe_site(self.k)
        verdict, note = self.store.get(d["cell_id"])
        self.note.setText(note)
        ratio = self.cut["signal_ratio"]
        floor = float(self.s.cfg.get("signal_warn_below",
                                     1.5 if self.s.source.kind == "tiles" else 1.2))
        warn = "" if np.isfinite(ratio) and ratio >= floor else \
            f" <b style='color:#f66'>← 低于 {floor:g}，先查坐标系</b>"
        agree = "一致" if d["agrees"] else "<b style='color:#fa0'>不一致</b>"
        tally = self.s.stage_tally(self.k)
        stages = "，".join(f"{k} {v}" for k, v in tally.items() if v) or "—"
        self.info.setText(
            f"<b>site {d['site'] + 1} / {len(self.s.sites)}</b>　{_html(d['cell_id'])}<br>"
            f"原始像素 ({d['xyz_px']})　tile {_html(str(d['tile']))}<br>"
            f"细胞表：{_html(d['table_region'])}<br>"
            f"标签图：{_html(d['labels_region'])}（{agree}）<br>"
            f"离所选区域边界 <b>{d['depth']}</b>（正 = 区内）<br>"
            f"框内细胞 {len(self.cut['cells'])}，"
            f"其中所选区域 {int(self.cut['cells']['selected'].sum())}<br>"
            f"框内各级：{_html(stages)}<br>"
            f"signal ratio {ratio:.2f}{warn}<br>"
            f"判定：<b>{VERDICTS.get(verdict, '—') if verdict else '—'}</b>")
        self.tally.setText("已判：" + self.store.tally_text())
        source_lines = []
        try:
            for channel, hits in self.s.source_locations(self.k).items():
                source_lines.append(f"{channel}: {len(hits)} 个原始 tile 覆盖目标")
                for hit in hits:
                    source_lines.append(f"  {hit['path']}\n  tile 局部 xyz (0 起): {hit['local_xyz']}")
        except (OSError, ValueError, RuntimeError) as exc:
            source_lines.append(f"原始文件定位失败：{exc}")
        self.sources.setPlainText("\n".join(source_lines) or "原始文件追溯需要 source: tiles")

    def _on_move(self, viewer, event):
        pos = np.asarray(viewer.cursor.position, float)
        if pos.size != 3:
            return
        phys = pos[::-1]
        sid = int(self.s.labels.lookup(phys[None])[0])
        text = self.s.region_label(sid)
        if sid in self.s.region_ids:
            text += "   [所选区域内]"
        for name, (layer, table) in getattr(self, "point_layers", {}).items():
            if not layer.visible or len(table) == 0:
                continue
            idx = layer.get_value(pos, world=True)
            if idx is not None:
                r = table.iloc[int(idx)]
                text += f"\n● {r['cell_id']}  →  {self.s.region_label(r['region_id'])}"
                break
        # 光标落在哪个检测框里（框是轮廓，命中判定用的是框的范围本身，不是画出来的线）
        hits = []
        for layer_name, (layer, _stage, sub) in self.box_layers.items():
            if not layer.visible or not len(sub):
                continue
            m = ((sub[["x1", "x2"]].min(axis=1) <= phys[0]) &
                 (sub[["x1", "x2"]].max(axis=1) >= phys[0]) &
                 (sub[["y1", "y2"]].min(axis=1) <= phys[1]) &
                 (sub[["y1", "y2"]].max(axis=1) >= phys[1]) &
                 (np.abs(sub["zc"] - phys[2]) <= self.s.cell_voxel_um[2] / 2))
            for cls in sub.loc[m, "class"].unique()[:2]:
                hits.append(f"{layer_name} {cls}")
        if hits:
            text += "\n□ " + "   ".join(hits[:4])
        viewer.text_overlay.text = text


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    local_config.add_config_arg(parser, TOOL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--funnel", action="store_true",
                      help="分级漏斗 + s4/细胞表对账，不读图像（先跑这个）")
    mode.add_argument("--summary", action="store_true", help="只打印选中的 site，不读图像")
    mode.add_argument("--snapshot", action="store_true",
                      help="不开窗口，每个 site 一张 PNG 到 <out_dir>/snapshots/")
    parser.add_argument("--frame-check", action="store_true",
                        help="volume 模式下额外检查细胞坐标和原图的 z 偏移")
    parser.add_argument("--start", type=int, default=0, help="从第几个 site 开始（0 起）")
    args = parser.parse_args()

    cfg = local_config.load_config(TOOL, args.config, required=REQUIRED)
    session = DetectionSession(cfg, load_boxes=not (args.funnel or args.summary))
    print("\n".join(session.summary_lines()))

    if args.funnel:
        counts, lines = session.funnel()
        print()
        print("\n".join(lines))
        session.out_dir.mkdir(parents=True, exist_ok=True)
        out = session.out_dir / "stage_counts.csv"
        counts.to_csv(out, index=False)
        (session.out_dir / "funnel.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"\n逐类别计数 -> {out}\n这段文字 -> {session.out_dir / 'funnel.txt'}")
        return

    frame_lines = []
    if args.frame_check:
        frame_lines = rc.frame_check_lines(session, cfg.get("frame_check_marker", "RFP"))
        print("\n".join(frame_lines))

    if args.summary:
        cols = ["cell_id", "region_id", "lookup_id", "depth_um"]
        with pd.option_context("display.width", 160, "display.max_rows", 200):
            print(session.sites[cols].assign(
                region=session.sites["region_id"].map(session.region_label)).to_string())
        return
    if args.snapshot:
        run_snapshots(session)
        return

    import napari
    Viewer(session, start=args.start, extra_lines=frame_lines)
    napari.run()


if __name__ == "__main__":
    main()
