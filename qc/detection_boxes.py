"""brain_detector 的分级检测框，放进配准流水线的物理坐标系里。

qc/view_detection_qc.py 的 GUI-free 一半，不单独运行。

为什么需要这个文件
------------------
qc/view_region_cells.py 画的是 cell_registration.csv —— 那是整条链路的**最后
一步**，一个细胞一个点。点画对了只说明归区没错，说明不了这个细胞是怎么来的：
它是 2D 检出的哪几层合出来的？z-link 有没有把两个细胞并成一个？Sox9 是在哪一
步贴上去的？这些全发生在 brain_detector 里，而 brain_detector 自己的
src/utils/visualize.py 能把三级框都画出来，却是**按 tile** 走的，不知道脑区，
也没法问"皮层 L2/3 的 Sox9 判读和 L6 一样吗"。

这个文件补的就是中间那一段：按脑区选位置，把该位置上**每一级**的检测框都拿出
来，和原始 tile、图谱边界、最终细胞点画在同一个框里。

四级，以及它们各自的坐标系
--------------------------
    1_tile_2d_filtered/{tile}_{ch}_result.csv   tile 局部 xy，tile 局部 1-indexed z
    2_global_2d_raw/{ch}_2d_global.csv          全局 xy，全局 1-indexed z
    3_channel_3d/{ch}_3d_tracked.csv            同上，z-link 之后一个细胞一行
    4_colocalization/coloc_result.csv           同上，class 是最终的 marker 组合

s2/s3/s4 已经在全局坐标里，和 cell_registration.csv 的 0-2 列是同一个系，所以
这里默认读 s2 而不是逐 tile 的 s1 —— 同一批检测，省掉一次 tile 偏移换算，也就
少一个能出错的地方。没有 2_global_2d_raw 时才回退到逐 tile 的 s1，那时才用
TileGrid 的偏移：

    global_x = x_tilecsv + tile_x0
    global_z = z_tilecsv - tile_z0        （由 local_z_0idx = global_z - 1 + tile_z0 反解）

物理坐标和仓库里其它工具一致：global_px * cells.voxel_size_um = 微米。

抄而不是 import
---------------
颜色、class 解析、框的栅格化都是照 brain_detector/src/utils/visualize.py 重写
的，理由和 qc/crop_geometry.TileGrid 一样：这个仓库依赖 registration_ants，不
依赖 brain_detector，两台机器上装的东西不一样。抄过来的部分在下面各自标了出处，
改了那边就要回来改这边。

内存
----
2_global_2d_raw 是一个检测框在每一层各一行，整脑几千万行很正常，整个读进来会
吃掉几个 GB。所以这里所有读取都是分块的（read_csv(chunksize=...)）：看图时先
把所有 site 的框算出来，分块读的时候只留落在这些框里的行；对账时分块读、只累加
计数，一行都不留。两种用法的峰值内存都和文件大小无关。
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── 从 brain_detector/src/utils/visualize.py 抄来的显示约定 ───────────────────

CLASS_COLOR = {                       # 按 class 的 base（第一个下划线之前）
    "neuron":  [0.0, 0.55, 1.0, 1.0],
    "glia":    [1.0, 0.55, 0.0, 1.0],
    "nucleus": [0.2, 0.9,  0.2, 1.0],
}
_DEFAULT_CLASS_COLOR = [1.0, 1.0, 1.0, 1.0]

_MARKER_RGB = {                       # coloc 图层按 marker 组合混色
    "rfp":   [1.0,  0.20, 0.20],
    "gfp":   [0.20, 1.0,  0.20],
    "sox9":  [0.0,  0.80, 1.0],
    "olig2": [1.0,  0.20, 1.0],
}

STAGE_FILES = {                       # stage -> (子目录, 文件名模板)
    "s2": ("2_global_2d_raw",  "{ch}_2d_global.csv"),
    "s3": ("3_channel_3d",     "{ch}_3d_tracked.csv"),
    "s4": ("4_colocalization", "coloc_result.csv"),
}
STAGE_LABEL = {
    "s1": "逐 tile 2D（filtered）",
    "s2": "原始 2D（全局）",
    "s3": "z-link 去重后",
    "s4": "共定位后",
}
# brain_detector src/core/channel_stage3.py BOX_COLS
GLOBAL_COLS = ["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]
# brain_detector visualize.py _load_tile_csv_shapes 的位置约定（逐 tile 文件）
TILE_COLS = ["slice_name", "x1", "y1", "x2", "y2", "class", "score", "mean", "z"]

CHUNK_ROWS = 2_000_000


# ── class 字符串 ──────────────────────────────────────────────────────────────

def split_class(class_str):
    """'neuron_GFP_Sox9' -> ('neuron', ['GFP', 'Sox9'])。

    照 brain_detector/src/utils/markers.py 的 clean_markers：丢掉纯数字的伪
    marker。旧结果里通道 id 'GFP_3'（曝光时长后缀）会拆出一个凭空的 '3'，那不
    是一个标记物，跟着它走会多出一整类细胞。
    """
    parts = str(class_str).split("_")
    markers = [m for m in parts[1:] if not m.isdigit()]
    return parts[0], (markers if markers else parts[1:])


def class_markers(class_str):
    return split_class(class_str)[1]


def marker_combo_color(markers):
    """-> (neuron_color, glia_color)，按 marker 混色再归一化到最亮的那一维。"""
    rgbs = [_MARKER_RGB.get(str(m).lower(), [0.7, 0.7, 0.7]) for m in markers]
    if not rgbs:
        rgb = [0.9, 0.9, 0.9]
    else:
        rgb = [sum(c[i] for c in rgbs) / len(rgbs) for i in range(3)]
        mx = max(rgb) or 1.0
        rgb = [min(c / mx, 1.0) for c in rgb]
    return rgb + [1.0], [c * 0.65 for c in rgb] + [1.0]


def class_color(class_str):
    return CLASS_COLOR.get(split_class(class_str)[0], _DEFAULT_CLASS_COLOR)


def coloc_groups(classes):
    """出现过的 marker 组合 -> 图层定义，每个组合一个。

    照 visualize.py `_auto_groups_from_classes`：单 marker 的类不算共定位，不
    出图层；匹配是**超集**（GFP+RFP+Sox9 的细胞也出现在 GFP+RFP 图层里），所以
    几个图层叠起来读出来的是一张 Venn 图，不是互斥的分区。
    """
    seen = {}
    for cls in classes:
        markers = tuple(sorted(class_markers(cls)))
        if len(markers) >= 2:
            seen[markers] = None
    groups = []
    for markers in sorted(seen, key=lambda m: (len(m), m)):
        ncol, gcol = marker_combo_color(markers)
        groups.append({"name": "+".join(markers), "markers": set(m.lower() for m in markers),
                       "neuron_color": ncol, "glia_color": gcol})
    return groups


# ── 读盘 ──────────────────────────────────────────────────────────────────────

def _normalise(df):
    """任意一级的原始表 -> 统一的 x1,y1,x2,y2,z,class(,score,mean)。

    表头对不上就按位置套名字：逐 tile 的文件列序和全局的不一样（class 在
    score 前面），而两边都可能是没有表头的旧文件。
    """
    cols = [str(c).strip() for c in df.columns]
    if not {"x1", "y1", "x2", "y2", "z"}.issubset(set(cols)):
        names = TILE_COLS if df.shape[1] == len(TILE_COLS) else GLOBAL_COLS
        if df.shape[1] < len(names):
            raise ValueError(f"检测表只有 {df.shape[1]} 列，对不上 {names}")
        df = df.iloc[:, :len(names)].copy()
        df.columns = names
    else:
        df = df.copy()
        df.columns = cols
    # float64, not float32.  match_coloc_to_cells() below compares
    # (x1 + x2) / 2 against the cell table for EQUALITY, and a stitched x can
    # reach ~2e4 px, where float32 resolves to ~2e-3 -- right on the rounding
    # boundary that comparison uses.  Downcasting here would invent a few
    # percent of spurious mismatches in the one check that is supposed to be
    # exact.  The readers are chunked, so the extra 16 bytes a row costs
    # nothing at peak.
    out = pd.DataFrame({
        "x1": pd.to_numeric(df["x1"], errors="coerce").astype("float64"),
        "y1": pd.to_numeric(df["y1"], errors="coerce").astype("float64"),
        "x2": pd.to_numeric(df["x2"], errors="coerce").astype("float64"),
        "y2": pd.to_numeric(df["y2"], errors="coerce").astype("float64"),
        "z": pd.to_numeric(df["z"], errors="coerce").astype("float64"),
    })
    out["class"] = df["class"].astype(str) if "class" in df.columns else "unknown"
    for extra in ("score", "mean"):
        if extra in df.columns:
            out[extra] = pd.to_numeric(df[extra], errors="coerce").astype("float32")
    if "tile_name" in df.columns:
        out["tile_name"] = df["tile_name"].astype(str)
    return out.dropna(subset=["x1", "y1", "x2", "y2", "z"])


def _read_chunks(path, chunk_rows=CHUNK_ROWS):
    """分块读一个检测表，每块都已 _normalise。文件不存在就什么都不产出。"""
    if not os.path.isfile(path):
        return
    head = pd.read_csv(path, nrows=0)
    headerless = not {"x1", "y1"}.issubset({str(c).strip() for c in head.columns})
    reader = pd.read_csv(path, chunksize=chunk_rows,
                         header=None if headerless else "infer", low_memory=False)
    for chunk in reader:
        got = _normalise(chunk)
        if len(got):
            yield got


class DetectionRun:
    """一个 brain_detector 结果目录（pATHRESULT）里的各级检测框。

    `channels` 是 {通道 id: tile 根目录} 或只是一串通道 id —— 这里只用到 id，
    tile 目录是 s1 回退时换算偏移用的，由调用方把 qc/crop_geometry.TileGrid
    传进来（`tile_grids`），不在这里自己开。
    """

    def __init__(self, result_dir, channels, cell_voxel_um,
                 stages=("s2", "s3", "s4"), tile_grids=None, log=print):
        self.dir = Path(result_dir)
        # 路径打错和"这个样本的检测什么都没产出"在下面长得一模一样（每一级各一条
        # warning，然后一个框都没有），所以顶层目录在这里硬查一次。
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"detection_dir 不存在：{self.dir}\n"
                "这要指到 brain_detector 的 pATHRESULT，也就是底下有 1_tile_2d_filtered / "
                "2_global_2d_raw / 3_channel_3d / 4_colocalization 的那一层。"
                "检测结果通常和全分辨率 tile 在同一台机器上。")
        stage_dirs = [d for d, _ in STAGE_FILES.values()] + ["1_tile_2d_filtered"]
        if not any((self.dir / d).is_dir() for d in stage_dirs):
            raise FileNotFoundError(
                f"{self.dir} 下一个检测阶段的目录都没有（找的是 {', '.join(stage_dirs)}）。"
                "这不像是 brain_detector 的结果目录。")
        self.channels = list(channels)
        self.voxel = np.asarray(cell_voxel_um, float)
        self.tile_grids = tile_grids or {}
        self.log = log
        self.warnings = []
        self.stages = [s for s in stages if s in ("s1", "s2", "s3", "s4")]
        self.sources = self._resolve_sources()

    # -- 找文件 --------------------------------------------------------------

    def _resolve_sources(self):
        """-> [(stage, channel_or_None, path 或 None)]，None 表示要逐 tile 回退。"""
        out = []
        for stage in self.stages:
            if stage == "s4":
                path = self.dir / STAGE_FILES["s4"][0] / STAGE_FILES["s4"][1]
                if path.is_file():
                    out.append((stage, None, path))
                else:
                    self.warnings.append(f"没有 {path} —— 共定位结果这一级看不到。")
                continue
            if stage == "s1":
                out += [(stage, ch, None) for ch in self.channels]
                continue
            sub, tpl = STAGE_FILES[stage]
            for ch in self.channels:
                path = self.dir / sub / tpl.format(ch=ch)
                if path.is_file():
                    out.append((stage, ch, path))
                elif stage == "s2":
                    # 拼接过的全局 2D 不在，退回逐 tile 的 filtered（stage 3 自己
                    # 读的就是它，所以两者描述的是同一批框）
                    self.warnings.append(
                        f"没有 {path}，[{ch}] 的原始 2D 改从 1_tile_2d_filtered 逐 tile 读。")
                    out.append(("s1", ch, None))
                else:
                    self.warnings.append(f"没有 {path} —— [{ch}] 的 {STAGE_LABEL[stage]} 这一级看不到。")
        return out

    def _tile_files(self, ch):
        """s1 回退：该通道每个 tile 的 filtered CSV -> (path, tile_x0, y0, z0)。"""
        grid = self.tile_grids.get(ch)
        sub = self.dir / "1_tile_2d_filtered"
        if grid is None:
            self.warnings.append(f"[{ch}] 要逐 tile 读 2D，但没有它的 TileGrid（配置里的 channels "
                                 f"要指到拼接 xml 所在那层），跳过。")
            return []
        out = []
        for tile in grid.tiles:
            path = sub / f"{tile['name']}_{ch}_result.csv"
            if path.is_file():
                out.append((path, tile["x0"], tile["y0"], tile["z0"]))
        if not out:
            self.warnings.append(f"[{ch}] 在 {sub} 下一个 tile 的 2D 结果都没找到。")
        return out

    # -- 迭代 ----------------------------------------------------------------

    def iter_boxes(self):
        """逐块产出 (stage, channel, DataFrame)，坐标一律已换算到**全局像素**。"""
        for stage, ch, path in self.sources:
            if stage == "s1":
                for tpath, x0, y0, z0 in self._tile_files(ch):
                    for chunk in _read_chunks(tpath):
                        chunk = chunk.copy()
                        chunk[["x1", "x2"]] += x0
                        chunk[["y1", "y2"]] += y0
                        chunk["z"] -= z0          # local_z_0idx = global_z - 1 + tile_z0
                        yield stage, ch, chunk
            else:
                for chunk in _read_chunks(path):
                    yield stage, ch, chunk

    def to_phys(self, df):
        """全局像素的框 -> 微米。就地改一份拷贝并返回。"""
        out = df.copy()
        out[["x1", "x2"]] = out[["x1", "x2"]].to_numpy(float) * self.voxel[0]
        out[["y1", "y2"]] = out[["y1", "y2"]].to_numpy(float) * self.voxel[1]
        out["zc"] = out["z"].to_numpy(float) * self.voxel[2]
        return out

    # -- 取一批框 ------------------------------------------------------------

    def collect(self, windows_phys):
        """一次分块扫盘，取回落在任意一个窗口里的框。

        `windows_phys` 是 [(lo_xyz, hi_xyz), ...]，微米。一次把所有 site 都取
        完，因为整脑的原始 2D 表扫一遍要几十秒，逐个 site 扫会把这个代价乘上
        site 个数。

        -> {(stage, channel): DataFrame}，坐标是微米，另带 `zc`（框所在层的
           物理 z）和 `window`（第几个窗口）。
        """
        if not windows_phys:
            return {}
        lo = np.array([w[0] for w in windows_phys], float)   # (W, 3)
        hi = np.array([w[1] for w in windows_phys], float)
        acc = {}
        for stage, ch, chunk in self.iter_boxes():
            p = self.to_phys(chunk)
            cx = (p["x1"].to_numpy(float) + p["x2"].to_numpy(float)) / 2
            cy = (p["y1"].to_numpy(float) + p["y2"].to_numpy(float)) / 2
            cz = p["zc"].to_numpy(float)
            # (N, W) 的命中表；site 个数是几十，这一步比逐 site 扫盘便宜得多
            inside = ((cx[:, None] >= lo[None, :, 0]) & (cx[:, None] <= hi[None, :, 0]) &
                      (cy[:, None] >= lo[None, :, 1]) & (cy[:, None] <= hi[None, :, 1]) &
                      (cz[:, None] >= lo[None, :, 2]) & (cz[:, None] <= hi[None, :, 2]))
            hit = inside.any(axis=1)
            if not hit.any():
                continue
            got = p.loc[hit].copy()
            got["window"] = inside[hit].argmax(axis=1)
            key = (stage, ch)
            acc.setdefault(key, []).append(got)
        return {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}


# ── 栅格化成 napari 的 Labels ─────────────────────────────────────────────────

def rasterize(boxes_phys, origin_um, voxel_um, shape_xyz, outline_width=2,
              color_of=None, dash_of=None):
    """框 -> (uint32 (Z, Y, X) 体积, {label_id: RGBA})。

    照 visualize.py `_rasterize_shapes_to_labels`：画的是每个框在它自己那一层
    上的**轮廓**，不是实心块 —— 实心块会把下面的图像盖住，而这里要看的恰好是
    框和图像对不对得上。相同颜色的框共用一个 label id，所以图层的颜色表长度是
    颜色数而不是框数。

    `dash_of(row) -> int`：非 0 就画虚线（visualize.py 用虚线区分 glia）。
    """
    origin_um = np.asarray(origin_um, float)
    voxel_um = np.asarray(voxel_um, float)
    nx, ny, nz = (int(v) for v in shape_xyz)
    vol = np.zeros((nz, ny, nx), np.uint32)
    color_map = {}
    color_to_id = {}
    hw = max(1, int(outline_width) // 2)

    if not len(boxes_phys):
        return vol, color_map

    xs1 = np.rint((boxes_phys["x1"].to_numpy(float) - origin_um[0]) / voxel_um[0]).astype(int)
    xs2 = np.rint((boxes_phys["x2"].to_numpy(float) - origin_um[0]) / voxel_um[0]).astype(int)
    ys1 = np.rint((boxes_phys["y1"].to_numpy(float) - origin_um[1]) / voxel_um[1]).astype(int)
    ys2 = np.rint((boxes_phys["y2"].to_numpy(float) - origin_um[1]) / voxel_um[1]).astype(int)
    zz = np.rint((boxes_phys["zc"].to_numpy(float) - origin_um[2]) / voxel_um[2]).astype(int)

    for i, (_, row) in enumerate(boxes_phys.iterrows()):
        z = int(zz[i])
        if z < 0 or z >= nz:
            continue
        col = tuple(color_of(row) if color_of else class_color(row["class"]))
        if col not in color_to_id:
            color_to_id[col] = len(color_to_id) + 1
            color_map[color_to_id[col]] = list(col)
        lid = color_to_id[col]
        x1, x2 = sorted((int(xs1[i]), int(xs2[i])))
        y1, y2 = sorted((int(ys1[i]), int(ys2[i])))
        x1c, x2c = max(0, x1), min(nx - 1, x2)
        y1c, y2c = max(0, y1), min(ny - 1, y2)
        if x1c > x2c or y1c > y2c:
            continue
        dash = int(dash_of(row)) if dash_of else 0
        for d in range(hw):
            ty, by = np.clip(y1c + d, 0, ny - 1), np.clip(y2c - d, 0, ny - 1)
            lx, rx = np.clip(x1c + d, 0, nx - 1), np.clip(x2c - d, 0, nx - 1)
            if dash > 0:
                dh = max(1, dash // 2)
                xr = np.arange(x1c, x2c + 1)
                xr = xr[(xr // dh) % 2 == 0]
                vol[z, ty, xr] = lid
                vol[z, by, xr] = lid
                yr = np.arange(y1c, y2c + 1)
                yr = yr[(yr // dh) % 2 == 0]
                vol[z, yr, lx] = lid
                vol[z, yr, rx] = lid
            else:
                vol[z, ty, x1c:x2c + 1] = lid
                vol[z, by, x1c:x2c + 1] = lid
                vol[z, y1c:y2c + 1, lx] = lid
                vol[z, y1c:y2c + 1, rx] = lid
    return vol, color_map


# ── 分级对账 ──────────────────────────────────────────────────────────────────

def region_stage_counts(run, labels, ids, log=print):
    """整个所选脑区里，每一级每个类别有多少个框。

    归区一律用 labels_in_sample 回查框中心（labels.lookup），**四级和细胞表都
    用同一个口径**，否则少掉的那几个说不清是真的少了还是换了个查法。细胞表自己
    第 9 列的归区是另一条路（把细胞推进图谱空间查），单独在 compare 里报。

    -> DataFrame: stage, channel, class, n_in_region, n_total
    """
    ids = set(int(i) for i in ids)
    rows = {}
    for stage, ch, chunk in run.iter_boxes():
        p = run.to_phys(chunk)
        cx = (p["x1"].to_numpy(float) + p["x2"].to_numpy(float)) / 2
        cy = (p["y1"].to_numpy(float) + p["y2"].to_numpy(float)) / 2
        phys = np.column_stack([cx, cy, p["zc"].to_numpy(float)])
        found = labels.lookup(phys)
        keep = np.isin(found, list(ids))
        cls = p["class"].to_numpy()
        for name, n_tot in zip(*np.unique(cls, return_counts=True)):
            key = (stage, ch or "-", str(name))
            rows.setdefault(key, [0, 0])[1] += int(n_tot)
        if keep.any():
            for name, n_in in zip(*np.unique(cls[keep], return_counts=True)):
                rows[(stage, ch or "-", str(name))][0] += int(n_in)
    out = pd.DataFrame(
        [{"stage": s, "channel": c, "class": k, "n_in_region": v[0], "n_total": v[1]}
         for (s, c, k), v in rows.items()])
    return out.sort_values(["stage", "channel", "class"]).reset_index(drop=True)


def match_coloc_to_cells(run, cells, decimals=3):
    """coloc_result.csv 和 cell_registration/ 的逐行对账。

    这是整条链路上唯一一处**应该精确相等**的接缝：run_inference.py 写质心时用的
    就是 cx = (x1 + x2) / 2、cy = (y1 + y2) / 2、z 原样，而 cell_points.py 把这
    三个数原样抄进 cell_registration.csv 的 0-2 列（reposition 也不动它们）。
    所以这里对不上不是精度问题，是中间真的丢了行或者换了一版结果。

    -> dict：两边的行数、精确匹配数、各自多出来的行数，以及按类别的缺口。
    """
    key_cols = ["cx", "cy", "z", "class"]
    want = pd.DataFrame({
        "cx": cells["x"].to_numpy(float).round(decimals),
        "cy": cells["y"].to_numpy(float).round(decimals),
        "z": cells["z"].to_numpy(float).round(decimals),
        "class": cells["class_name"].astype(str).to_numpy(),
    })
    got_parts = []
    for stage, _, chunk in run.iter_boxes():
        if stage != "s4":
            continue
        got_parts.append(pd.DataFrame({
            "cx": ((chunk["x1"].to_numpy(float) + chunk["x2"].to_numpy(float)) / 2).round(decimals),
            "cy": ((chunk["y1"].to_numpy(float) + chunk["y2"].to_numpy(float)) / 2).round(decimals),
            "z": chunk["z"].to_numpy(float).round(decimals),
            "class": chunk["class"].astype(str).to_numpy(),
        }))
    if not got_parts:
        return {"available": False}
    got = pd.concat(got_parts, ignore_index=True)

    merged = want.merge(got.assign(_s4=1).drop_duplicates(key_cols),
                        on=key_cols, how="left")
    matched = int(merged["_s4"].notna().sum())
    back = got.merge(want.assign(_c=1).drop_duplicates(key_cols),
                     on=key_cols, how="left")
    by_class = (merged.assign(miss=merged["_s4"].isna())
                .groupby("class")["miss"].agg(["size", "sum"])
                .rename(columns={"size": "n_cells", "sum": "n_unmatched"}))
    return {
        "available": True,
        "n_s4": len(got),
        "n_cells": len(want),
        "n_matched": matched,
        "n_cells_unmatched": len(want) - matched,
        "n_s4_unmatched": int(back["_c"].isna().sum()),
        "by_class": by_class[by_class["n_unmatched"] > 0].sort_values("n_unmatched",
                                                                     ascending=False),
    }


def _sox9_split(counts_by_class):
    """{class: n} -> (Sox9+ 数, 总数)。marker 用 class_markers 解析，所以
    'GFP_3' 拆出来的伪 marker '3' 不会被当成一个标记物。"""
    pos = tot = 0
    for name, n in counts_by_class.items():
        tot += n
        if any(m.lower() == "sox9" for m in class_markers(name)):
            pos += n
    return pos, tot


def funnel_lines(stage_counts, cells_in_region, region_name):
    """region_stage_counts + 细胞表 -> 可以直接打印的漏斗。

    每一级看两个数：这一级有多少，以及 Sox9+ 占多少。占比是这套数据里唯一跨样本
    可比的量（标记效率是动物级的、没有内参），所以它在哪一级变的，就是该去查哪
    一级。
    """
    lines = [f"检测分级漏斗 —— {region_name}（归区口径：labels_in_sample 回查框中心）", ""]
    if len(stage_counts):
        for stage in [s for s in ("s1", "s2", "s3", "s4") if s in set(stage_counts["stage"])]:
            sub = stage_counts[stage_counts["stage"] == stage]
            head = f"  {stage}  {STAGE_LABEL[stage]}"
            per_ch = sub.groupby("channel")["n_in_region"].sum()
            if stage == "s4":
                counts = sub.groupby("class")["n_in_region"].sum().to_dict()
                pos, tot = _sox9_split(counts)
                lines.append(f"{head}：{tot:,}")
                for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                    lines.append(f"        {name:30s} {n:9,}")
                if tot:
                    lines.append(f"        {'Sox9+ 占比':30s} {100 * pos / tot:8.2f}%")
            else:
                body = "   ".join(f"{c} {int(n):,}" for c, n in per_ch.items())
                lines.append(f"{head}：{body}")
        s2 = stage_counts[stage_counts["stage"].isin(("s1", "s2"))]
        s3 = stage_counts[stage_counts["stage"] == "s3"]
        if len(s2) and len(s3):
            lines.append("")
            lines.append("  z-link 压缩比（2D 框·层 ÷ 细胞，≈ 每个细胞跨几层）：")
            a = s2.groupby("channel")["n_in_region"].sum()
            b = s3.groupby("channel")["n_in_region"].sum()
            for ch in sorted(set(a.index) & set(b.index)):
                if b[ch]:
                    lines.append(f"        {ch:10s} {a[ch] / b[ch]:6.2f}")
    lines.append("")
    counts = cells_in_region.groupby("class_name").size().to_dict()
    pos, tot = _sox9_split(counts)
    lines.append(f"  细胞表 cell_registration/（第 9 列归区）：{tot:,}")
    if tot:
        lines.append(f"        {'Sox9+ 占比':30s} {100 * pos / tot:8.2f}%")
    return lines


def match_lines(match):
    if not match.get("available"):
        return ["s4 → cell_registration 对账：没有 coloc_result.csv，做不了。"]
    n_c, n_m = match["n_cells"], match["n_matched"]
    pct = 100 * n_m / n_c if n_c else float("nan")
    lines = [
        "s4 → cell_registration 对账（(x1+x2)/2, (y1+y2)/2, z, class 精确相等）：",
        f"  coloc_result.csv {match['n_s4']:,} 行，细胞表 {n_c:,} 行，匹配上 {n_m:,}（{pct:.2f}%）",
    ]
    if match["n_cells_unmatched"] or match["n_s4_unmatched"]:
        lines.append(f"  细胞表里没有对应 s4 行的 {match['n_cells_unmatched']:,}；"
                     f"s4 里没进细胞表的 {match['n_s4_unmatched']:,}")
        lines.append("  ⚠️ 这一步本该精确相等。对不上说明细胞表和这份 coloc_result.csv "
                     "不是同一次跑出来的，或者中间丢了行 —— 先查清楚，"
                     "再看任何按类别/按区的统计。")
        if len(match["by_class"]):
            lines.append("  缺口最大的类别：")
            for name, r in match["by_class"].head(8).iterrows():
                lines.append(f"        {name:30s} {int(r.n_unmatched):9,} / {int(r.n_cells):,}")
    else:
        lines.append("  ✅ 完全一致 —— 从检测到细胞表这一段没有丢东西，"
                     "类别比例的问题只可能在检测内部（s2/s3/s4）或配准归区。")
    return lines
