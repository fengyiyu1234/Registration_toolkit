#DISPLAY=:1 python single_sample.py
import argparse
import napari
import pandas as pd
import numpy as np
import tifffile
import SimpleITK as sitk
import os
import glob
import json
import re
import html


from PyQt5.QtWidgets import (QComboBox, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFrame,
                             QCheckBox, QLineEdit, QPushButton, QDoubleSpinBox, QScrollArea,
                             QFileDialog, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt

from shared import atlas_reference   # check_label_dtype (lossy label dtypes)
from shared import local_config      # configs/<tool>.yaml
from shared import ontology_tree_ui  # shrinkable / set_dock_width

# ================= ⚙️ 用户配置区域 =================
#
# 路径不再写死在这里，而是放在 configs/single_sample.yaml（gitignored，模板是
# configs/single_sample.example.yaml）—— 换样本只改 yaml，不会变成 git diff。
# main() 里就地填充这个 dict，下面各处 CONFIG[...] 的读法保持不变。

CONFIG = {}

_REQUIRED_KEYS = ("sample_dir", "std_atlas_path", "ontology_json_path")

# ================= 🧠 1. JSON 脑区层级管理器 =================

# 侧边面板的【起始】宽度（px），不是上限 —— 面板都是 ontology_tree_ui.shrinkable
# 的，两条边都能拖。以前这里写的是 setMaximumWidth(320)，结果就是脑区名一长
# 就只能看见开头几个字、还拖不宽。
PANEL_START_WIDTH_PX = 340


class OntologyManager:
    """脑区字典，两套编号各建一套索引。

    标签体积（标准图谱标签图、warp 回样本空间的标签图）里存的一律是 atlas 原始
    id；细胞表 cell_registration.csv 第 9 列存的却分两种情况 ——
      ClearMap  : graph_order（cellMap.py 里 convert_label(..., value='graph_order')）
      ANTs      : atlas 原始 id（registration_ants/cell_points.py 的列说明）
    两套编号的数值大面积重合（本机 CCF v3 的 1327 个脑区里有 1126 个数值既是某个
    id 又是某个 graph_order），所以不能像以前那样塞进同一张表按数值查名字 —— 那
    等于让 ClearMap 的细胞去认 id 空间的名字，张冠李戴且完全静默。查询接口统一带
    一个 key 参数说明数值属于哪套编号，默认 'id'（= 标签体积的编号空间）。
    """
    def __init__(self, json_path):
        # id 空间：atlas 标注文件里的原始 id
        self.id_to_name = {}
        self.name_to_id = {}
        # region_id → [(name, acronym), ...]，从根节点到该脑区自身的完整层级链，
        # 用来在 hover 时把"这是哪一级"说清楚。
        self.id_to_path = {}
        # graph_order 空间（ClearMap 细胞表用的编号）
        self.go_to_name = {}
        self.go_to_path = {}
        self.go_to_id = {}
        self.has_graph_order = False
        # float32 别名。ClearMap 的图谱标注是 float32 存的
        # (DeMBA_P5_annotation_trimmed*.tif)，而 CCF v3 里有 124 个 id 超过 float32
        # 的 24 位有效精度，写进文件时就被舍入成了另一个数（526157192 → 526157184），
        # 拿文件里的值去 ontology 查必然查不到 —— 本机图谱里这类体素占 1.0%，
        # hover 显示成 "Region 526157184"、按名字搜也搜不出来。
        # 两张表分别给"读文件的值→真实 id"和"真实 id→文件里的值"兜一次。
        self.f32_to_ids = {}
        self.id_to_f32 = {}
        self.parse_ontology(json_path)

    def parse_ontology(self, json_path):
        if not os.path.exists(json_path):
            print(f"❌ JSON not found: {json_path}")
            return

        print(f"📖 Parsing JSON Ontology...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        def extract_node(node, ancestors=()):
            if isinstance(node, list):
                for item in node: extract_node(item, ancestors)
                return
            if isinstance(node, dict):
                node_id = node.get('id') or node.get('structure_id')
                graph_order = node.get('graph_order')
                node_name = node.get('name') or node.get('safe_name') or node.get('acronym')

                path = ancestors
                if node_name is not None:
                    s_name = DataLoader.clean_part(node_name)
                    s_acr = DataLoader.clean_part(node.get('acronym'))
                    path = ancestors + ((s_name, s_acr),)
                    if graph_order is not None:
                        self.has_graph_order = True
                        self.go_to_name[int(graph_order)] = s_name
                        self.go_to_path[int(graph_order)] = path
                        if node_id is not None:
                            self.go_to_id[int(graph_order)] = int(node_id)
                    if node_id is not None:
                        self.id_to_name[int(node_id)] = s_name
                        self.id_to_path[int(node_id)] = path
                        # 同名脑区（不同半球等）保留先遇到的那个 id，和以前一致
                        if s_name not in self.name_to_id:
                            self.name_to_id[s_name] = int(node_id)
                        alias = int(np.float32(int(node_id)))
                        if alias != int(node_id):
                            self.id_to_f32[int(node_id)] = alias
                            self.f32_to_ids.setdefault(alias, []).append(int(node_id))

                children = node.get('children') or node.get('msg')
                if children: extract_node(children, path)

        if isinstance(data, dict):
            if 'msg' in data: extract_node(data['msg'])
            elif 'children' in data: extract_node(data['children'])
            else: extract_node(data)
        elif isinstance(data, list):
            extract_node(data)

    def resolve_label_value(self, value):
        """把标签体积里读到的数值换成 ontology 里真实存在的 id。

        返回 (id, ambiguous)。ambiguous=True 表示这个被 float32 舍入过的值对应
        多个真实 id（如 526157192 和 526157196 都存成 526157184，通常是同一脑区的
        相邻亚层），只能取第一个 —— 调用方该把结果标成"近似"，不要说得太确定。
        """
        try:
            v = int(value)
        except (TypeError, ValueError):
            return None, False
        if v in self.id_to_name:
            return v, False
        cands = self.f32_to_ids.get(v)
        if cands:
            return cands[0], len(cands) > 1
        return v, False

    def label_value_variants(self, region_id):
        """一个真实 id 在标签体积里可能以哪些数值出现（自己 + float32 别名）。"""
        alias = self.id_to_f32.get(int(region_id))
        return (int(region_id), alias) if alias is not None else (int(region_id),)

    def get_name(self, region_id, key='id'):
        if key == 'graph_order':
            try:
                return self.go_to_name.get(int(region_id), f"Region {region_id}")
            except (TypeError, ValueError):
                return f"Region {region_id}"
        rid, _ = self.resolve_label_value(region_id)
        if rid is None:
            return f"Region {region_id}"
        return self.id_to_name.get(rid, f"Region {region_id}")

    def get_path(self, region_id, key='id'):
        """返回 [(name, acronym), ...]，从最顶层祖先到该脑区自身。找不到就返回空列表。"""
        if key == 'graph_order':
            try:
                return list(self.go_to_path.get(int(region_id), ()))
            except (TypeError, ValueError):
                return []
        rid, _ = self.resolve_label_value(region_id)
        return list(self.id_to_path.get(rid, ())) if rid is not None else []

    def to_label_id(self, value, key='id'):
        """把细胞表里的编号换算成标签体积用的 atlas id。

        细胞点的颜色是拿这个 id 去问标签图层要的（get_color），不换算的话
        ClearMap 的 graph_order 会取到另一个脑区的颜色。查不到就退回 0。
        """
        try:
            value = int(value)
        except (TypeError, ValueError):
            return 0
        # 细胞表里 0 = background、-1 = no label（cellMap.py 的 raw_ids 约定），
        # 不是编号；graph_order 0 恰好是 root，不排掉的话所有背景细胞都会被涂成 root 色。
        if value <= 0:
            return 0
        if key == 'graph_order':
            return self.go_to_id.get(value, 0)
        return value

    def detect_cell_value_key(self, values, names):
        """判断细胞表第 9 列是 id 还是 graph_order：拿第 10 列的脑区名当答案对一遍。

        两条流程都会把脑区名一起写进 csv（ClearMap 第 10 列 name / ANTs 同一位置），
        所以哪套编号能对上更多行，第 9 列就是哪套编号 —— 比按数值范围猜可靠（数值
        大面积重合），也不依赖"哪个 pipeline 产的"这种外部信息。
        """
        if not self.has_graph_order:
            return 'id'  # 字典里根本没有 graph_order（如 DevCCF），只可能是 id
        score = {'id': 0, 'graph_order': 0}
        for v, n in zip(values, names):
            n = DataLoader.clean_part(n).lower()
            if not n or n in ('background', 'no label'):
                continue
            for key in score:
                if self.get_name(v, key=key).lower() == n:
                    score[key] += 1
        if score['graph_order'] == score['id'] == 0:
            return 'id'
        return 'graph_order' if score['graph_order'] > score['id'] else 'id'

    def get_lineage_lines(self, region_id, key='id'):
        """逐级 'acronym — full name'，用于终端打印时看清每一级到底是什么。"""
        return [f"{i}. {acr + ' — ' if acr else ''}{name}"
                for i, (name, acr) in enumerate(self.get_path(region_id, key=key))]

    def get_lineage_html(self, region_id, key='id'):
        """面板里用的层级块：从最顶层祖先一路列到该脑区自身，自身加粗。

        napari 0.8 的状态栏由后台 StatusChecker 线程按图层信息重算，手写 viewer.status
        会被立刻冲掉，所以脑区名和上级链只在这个面板里显示。
        """
        path = self.get_path(region_id, key=key)
        if not path:
            return "<i style='color:#a66;'>This ID is not in the ontology</i>"
        lines = []
        for i, (name, acr) in enumerate(path):
            text = html.escape(f"{acr} — {name}" if acr else name)
            if i == len(path) - 1:
                text = f"<b>{text}</b>"
            lines.append(f"<span style='color:#666;'>{i:>2}</span>&nbsp;{text}")
        return "<br>".join(lines)

# ================= 📂 2. 数据加载与处理 =================

class DataLoader:
    @staticmethod
    def clean_part(val):
        if pd.isna(val): return ""
        s = str(val).strip()
        s = re.sub(r"^b['\"]", "", s)
        s = re.sub(r"['\"]$", "", s)
        return s.strip().strip(',')

    @staticmethod
    def rotate_vol(arr, k):
        return np.rot90(arr, k=k, axes=(1, 2))

    @staticmethod
    def rotate_pts(pts, k, orig_shape):
        """将 (z,y,x) 坐标变换到旋转后的体积空间"""
        _, H, W = orig_shape
        pts = pts.copy()
        y, x = pts[:, 1].copy(), pts[:, 2].copy()
        if k == 1:   # 逆时针 90°: (y,x) → (W-1-x, y)
            pts[:, 1] = W - 1 - x
            pts[:, 2] = y
        elif k == 2: # 180°
            pts[:, 1] = H - 1 - y
            pts[:, 2] = W - 1 - x
        elif k == 3: # 顺时针 90°
            pts[:, 1] = x
            pts[:, 2] = H - 1 - y
        return pts

    @staticmethod
    def load_volume(path, dtype=None):
        """Read a 3D volume regardless of format: .tif/.tiff via tifffile,
        everything else (.nii/.nii.gz/.mhd/.mha/.nrrd) via SimpleITK -- ANTs
        writes standard NIfTI, and sitk reads it through the same ITK IO
        layer it already uses for ClearMap's .mhd files, so no special
        handling is needed per format beyond picking the reader."""
        if not path or not os.path.exists(path): return None
        if path.lower().endswith(('.tif', '.tiff')):
            arr = tifffile.imread(path)
        else:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
        if dtype is None:
            return arr
        # 标签图按无符号整型读时先把负值归零。ClearMap 的 volume/result.mhd 是
        # elastix 按 (ResultImagePixelType "short") 写出来的，而 CCF/DeMBA 的
        # 部分 id 超过 32767（本机 p5 图谱里 128 个），在 int16 里已经溢出成负数；
        # 直接 astype(uint32) 会变成 40 多亿的假 id，hover 显示 "Region 4294935936"。
        # 归零 = 当作背景，比假 id 诚实（受影响体素约 1%）。
        # Same class of problem as the signed-overflow branch below, opposite
        # direction: a float32 label volume has already lost the ids above
        # 2**24 by the time it reaches us, and casting to uint32 here makes the
        # rounded values look like clean integers again. Checked before the
        # cast, because afterwards there is nothing left to see.
        if np.issubdtype(np.dtype(dtype), np.unsignedinteger) and np.issubdtype(arr.dtype, np.floating):
            atlas_reference.check_label_dtype(arr.dtype, np.unique(arr), path)
        if np.issubdtype(np.dtype(dtype), np.unsignedinteger) and np.issubdtype(arr.dtype, np.signedinteger):
            n_neg = int((arr < 0).sum())
            if n_neg:
                print(f"⚠️ Label volume has {n_neg} negative voxels "
                      f"({os.path.basename(path)} is {arr.dtype}, so large ids overflowed "
                      f"when it was written); treating them as background.")
                arr = np.where(arr < 0, 0, arr)
        return arr.astype(dtype)

    @staticmethod
    def open_volume_lazy(path):
        """打开体积图像但尽量不整卷读进内存。

        原始配准 tiff 动辄几 GB~几十 GB，整卷 imread 会直接吃光内存；napari 每次
        只显示一层，所以给它一个能按需切片的对象就够了。优先级：
          memmap（未压缩 tiff，零拷贝，最快）→ zarr/dask（压缩 tiff，按 chunk 解码）
          → 整卷读入（前两者都不行时的兜底）
        非 tiff（nii/mhd）没有懒加载路径，只能整卷读 —— 它们本来就是降采样过的小文件。
        """
        if not path or not os.path.exists(path): return None
        if not path.lower().endswith(('.tif', '.tiff')):
            return sitk.GetArrayFromImage(sitk.ReadImage(path))

        size_gb = os.path.getsize(path) / 2**30
        try:
            arr = tifffile.memmap(path, mode='r')
            print(f"   ↳ opened as a memmap ({size_gb:.1f} GB, shape={arr.shape}), read on demand")
            return arr
        except Exception:
            pass  # 压缩或分块不连续的 tiff memmap 不了，走 zarr
        try:
            import dask.array as da
            store = tifffile.imread(path, aszarr=True)
            arr = da.from_zarr(store)
            print(f"   ↳ opened via zarr/dask ({size_gb:.1f} GB, shape={arr.shape}), decoded on demand")
            return arr
        except Exception as e:
            print(f"   ↳ no lazy path available ({type(e).__name__}: {e}), reading the whole "
                  f"volume into memory ({size_gb:.1f} GB)")
            return tifffile.imread(path)

    @staticmethod
    def volume_shape(path):
        """只读 header 拿到 (z, y, x) 形状，不解码任何像素。

        算高分辨率原图的 scale 只需要降采样网格的形状，没必要把降采样图整卷读进来
        （nii.gz 尤其慢）。ITK 的 GetSize() 是 (x,y,z)，反过来才是 numpy 轴序。
        """
        if not path or not os.path.exists(path): return None
        try:
            if path.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(path) as tf:
                    return tuple(tf.series[0].shape)
            reader = sitk.ImageFileReader()
            reader.SetFileName(path)
            reader.ReadImageInformation()
            return tuple(reversed(reader.GetSize()))
        except Exception as e:
            print(f"⚠️ Could not read the shape ({type(e).__name__}: {e}): {path}")
            return None

    @staticmethod
    def estimate_contrast(arr, percentiles=(0.5, 95.5), max_planes=64, max_pixels=4_000_000):
        """抽样估计显示用的对比度，返回 (contrast_limits, 采样到的 (min, max))。

        normalize_image_8bit 那套要把整卷读进来算百分位，对懒加载的大图等于白懒加载。
        这里先按形状把三个轴的步长都算好、一次切片取样，读进内存的量和图像大小无关
        —— 写成 arr[::z_step] 再降采样的话，dask 分支会先把几十个整层解码出来
        （2.8 GB 的原图 ≈ 18 MB/层 × 79 层 = 1.4 GB），等于没懒加载。

        第二个返回值给 contrast_limits_range 用：napari 收到显式 contrast_limits
        时会把滑条范围也锁成同一对值，不放开的话亮度就提不过 95.5 分位。
        """
        z_step = max(1, arr.shape[0] // max_planes)
        n_planes = -(-arr.shape[0] // z_step)
        yx_step = max(1, int(np.sqrt(n_planes * arr.shape[1] * arr.shape[2] / max_pixels)))
        sub = np.asarray(arr[::z_step, ::yx_step, ::yx_step])
        low, high = np.percentile(sub, list(percentiles))
        high = max(float(high), float(low) + 1)
        return (float(low), high), (float(sub.min()), max(float(sub.max()), high))

    @staticmethod
    def resolve_native_paths(target_dir):
        """Find everything the Native view needs, whichever pipeline produced it.

        Returns a dict:
          pipeline  'clearmap' | 'ants' | None
          img       降采样样本图，定义 Native 视图的世界坐标网格（= 细胞表第 4-6
                    列 "resample 空间" 坐标所在的网格，两条流程都是**未裁剪**的
                    那一份：ClearMap 的 resampled.tif（cellMap.py 里
                    sink_shape 固定用全图）/ ANTs 的 <name>_fine_*um.nii.gz）
          labels    图谱标签图 warp 回样本空间的结果 —— 两条流程的 Native 视图都靠
                    它画脑区轮廓、hover 查脑区（ClearMap 的 volume/result.mhd 由
                    cellMap.py 的 transform_annotation_volume() 生成，ANTs 的是
                    <name>_labels_in_sample.nii.gz）
          cropped   labels 是否处在"裁剪后"的网格上（见下）

        ClearMap 的 registration.crop_for_registration 会先把 resampled.tif 裁一
        块出来当 elastix 的 fixed image，于是 volume/result.mhd 也在裁剪后的小网格
        上，而细胞坐标在全图网格上 —— 两者要对齐得先按裁剪偏移补回去
        （clearmap_crop_info + pad_to_grid 负责）。ANTs 那边 labels_in_sample
        是写回未裁剪 fine 网格的（实测 251×517×295，和 fine_20um 同形状），不需要。
        """
        cm_img = os.path.join(target_dir, 'resampled.tif')
        cm_labels = os.path.join(target_dir, 'volume', 'result.mhd')
        if os.path.exists(cm_img) or os.path.exists(cm_labels):
            return {
                'pipeline': 'clearmap',
                'img': cm_img if os.path.exists(cm_img) else None,
                'labels': cm_labels if os.path.exists(cm_labels) else None,
                'cropped': True,
            }

        fine_matches = glob.glob(os.path.join(target_dir, '*_fine_*um.nii.gz'))
        labels_matches = glob.glob(os.path.join(target_dir, '*_labels_in_sample.nii.gz'))
        return {
            'pipeline': 'ants' if (fine_matches or labels_matches) else None,
            'img': fine_matches[0] if fine_matches else None,
            'labels': labels_matches[0] if labels_matches else None,
            'cropped': False,
        }

    # "Cropped resampled image: (296, 517, 252) -> (296, 517, 232), offset=[ 0.  0. 20.]"
    # —— cellMap.py 的 crop_resampled_for_registration() 打进 log.txt 的那行，
    # 是裁剪偏移唯一被落盘的地方（形状之差只能给出裁掉了多少，给不出从哪裁的）。
    _CM_CROP_RE = re.compile(
        r"Cropped resampled image:\s*\(([^)]*)\)\s*->\s*\(([^)]*)\)\s*,\s*offset=\[([^\]]*)\]")

    @staticmethod
    def clearmap_crop_info(target_dir):
        """从 log.txt 解析裁剪信息，返回 {'offset', 'full_shape', 'cropped_shape'}
        （都换成 napari 的 (z, y, x) 轴序），解析不到就返回 None。

        log.txt 里的三个数是 ClearMap 自己的 (X, Y, Z) 轴序，而磁盘上的 tif /
        mhd 读进来是 (Z, Y, X)（ClearMap/IO/TIF.py 的 array_to_tif 写盘时就把轴
        反过来了），所以倒序即可。
        """
        log_path = os.path.join(target_dir, 'log.txt')
        if not os.path.exists(log_path):
            return None
        info = None
        try:
            with open(log_path, 'r', errors='replace') as f:
                for line in f:
                    m = DataLoader._CM_CROP_RE.search(line)
                    if m:  # 同一目录可能跑过多次，取最后一次
                        info = m
        except Exception as e:
            print(f"⚠️ Could not parse the crop info in log.txt ({type(e).__name__}: {e})")
            return None
        if info is None:
            return None

        def _triple(text):
            vals = [int(round(float(v))) for v in text.replace(',', ' ').split()]
            return tuple(reversed(vals)) if len(vals) == 3 else None

        out = {'full_shape': _triple(info.group(1)),
               'cropped_shape': _triple(info.group(2)),
               'offset': _triple(info.group(3))}
        return out if all(v is not None for v in out.values()) else None

    @staticmethod
    def pad_to_grid(arr, offset, full_shape, what=""):
        """把裁剪后的小体积按 offset 摆回全图网格（周围补 0）。

        用补零而不是给图层加 translate：hover / 点击取值是直接拿
        viewer.cursor.position 去索引 current_atlas_labels 的（见 setup_callbacks），
        translate 只挪显示、不挪数组下标，标签图一旦带 translate，鼠标读到的就是
        错位的体素。补零后"世界坐标 = 全图网格下标"这条前提对所有图层都成立，
        旋转、highlight 图层 np.zeros_like 之类的下游逻辑一行都不用改。
        20 µm 网格下整卷也就几十 MB，代价可以忽略。
        """
        if arr is None: return None
        if tuple(arr.shape) == tuple(full_shape) and not any(offset):
            return arr
        if any(o < 0 for o in offset) or any(
                o + s > f for o, s, f in zip(offset, arr.shape, full_shape)):
            print(f"⚠️ {what}: shape {tuple(arr.shape)} + offset {tuple(offset)} runs past "
                  f"the full grid {tuple(full_shape)}; giving up on aligning it.")
            return None
        out = np.zeros(tuple(full_shape), dtype=arr.dtype)
        out[offset[0]:offset[0] + arr.shape[0],
            offset[1]:offset[1] + arr.shape[1],
            offset[2]:offset[2] + arr.shape[2]] = arr
        print(f"   ↳ {what}: padded from the cropped grid {tuple(arr.shape)} back onto the "
              f"full grid {tuple(full_shape)}, offset (z,y,x)={tuple(offset)}")
        return out

    @staticmethod
    def normalize_image_8bit(img_path):
        img = DataLoader.load_volume(img_path)
        if img is None: return None, None
        low, high = np.percentile(img, [0.5, 95.5])
        img_clipped = np.clip(img, low, high)
        return ((img_clipped - low) / (high - low) * 255).astype(np.uint8), img.shape

    @staticmethod
    def load_cells_from_registration(folder_path, ontology, coord_start):
        """动态扫描 cell_registration 下的子文件夹 (class_name)，读取其中的 cell_registration.csv

        列布局 (0-indexed): 0-2 原始像素坐标, 3-5 resample 空间坐标, 6-8 atlas 空间坐标,
        9 mapped_id, 10 region name, 11 slice_name, 12 tile_name, 13 score
        (后三者仅在 cellMap.py 较新版本生成的文件中存在)。coord_start 指定坐标起始列
        (3=resample空间, 6=atlas空间)。

        第 9 列的编号空间随流程而异（ClearMap 是 graph_order、ANTs 是 atlas id），
        所以脑区名直接用文件里第 10 列写好的那个，只在它为空时才回退到按编号查字典；
        另外额外算一列 label_id = 换算到标签图编号空间的 id，供细胞点取脑区颜色用。
        """
        all_dfs = []
        if not os.path.exists(folder_path): return pd.DataFrame()

        for entry in os.scandir(folder_path):
            if not entry.is_dir(): continue
            class_name = entry.name # 文件夹名即细胞分类名
            csv_path = os.path.join(entry.path, 'cell_registration.csv')

            if os.path.exists(csv_path):
                try:
                    df_raw = pd.read_csv(csv_path, header=None, names=range(20), engine='python')
                    if len(df_raw) > 0:
                        coords = df_raw.iloc[:, coord_start:coord_start + 3].values.astype(float)
                        valid_mask = ~np.isnan(coords).any(axis=1)
                        coords = coords[valid_mask]

                        ids = df_raw.iloc[:, 9].values[valid_mask]
                        ids = pd.to_numeric(pd.Series(ids), errors='coerce').fillna(0).astype(int).values

                        napari_pts = coords[:, [2, 1, 0]]

                        df_clean = pd.DataFrame(napari_pts, columns=['z', 'y', 'x'])
                        df_clean['class_name'] = class_name
                        df_clean['mapped_id'] = ids
                        # 第 10 列的脑区名（ClearMap 写 name，ANTs 同一位置写 name）。
                        # 编号空间判定和名字显示都用它，见下面的 detect_cell_value_key。
                        csv_names = (df_raw.iloc[:, 10].values[valid_mask]
                                     if df_raw.shape[1] > 10 else np.array([''] * len(ids)))
                        df_clean['csv_region'] = [DataLoader.clean_part(n) for n in csv_names]

                        # 原始像素坐标 (0-2列) 始终存在，用于标记细胞后回溯原始 tile。
                        raw_xyz = df_raw.iloc[:, 0:3].values.astype(float)[valid_mask]
                        df_clean['raw_x'] = raw_xyz[:, 0]
                        df_clean['raw_y'] = raw_xyz[:, 1]
                        df_clean['raw_z'] = raw_xyz[:, 2]

                        # slice_name/tile_name/score 只在新版 cellMap.py 输出的
                        # cell_registration.csv 中存在 (11/12/13列)；旧文件留空。
                        if df_raw.shape[1] > 12 and df_raw.iloc[:, 12].notna().any():
                            df_clean['slice_name'] = df_raw.iloc[:, 11].values[valid_mask]
                            df_clean['tile_name'] = df_raw.iloc[:, 12].values[valid_mask]
                            df_clean['score'] = (pd.to_numeric(df_raw.iloc[:, 13], errors='coerce').values[valid_mask]
                                                  if df_raw.shape[1] > 13 else np.nan)
                        else:
                            df_clean['slice_name'] = np.nan
                            df_clean['tile_name'] = np.nan
                            df_clean['score'] = np.nan

                        all_dfs.append(df_clean)
                        print(f"✅ Loaded {len(df_clean)} '{class_name}' cells.")
                except Exception as e:
                    print(f"❌ Failed to parse '{class_name}' coordinates: {e}")

        if not all_dfs:
            return pd.DataFrame()

        df = pd.concat(all_dfs, ignore_index=True)

        # 先定编号空间（拿名字列对答案），再据此补 region / label_id 两列。
        sample = df.sample(min(len(df), 5000), random_state=0)
        key = ontology.detect_cell_value_key(sample['mapped_id'].values, sample['csv_region'].values)
        print(f"🔢 Column 9 of the cell table reads as the '{key}' numbering space "
              f"({'ClearMap' if key == 'graph_order' else 'ANTs / raw atlas id'} style).")

        # 名字优先用文件里那一列；空的（老文件没写 name）才按编号查字典。
        fallback = df['csv_region'].eq('')
        df['region'] = df['csv_region']
        if fallback.any():
            df.loc[fallback, 'region'] = [ontology.get_name(uid, key=key)
                                          for uid in df.loc[fallback, 'mapped_id']]
        id_map = {uid: ontology.to_label_id(uid, key=key) for uid in df['mapped_id'].unique()}
        df['label_id'] = df['mapped_id'].map(id_map)
        return df


# ================= 🎮 3. 主控制器 =================

class MainController:
    def __init__(self, viewer):
        self.viewer = viewer

        self.viewer.axes.visible = False

        self.ontology = OntologyManager(CONFIG['ontology_json_path'])
        
        # 获取基础工作目录信息
        self.target_dir = CONFIG['sample_dir']
        self.sample_name = os.path.basename(os.path.normpath(self.target_dir))
        
        self.current_atlas_labels = None
        self.current_cells_df = pd.DataFrame() 
        self.highlight_atlas = None
        self.highlight_cells = None
        self.last_hover_val = -1
        
        self.mode = None
        self.cell_checkboxes = {}
        self.last_search_mode = "Exact"

        # 🚩 Flagging: click a cell -> pin it -> export tile/slice provenance
        # for suspicious cells so the user can pull up the raw TB-scale tile.
        self.flagged_cells = []       # list[dict], persists across Native/Atlas views
        self.last_clicked_cell = None # dict, most recent cell click (candidate to pin)
        self.flag_layer = None        # current napari points layer showing pins for this view
        self.last_highlighted_df = pd.DataFrame()  # last Search Regions result, for bulk export

        self.setup_ui()
        self.setup_callbacks()
        
        # 默认自动加载 Native 视图
        self.on_mode_change(self.combo_sample.currentText())

    def setup_highlight_layers(self, shape):
        """shape=None（没有任何标签体积可高亮）时只建细胞高亮层 —— 细胞的脑区筛选
        靠 csv 里的名字，不需要标签图，缺标签图时搜索仍应该能用。"""
        for name in [">> Highlight Atlas <<", ">> Highlight Cells <<", "✨ Selection"]:
            if name in self.viewer.layers: self.viewer.layers.remove(name)

        self.highlight_atlas = None
        if shape is not None:
            self.highlight_atlas = self.viewer.add_labels(np.zeros(shape, dtype=np.uint32),
                                                          name=">> Highlight Atlas <<", opacity=0.8)
        self.highlight_cells = self.viewer.add_points(np.empty((0, 3)), ndim=3, name=">> Highlight Cells <<",
                                                      face_color='white', border_color='yellow', size=self.spin_point_size.value(), opacity=1.0)

    @staticmethod
    def _fallback_region_color(label_id):
        """没有标签图层时给细胞点配色：按 id 定死的伪随机颜色。

        标签图缺失（如 ClearMap 的 volume/result.mhd 没生成成功）时以前是直接
        return、一个细胞都不画 —— 而细胞的脑区归属本来就写在 csv 里，不依赖标签图，
        不画等于把这个目录里唯一还完好的数据也藏起来。这里保证同一脑区在两种视图、
        多次启动之间颜色一致（种子只由 id 决定），只是和标签图的配色不是同一套。
        """
        rng = np.random.default_rng((abs(int(label_id)) * 2654435761) % (2 ** 32))
        return np.array([*rng.uniform(0.35, 1.0, 3), 1.0])

    def render_cells_from_df(self, df_cells, labels_layer):
        if df_cells.empty: return

        # --- 新增：如果未勾选，则过滤掉 mapped_id 为 0 的背景细胞 ---
        if hasattr(self, 'cb_show_bg') and not self.cb_show_bg.isChecked():
            df_cells = df_cells[df_cells['mapped_id'] != 0]
        # -------------------------------------------------------------
        if df_cells.empty: return

        # 颜色跟标签图层保持一致（拿换算到 id 空间的 label_id 去问它要颜色）；
        # 没有标签图层就退回自算的固定配色，细胞照画。
        id_color_map = {0: np.array([0.5, 0.5, 0.5, 1.0])}
        for uid in df_cells['label_id'].unique():
            if uid == 0: continue
            id_color_map[uid] = (labels_layer.get_color(uid) if labels_layer is not None
                                 else self._fallback_region_color(uid))

        # 动态读取并渲染独特的 class_name
        unique_classes = sorted(df_cells['class_name'].unique())
        
        for cls_name in unique_classes:
            sub_df = df_cells[df_cells['class_name'] == cls_name]
            if len(sub_df) > 0:
                coords = sub_df[['z', 'y', 'x']].values
                colors = np.array([id_color_map[uid] for uid in sub_df['label_id']])
                
                is_vis = self.cell_checkboxes[cls_name].isChecked() if cls_name in self.cell_checkboxes else True
                
                layer = self.viewer.add_points(
                    coords, name=f"Cell: {cls_name}", face_color=colors,
                    symbol='disc', size=self.spin_point_size.value(), border_width=0, blending='translucent', visible=is_vis
                )
                # Carried through for click-to-pin export (tile/slice
                # provenance -- see DataLoader.load_cells_from_registration).
                layer.features = pd.DataFrame({
                    'Region': sub_df['region'].values,
                    'Class': sub_df['class_name'].values,
                    'Tile': sub_df['tile_name'].values,
                    'Slice': sub_df['slice_name'].values,
                    'Score': sub_df['score'].values,
                    'RawX': sub_df['raw_x'].values,
                    'RawY': sub_df['raw_y'].values,
                    'RawZ': sub_df['raw_z'].values,
                })
                layer.events.highlight.connect(self.on_cell_layer_click)

    def perform_search(self, search_mode=None):
        if search_mode is not None:
            self.last_search_mode = search_mode
        search_mode = self.last_search_mode

        keyword = self.input_search.text().strip()
        self.last_highlighted_df = pd.DataFrame()

        if not keyword:
            if self.highlight_cells is not None: self.highlight_cells.data = np.empty((0, 3))
            if self.highlight_atlas is not None and self.current_atlas_labels is not None:
                self.highlight_atlas.data = np.zeros_like(self.current_atlas_labels)
            self.viewer.status = "Ready."
            return

        self.viewer.status = f"Searching: {keyword}..."

        # 1. 过滤 Atlas。name_to_id 是 atlas 原始 id 空间的表，标签体积里存的也是
        #    原始 id，两边编号空间一致（细胞那边的 graph_order 不参与这一步）。
        matched_ids = []
        if search_mode == 'Exact':
            for name, region_id in self.ontology.name_to_id.items():
                if name.lower() == keyword.lower(): matched_ids.append(region_id)
        else:
            for name, region_id in self.ontology.name_to_id.items():
                if keyword.lower() in name.lower(): matched_ids.append(region_id)
        # float32 存的图谱里大 id 会被舍入（见 OntologyManager 的 f32 别名），
        # 只按真实 id 找会漏掉整块脑区。
        matched_ids = [v for rid in matched_ids
                       for v in self.ontology.label_value_variants(rid)]

        if self.highlight_atlas is not None and self.current_atlas_labels is not None:
            if matched_ids:
                mask = np.isin(self.current_atlas_labels, matched_ids)
                h_data = np.zeros_like(self.current_atlas_labels)
                h_data[mask] = self.current_atlas_labels[mask]
                self.highlight_atlas.data = h_data
            else:
                self.highlight_atlas.data = np.zeros_like(self.current_atlas_labels)

        # 2. 过滤 Cells
        if not self.current_cells_df.empty:
            active_classes = [name for name, cb in self.cell_checkboxes.items() if cb.isChecked()]
            if search_mode == 'Exact':
                region_mask = self.current_cells_df['region'].str.lower() == keyword.lower()
            else:
                region_mask = self.current_cells_df['region'].str.contains(keyword, case=False, regex=False)
                
            class_mask = self.current_cells_df['class_name'].isin(active_classes)
            subset_df = self.current_cells_df[region_mask & class_mask]
            
            # --- 新增：搜索高亮时同步排除背景细胞 ---
            if hasattr(self, 'cb_show_bg') and not self.cb_show_bg.isChecked():
                subset_df = subset_df[subset_df['mapped_id'] != 0]
            # ----------------------------------------

            subset_points = subset_df[['z', 'y', 'x']].values

            self.highlight_cells.data = subset_points if len(subset_points) > 0 else np.empty((0, 3))
            self.last_highlighted_df = subset_df.copy()
            self.viewer.status = f"✅ [{search_mode}] Found {len(matched_ids)} regions | Cells: {len(subset_points)}"

    def update_class_filter_ui(self, df_cells):
        """核心：根据读取到的 df 动态生成 CheckBox"""
        for i in reversed(range(self.layout_classes.count())): 
            widget = self.layout_classes.itemAt(i).widget()
            if widget: 
                widget.setParent(None)
                widget.deleteLater()
        self.cell_checkboxes.clear()

        if df_cells.empty: return

        unique_classes = sorted(df_cells['class_name'].unique())

        for cls_name in unique_classes:
            cb = QCheckBox(cls_name)
            # 默认全部不勾选：细胞点铺满整卷会盖住图谱轮廓，先看配准、再按需打开类别。
            cb.setChecked(False)
            cb.stateChanged.connect(lambda state, n=cls_name: self.on_cell_check_toggle(n, state))
            self.layout_classes.addWidget(cb)
            self.cell_checkboxes[cls_name] = cb

    # --- 视图加载 ---

    def _resolve_native_image_path(self, resampled_path):
        """Native 视图显示哪张图：配置里指定了原始配准图就用它，否则用降采样图。"""
        hires = CONFIG.get('native_image_path')
        if not hires:
            return resampled_path
        if not os.path.exists(hires):
            print(f"⚠️ native_image_path does not exist, falling back to the downsampled "
                  f"image: {hires}")
            return resampled_path
        print(f"🔍 Using the full-resolution image: {hires}")
        return hires

    def _native_image_placement(self, img_shape, ref_shape):
        """把原始分辨率图像摆到降采样网格（= 世界坐标系）上，返回 (ok, scale)。

        细胞点用的是 cell_registration.csv 里的 resample 空间坐标，标签图也在同一
        网格上，所以世界坐标保持 = 降采样网格，只给这张高分辨率图加 scale，其余图层
        和 hover 取值（直接拿 viewer.cursor.position 去索引标签数组）的逻辑都不用动。
        scale=None 表示两者本来就同一网格，不用缩放；ok=False 表示对不齐、不该显示
        —— 高分图按 1:1 摆进去会和细胞点差好几倍，还不如不画。

        没有 translate：Registration_ants 用 ants.resample_image 生成降采样图，ITK
        重采样保留 origin，原图体素 0 的中心就是降采样体素 0 的中心（实测 raw idx i
        → fine idx i × spacing 比，精确成立），而 napari 把体素 i 画在世界坐标
        i*scale 处，正好对上。只有 cv2.resize/scipy.zoom 那种"外框对齐"的重采样才需
        要 (scale-1)/2 的半格补偿 —— 加错方向会整体偏半个降采样体素。

        scale 默认取两者形状之比。ants.resample_image 的输出尺寸是
        round(size*spacing/target)，所以这个比值最多差半个降采样体素（本样本
        2273→296，误差 0.2%，边缘累计约 0.5 个 20 µm 体素）；要精确就在 config 里
        写 native_image_scale = 原始体素尺寸 ÷ fine_target_um，按 (z,y,x) 序。
        """
        override = CONFIG.get('native_image_scale')
        if override:
            scale = tuple(float(s) for s in override)
        elif ref_shape is None:
            print("⚠️ No downsampled grid found (resampled.tif / *_fine_*um.nii.gz / label "
                  "volume), so the full-resolution image cannot be aligned to the cell "
                  "coordinate system.\n"
                  "   Use the downsampled image instead, or set native_image_scale "
                  "explicitly in the config.")
            return False, None
        elif tuple(img_shape) == tuple(ref_shape):
            return True, None
        else:
            scale = tuple(r / i for r, i in zip(ref_shape, img_shape))
        print(f"   ↳ aligned to the downsampled grid "
              f"{tuple(ref_shape) if ref_shape else '(scale from the config)'}: "
              f"scale={tuple(round(s, 4) for s in scale)}")
        return True, scale

    def _native_grid_offset(self, vol_shape, grid_shape, what):
        """裁剪后的体积在全图网格里的 (z,y,x) 偏移；返回 None 表示不该显示这一卷。

        形状本来就一致 → 偏移全 0。不一致时只可能是 ClearMap 的
        crop_for_registration：偏移从 log.txt 解析（config 里的
        labels_crop_offset 优先，日志被清掉或跑的是老版本 cellMap.py 时用它兜底）。
        两处都拿不到就宁可不画 —— 静默错位几十个体素比缺一层危险得多，判断配准好坏
        看的就是这个偏移量级的边界差。
        """
        if tuple(vol_shape) == tuple(grid_shape):
            return (0, 0, 0)
        override = CONFIG.get('labels_crop_offset')
        if override:
            offset = tuple(int(v) for v in override)
        else:
            info = DataLoader.clearmap_crop_info(self.target_dir)
            offset = info['offset'] if info else None
        if offset is None:
            print(f"⚠️ {what}: shape {tuple(vol_shape)} does not match the sample grid "
                  f"{tuple(grid_shape)}, and the crop offset cannot be determined --\n"
                  "   this is what ClearMap's registration.crop_for_registration looks like. "
                  "The offset is in that\n   directory's log.txt, on the "
                  "\"Cropped resampled image: ... offset=[...]\" line; if the log is gone, "
                  "write\n   labels_crop_offset: [z, y, x] in the config by hand "
                  "(napari axis order, i.e. the reverse\n   of the [x,y,z] in "
                  "log.txt).\n"
                  "   Skipping this layer for now, with no alignment.")
            return None
        return offset

    def load_sample_native_view(self):
        self.viewer.layers.clear()
        # 图层清空后这些引用指向的都是已经被移除的图层，必须一起作废，
        # 否则 perform_search 会往一个不在 viewer 里的图层写数据。
        self.current_cells_df = pd.DataFrame()
        self.current_atlas_labels = None
        self.highlight_atlas = None
        self.highlight_cells = None
        print(f"\n🚀 Loading Native View from: {self.target_dir}")

        paths = DataLoader.resolve_native_paths(self.target_dir)
        resampled_path, mhd_path = paths['img'], paths['labels']
        if paths['pipeline']:
            print(f"   ↳ recognised as {paths['pipeline']} output")
        cell_reg_dir = os.path.join(self.target_dir, 'cell_registration')

        img_path = self._resolve_native_image_path(resampled_path)
        img = DataLoader.open_volume_lazy(img_path)
        if img is not None:
            if img.ndim > 3:
                img = np.squeeze(img)  # ImageJ 超栈常带 (Z,1,Y,X) 这种单元素轴
            if img.ndim != 3:
                print(f"⚠️ Image is not 3D (shape={img.shape}), skipping: {img_path}")
                img = None
        mhd = DataLoader.load_volume(mhd_path, dtype=np.uint32)
        # 使用 cell_registration.csv 中已经算好的 resample 空间坐标 (第4-6列, 0-indexed 3:6)
        df_cells = DataLoader.load_cells_from_registration(cell_reg_dir, self.ontology, coord_start=3)

        # 世界坐标网格 = 降采样图本身：cell_registration.csv 的 "resample space" 列就是按
        # 它算的（ANTs 见 registration_ants/cell_points.py 拿 sample_fine 做
        # physical→index；ClearMap 见 cellMap.py 的 transformation()，sink_shape 固定
        # 用未裁剪的 resampled 形状），标签图则可能处在裁剪后的小网格上，得补回来。
        # 放在旋转之前算 —— 这时所有 shape 都还是文件里的原始轴序。
        grid_shape = DataLoader.volume_shape(resampled_path)
        if grid_shape is None and paths['cropped']:
            # 降采样图被删了（大文件常被清），但 log.txt 里同时记着裁剪前后的形状。
            info = DataLoader.clearmap_crop_info(self.target_dir)
            if info:
                grid_shape = info['full_shape']
                print(f"   ↳ downsampled image is gone; took the full grid {grid_shape} from log.txt")
        if grid_shape is None and mhd is not None and not paths['cropped']:
            grid_shape = mhd.shape
        if grid_shape is None and mhd is not None:
            print("⚠️ Cannot get the uncropped sample grid shape, so there is no way to "
                  "confirm the label volume and the cell points share one grid; showing the "
                  "label volume as-is.")

        if mhd is not None and grid_shape is not None:
            offset = self._native_grid_offset(mhd.shape, grid_shape, "sample-space labels")
            mhd = (DataLoader.pad_to_grid(mhd, offset, grid_shape, "sample-space labels")
                   if offset is not None else None)

        scale = None
        if img is not None:
            ok, scale = self._native_image_placement(img.shape, grid_shape)
            if not ok:
                img = None

        k = CONFIG.get('view_rotate_k', 0)
        if k:
            if img is not None:
                img = DataLoader.rotate_vol(img, k)
                if scale and k % 2:  # rot90 把 y/x 两轴换了位置，scale 也得跟着换
                    scale = (scale[0], scale[2], scale[1])
            if mhd is not None:
                mhd = DataLoader.rotate_vol(mhd, k)
            # 细胞点跟着世界坐标网格转，而不是跟着标签图转 —— 以前用的是标签图形状，
            # 标签图缺失时整段跳过，图转了点没转，无声错位。
            if not df_cells.empty and grid_shape is not None:
                df_cells[['z', 'y', 'x']] = DataLoader.rotate_pts(
                    df_cells[['z', 'y', 'x']].values, k, grid_shape)
            if grid_shape is not None and k % 2:
                grid_shape = (grid_shape[0], grid_shape[2], grid_shape[1])

        if img is not None:
            limits, full_range = DataLoader.estimate_contrast(img)
            img_layer = self.viewer.add_image(img, name="Raw Image", colormap="gray",
                                              blending='additive', scale=scale,
                                              contrast_limits=limits)
            # 显式传 contrast_limits 时 napari 会把 contrast_limits_range 锁成同一对
            # 值，滑条就提不过 95.5 分位了；QC 时经常要拉亮看暗处，所以放开范围。
            img_layer.contrast_limits_range = full_range
            if scale is not None:
                # 3D 纹理每轴上限一般是 2048（本机 llvmpipe 实测就是 2048），原图
                # y/x 都超了，napari 会整卷读进内存再抽稀到上限以内 —— 既看不到全
                # 分辨率，又把懒加载的好处清零。3D 看整体形状请用降采样图。
                print(f"   ↳ the full-resolution image is for 2D plane-by-plane viewing only: "
                      f"{img.shape} has an axis past the 3D texture limit (usually 2048), so "
                      "ndisplay=3 makes napari downsample it -- after reading the whole volume "
                      "into memory.")
        elif img_path:
            print(f"⚠️ Sample image not displayed: {img_path}")
        else:
            print(f"⚠️ No sample image found (resampled.tif or *_fine_*um.nii.gz) in: "
                  f"{self.target_dir}")

        labels_layer = None
        if mhd is not None:
            self.current_atlas_labels = mhd
            # 默认【填充】：一眼看清哪块是哪个脑区、边界围出来的是什么形状，
            # 这是日常查看时想要的。轮廓模式（只画边界、压在高分辨率原图上直接
            # 看边界偏了多少）仍然一键可切，见控制面板里的 cb_outline ——
            # 判断配准精度时切过去。
            labels_layer = self.viewer.add_labels(mhd, name="Atlas Regions", opacity=0.5)
            self.setup_highlight_layers(mhd.shape)
        else:
            print("⚠️ No sample-space label volume found (or it could not be aligned) "
                  "(ClearMap: volume/result.mhd / ANTs: *_labels_in_sample.nii.gz):\n"
                  "   hover-for-region and the atlas outlines are unavailable. Each cell's "
                  "region is written in\n   cell_registration.csv and does not "
                  "depend on this volume, so the cell points, the region search\n"
                  "   and filtering by region all still work.\n"
                  "   ClearMap writes this file from cellMap.py's transform_annotation_volume(); "
                  "re-run that to get it back.")
            self.setup_highlight_layers(None)

        self.current_cells_df = df_cells
        self.update_class_filter_ui(df_cells)
        self.render_cells_from_df(df_cells, labels_layer)
        self._apply_grid_layout("Atlas Regions")
        self._apply_region_contour()
        self.setup_flag_layer()

    def load_sample_atlas_view(self):
        self.viewer.layers.clear()
        self.current_cells_df = pd.DataFrame()
        self.current_atlas_labels = None
        self.highlight_atlas = None
        self.highlight_cells = None
        print(f"\n🚀 Loading Atlas Space View from: {self.target_dir}")

        atlas_layer = None
        atlas_path = CONFIG['std_atlas_path']
        cell_reg_dir = os.path.join(self.target_dir, 'cell_registration')
        # 使用 cell_registration.csv 中的 atlas 空间坐标 (第7-9列, 0-indexed 6:9)
        df_cells = DataLoader.load_cells_from_registration(cell_reg_dir, self.ontology, coord_start=6)

        k = CONFIG.get('view_rotate_k', 0)

        # ANTs pipeline 额外产出了"样本变形到图谱空间"的灰度图 (<name>_in_atlas.nii.gz)，
        # 跟 Atlas Anatomy 标签图是同一个网格，直接叠加显示，方便肉眼核对配准准不准。
        # ClearMap 原生流程没有这个文件，找不到就跳过，不影响原有功能。
        in_atlas_matches = glob.glob(os.path.join(self.target_dir, '*_in_atlas.nii.gz'))
        if in_atlas_matches:
            img_norm, _ = DataLoader.normalize_image_8bit(in_atlas_matches[0])
            if img_norm is not None:
                if k: img_norm = DataLoader.rotate_vol(img_norm, k)
                self.viewer.add_image(img_norm, name="Sample in Atlas", colormap="gray", blending='additive')

        if os.path.exists(atlas_path):
            data = DataLoader.load_volume(atlas_path, dtype=np.uint32)
            if k:
                orig_shape = data.shape
                data = DataLoader.rotate_vol(data, k)
                if not df_cells.empty:
                    df_cells[['z', 'y', 'x']] = DataLoader.rotate_pts(df_cells[['z', 'y', 'x']].values, k, orig_shape)
            self.current_atlas_labels = data
            atlas_layer = self.viewer.add_labels(self.current_atlas_labels, name="Atlas Anatomy", opacity=0.5)
            self.setup_highlight_layers(self.current_atlas_labels.shape)
        else:
            # 图谱标签图缺失也要能看细胞（细胞的 atlas 空间坐标和脑区名都在 csv 里）。
            print(f"⚠️ No reference atlas label file found, showing the cell points only: {atlas_path}")
            self.setup_highlight_layers(None)

        self.current_cells_df = df_cells
        self.update_class_filter_ui(df_cells)
        self.render_cells_from_df(df_cells, atlas_layer)
        self._apply_grid_layout("Atlas Anatomy")
        self._apply_region_contour()
        self.setup_flag_layer()

    # 跟着开关走的图层：脑区标签本身。高亮层（">> Highlight Atlas <<"）故意不跟 ——
    # 它是搜索的结果，永远填充才看得见，轮廓化等于把搜到的东西又藏起来。
    REGION_LAYER_NAMES = ("Atlas Regions", "Atlas Anatomy")

    def _apply_region_contour(self):
        """把"填充/轮廓"开关应用到当前存在的脑区标签层。

        每次切换视图都要重调一次：load_sample_space_view /
        load_sample_atlas_view 会清空并重建图层，新建的 Labels 层是 napari 的
        默认值（contour=0，填充），不会自己记得上次勾了什么。
        """
        width = 1 if getattr(self, 'cb_outline', None) and self.cb_outline.isChecked() else 0
        for name in self.REGION_LAYER_NAMES:
            if name in self.viewer.layers:
                self.viewer.layers[name].contour = width

    def _apply_grid_layout(self, atlas_layer_name):
        # 图谱图层移到最顶层：叠加模式下轮廓要压在原图上面才看得见。
        if atlas_layer_name in self.viewer.layers:
            idx = self.viewer.layers.index(atlas_layer_name)
            self.viewer.layers.move(idx, -1)
        self.viewer.grid.shape = (-1, 3)
        self.viewer.grid.enabled = self.cb_grid.isChecked()

    def on_cell_layer_click(self, event):
        layer = event.source
        if self.viewer.layers.selection.active != layer: return
        if len(layer.selected_data) > 0:
            idx = list(layer.selected_data)[0]
            full_name = layer.features['Region'].iloc[idx]
            self.input_search.setText(full_name)
            self.perform_search("Exact")

            pt = layer.data[idx]
            self.last_clicked_cell = {
                'sample': self.sample_name,
                'mode': self.mode,
                'class_name': layer.features['Class'].iloc[idx],
                'region': full_name,
                'z': float(pt[0]), 'y': float(pt[1]), 'x': float(pt[2]),
                'raw_x': layer.features['RawX'].iloc[idx],
                'raw_y': layer.features['RawY'].iloc[idx],
                'raw_z': layer.features['RawZ'].iloc[idx],
                'score': layer.features['Score'].iloc[idx],
                'slice_name': layer.features['Slice'].iloc[idx],
                'tile_name': layer.features['Tile'].iloc[idx],
            }
            self.lbl_last_click.setText(self._format_last_click(self.last_clicked_cell))

            layer.selected_data = set()

    # --- 🚩 Flagging: pin suspicious cells, export tile/slice provenance ---
    @staticmethod
    def _fmt(v, nd=1):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)): return '—'
            if isinstance(v, (int, float, np.floating, np.integer)): return f"{float(v):.{nd}f}"
            return str(v)
        except Exception:
            return '—'

    def _format_last_click(self, c):
        return (f"Region: {c['region']}  |  Class: {c['class_name']}\n"
                f"Tile: {self._fmt(c.get('tile_name'), 0)}   Slice: {self._fmt(c.get('slice_name'), 0)}\n"
                f"Raw XYZ: ({self._fmt(c.get('raw_x'))}, {self._fmt(c.get('raw_y'))}, {self._fmt(c.get('raw_z'))})"
                f"   Score: {self._fmt(c.get('score'), 3)}")

    def setup_flag_layer(self):
        """(Re)create the '🚩 Flagged Cells' points layer for the current
        view mode, restoring any previously-pinned points so flags survive
        switching between Native/Atlas (self.flagged_cells is the durable
        store; this layer is just its visual projection onto the current
        view's coordinate space)."""
        if "🚩 Flagged Cells" in self.viewer.layers:
            self.viewer.layers.remove("🚩 Flagged Cells")
        matches = [f for f in self.flagged_cells if f['mode'] == self.mode]
        pts = np.array([[f['z'], f['y'], f['x']] for f in matches]) if matches else np.empty((0, 3))
        self.flag_layer = self.viewer.add_points(
            pts, ndim=3, name="🚩 Flagged Cells", symbol='star',
            face_color='red', border_color='yellow', border_width=0.1,
            size=self.spin_point_size.value() * 2.2, opacity=0.95
        )

    def pin_last_clicked_cell(self):
        if not self.last_clicked_cell:
            QMessageBox.information(None, "Info", "No cell clicked yet. Click a cell point in "
                                    "the viewer first, then press Pin.")
            return
        entry = dict(self.last_clicked_cell)
        entry['pinned_at'] = pd.Timestamp.now().isoformat(timespec='seconds')
        self.flagged_cells.append(entry)
        if self.flag_layer is not None:
            pt = np.array([[entry['z'], entry['y'], entry['x']]])
            self.flag_layer.data = np.vstack([self.flag_layer.data, pt]) if len(self.flag_layer.data) else pt
        self.lbl_flag_count.setText(f"🚩 {len(self.flagged_cells)} cells flagged")

    def export_flagged_cells(self):
        if not self.flagged_cells:
            QMessageBox.information(None, "Info", "No cells flagged yet.")
            return
        path, _ = QFileDialog.getSaveFileName(None, "Export Flagged Cells", "flagged_cells.csv", "CSV Files (*.csv)")
        if not path: return
        cols = ['sample', 'mode', 'class_name', 'region', 'tile_name', 'slice_name',
                'raw_x', 'raw_y', 'raw_z', 'z', 'y', 'x', 'score', 'pinned_at']
        pd.DataFrame(self.flagged_cells)[cols].to_csv(path, index=False)
        QMessageBox.information(None, "Exported",
                                f"Exported {len(self.flagged_cells)} flagged cells -> {path}")

    def clear_flagged_cells(self):
        if not self.flagged_cells: return
        reply = QMessageBox.question(None, "Confirm",
                                     f"Clear all {len(self.flagged_cells)} flags? "
                                     f"This cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return
        self.flagged_cells = []
        if self.flag_layer is not None:
            self.flag_layer.data = np.empty((0, 3))
        self.lbl_flag_count.setText("🚩 0 cells flagged")

    def export_highlighted_cells(self):
        """Bulk export every cell currently matched by Search Regions --
        for when an entire anomalous region/cluster needs its source
        tiles/slices pulled, not just one outlier cell."""
        df = self.last_highlighted_df
        if df is None or df.empty:
            QMessageBox.information(None, "Info", "No highlighted cells to export. Search for "
                                    "a region with Search Regions first.")
            return
        path, _ = QFileDialog.getSaveFileName(None, "Export Highlighted Cells", "highlighted_cells.csv", "CSV Files (*.csv)")
        if not path: return
        export_df = df.copy()
        export_df['sample'] = self.sample_name
        cols = ['sample', 'class_name', 'region', 'mapped_id', 'tile_name', 'slice_name',
                'raw_x', 'raw_y', 'raw_z', 'z', 'y', 'x', 'score']
        cols = [c for c in cols if c in export_df.columns]
        export_df[cols].to_csv(path, index=False)
        QMessageBox.information(None, "Exported", f"Exported {len(export_df)} cells -> {path}")

    def setup_callbacks(self):
        @self.viewer.mouse_move_callbacks.append
        def on_mouse_move(viewer, event):
            if self.current_atlas_labels is None: return
            cursor = viewer.cursor.position
            if len(cursor) == 3:
                z, y, x = int(round(cursor[0])), int(round(cursor[1])), int(round(cursor[2]))
                shape = self.current_atlas_labels.shape
                if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                    val = self.current_atlas_labels[z, y, x]
                    if val != self.last_hover_val:
                        self.last_hover_val = val
                        if val > 0:
                            path = self.ontology.get_path(val)
                            if path:
                                name, acr = path[-1]
                                head = html.escape(f"{name} ({acr})" if acr else name)
                            else:
                                head = f"Region {val}"
                            # 文件里的值被 float32 舍入过时把真实 id 一并写出来，
                            # 别让人以为 ontology 里真有这么个 id；多解时标明是近似。
                            rid, ambiguous = self.ontology.resolve_label_value(val)
                            id_txt = f"ID {val}"
                            if rid is not None and rid != int(val):
                                id_txt += f" -> {rid}{' (ambiguous, first match)' if ambiguous else ''}"
                            # 面板里把整条链从根列到自身（自身加粗），不用鼠标旁边的浮窗，
                            # 免得挡住画面。
                            self.lbl_hover.setText(
                                f"📍 <b>{head}</b> · {id_txt}<br>"
                                + self.ontology.get_lineage_html(val)
                            )
                        else:
                            self.lbl_hover.setText("📍 Hover: Background")

        @self.viewer.mouse_drag_callbacks.append
        def on_click(viewer, event):
            active_layer = viewer.layers.selection.active
            if event.type != 'mouse_press' or self.current_atlas_labels is None: return
            if active_layer is not None and active_layer.mode == 'pan_zoom': return
            
            c = np.round(viewer.cursor.position).astype(int)
            shape = self.current_atlas_labels.shape
            if not all(0 <= c[i] < shape[i] for i in range(3)): return
            
            rid = self.current_atlas_labels[c[0], c[1], c[2]]
            if rid > 0: 
                name = self.ontology.get_name(rid)
                print(f"\n📍 Clicked Region: {name} (ID: {rid})")
                for line in self.ontology.get_lineage_lines(rid):
                    print(f"   {line}")

                if name and not name.startswith("Region"):
                    self.input_search.setText(name)
                    self.perform_search("Exact")

    def setup_ui(self):
        # napari 自己的 layer controls 缩到一半就不肯再缩了（QStackedWidget 的
        # minimumSizeHint 就是最高的那一页控件表单），左侧其他面板等于被它吃掉
        # 一大截高度还要不回来。包进滚动区并给个显式下限，拖到多矮都行，装不下
        # 的行改成滚动而不是被裁掉。
        ontology_tree_ui.free_layer_controls_height(self.viewer)
        dock = QWidget()
        # 起始宽度用 set_dock_width 给，而不是 setMaximumWidth —— 上限会让面板
        # 永远拖不宽（脑区名一长就只能看开头几个字），这正是要修的。
        ontology_tree_ui.shrinkable(dock)
        layout = QVBoxLayout(dock)
        
        layout.addWidget(QLabel("<b>1. Workspace Mode:</b>"))
        self.combo_sample = QComboBox()
        # 直接使用解析好的当前文件夹名称进行展示
        self.combo_sample.addItem(f"🐭 [Native] {self.sample_name}")
        self.combo_sample.addItem(f"📍 [Atlas ] {self.sample_name}")
        self.combo_sample.currentTextChanged.connect(self.on_mode_change)
        layout.addWidget(self.combo_sample)

        # 叠加 = 图谱轮廓压在（高分辨率的）原图上，一眼看出边界偏没偏；
        # 并排 = 每个图层各占一格、相机联动，适合对着看整体形状。
        self.cb_grid = QCheckBox("Grid side-by-side (unchecked = overlay)")
        self.cb_grid.setChecked(not CONFIG.get('native_image_path'))
        self.cb_grid.stateChanged.connect(
            lambda state: setattr(self.viewer.grid, 'enabled', state == Qt.Checked))
        layout.addWidget(self.cb_grid)

        # 填充 = 看清哪块是哪个脑区（默认）；轮廓 = 只画边界，压在高分辨率原图上
        # 直接看边界偏了多少，判断配准精度时用。napari 自己的 Labels 控件里也有
        # 一个 contour 数字框，这里只是把它提到常用位置、并且切视图后不丢。
        self.cb_outline = QCheckBox("Region outline only (unchecked = filled)")
        self.cb_outline.setChecked(False)
        self.cb_outline.stateChanged.connect(lambda _state: self._apply_region_contour())
        layout.addWidget(self.cb_outline)

        h_size = QHBoxLayout()
        h_size.addWidget(QLabel("<b>Cell Size:</b>"))
        self.spin_point_size = QDoubleSpinBox()
        self.spin_point_size.setRange(0.1, 50.0)
        self.spin_point_size.setValue(5.0)
        self.spin_point_size.setSingleStep(1.0)
        self.spin_point_size.valueChanged.connect(self.on_point_size_change)
        h_size.addWidget(self.spin_point_size)
        layout.addLayout(h_size)

        layout.addSpacing(5); line_flag = QFrame(); line_flag.setFrameShape(QFrame.HLine); layout.addWidget(line_flag); layout.addSpacing(5)

        layout.addWidget(QLabel("<b>🚩 Flag Suspicious Cell:</b> click a cell point, then Pin"))
        self.lbl_last_click = QLabel("No cell clicked yet")
        self.lbl_last_click.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_last_click.setWordWrap(True)
        layout.addWidget(self.lbl_last_click)

        h_flag_btns = QHBoxLayout()
        btn_pin = QPushButton("📌 Pin"); btn_export_flags = QPushButton("💾 Export"); btn_clear_flags = QPushButton("🗑 Clear")
        btn_pin.clicked.connect(self.pin_last_clicked_cell)
        btn_export_flags.clicked.connect(self.export_flagged_cells)
        btn_clear_flags.clicked.connect(self.clear_flagged_cells)
        h_flag_btns.addWidget(btn_pin); h_flag_btns.addWidget(btn_export_flags); h_flag_btns.addWidget(btn_clear_flags)
        layout.addLayout(h_flag_btns)

        self.lbl_flag_count = QLabel("🚩 0 cells flagged")
        self.lbl_flag_count.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_flag_count)

        layout.addSpacing(5); line2 = QFrame(); line2.setFrameShape(QFrame.HLine); layout.addWidget(line2); layout.addSpacing(5)

        layout.addWidget(QLabel("<b>2. Cell Class Filter:</b>"))

        # --- 新增：控制是否显示背景细胞的 CheckBox ---
        self.cb_show_bg = QCheckBox("Show Background Cells (ID=0)")
        self.cb_show_bg.setChecked(False) # 默认不显示背景细胞
        self.cb_show_bg.stateChanged.connect(self.on_bg_toggle)
        layout.addWidget(self.cb_show_bg)
        # ----------------------------------------
        
        # 动态 CheckBox 的容器
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.class_checkbox_container = QWidget()
        self.layout_classes = QVBoxLayout(self.class_checkbox_container)
        self.layout_classes.setContentsMargins(0,0,0,0)
        scroll.setWidget(self.class_checkbox_container)
        
        layout.addWidget(scroll)
        ontology_tree_ui.set_dock_width(
            self.viewer.window.add_dock_widget(dock, area='right', name="Control Panel"),
            PANEL_START_WIDTH_PX)

        self.setup_region_panel()

    def setup_region_panel(self):
        """搜索 + hover 单独一个 dock。

        与 Control Panel 拆开是因为 DevCCF 最深有十几级祖先，挤在控制面板里
        只能给个固定高度的小框，链一长就得滚动；单独成面板后谱系框拿
        stretch=1 吃掉剩下的全部空间，只有长到连这些空间都装不下才出现
        滚动条。面板本身高度固定，鼠标划过不同脉络时布局不会抖。
        """
        region_dock = QWidget()
        ontology_tree_ui.shrinkable(region_dock)
        rlayout = QVBoxLayout(region_dock)

        rlayout.addWidget(QLabel("<b>🔍 Search Regions:</b>"))
        self.input_search = QLineEdit(); self.input_search.setPlaceholderText("Region name...")
        self.input_search.returnPressed.connect(lambda: self.perform_search())
        rlayout.addWidget(self.input_search)

        h_search_btns = QHBoxLayout()
        btn_fuzzy = QPushButton("Fuzzy Search")
        btn_exact = QPushButton("Exact Search")
        btn_fuzzy.clicked.connect(lambda: self.perform_search("Fuzzy"))
        btn_exact.clicked.connect(lambda: self.perform_search("Exact"))
        h_search_btns.addWidget(btn_fuzzy)
        h_search_btns.addWidget(btn_exact)
        rlayout.addLayout(h_search_btns)

        btn_export_highlight = QPushButton("📤 Export Highlighted Cells (tile/slice)")
        btn_export_highlight.clicked.connect(self.export_highlighted_cells)
        rlayout.addWidget(btn_export_highlight)

        rlayout.addSpacing(5); rline = QFrame(); rline.setFrameShape(QFrame.HLine); rlayout.addWidget(rline); rlayout.addSpacing(5)

        rlayout.addWidget(QLabel("<b>📍 Hovered Region:</b>"))
        self.lbl_hover = QLabel("📍 Hover: None")
        self.lbl_hover.setStyleSheet("color: #999; font-size: 11px;")
        self.lbl_hover.setWordWrap(True)
        self.lbl_hover.setTextFormat(Qt.RichText)
        self.lbl_hover.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_hover.setTextInteractionFlags(Qt.TextSelectableByMouse)

        hover_scroll = QScrollArea()
        hover_scroll.setWidgetResizable(True)
        hover_scroll.setFrameShape(QFrame.NoFrame)
        hover_scroll.setMinimumHeight(120)
        hover_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hover_scroll.setWidget(self.lbl_hover)
        rlayout.addWidget(hover_scroll, 1)

        # add_vertical_stretch=False：napari 默认会在 dock 底部塞一个弹簧把控件
        # 顶到顶部，那样 hover 框的 stretch 就失效了。旧版本没这个参数就回退。
        try:
            added = self.viewer.window.add_dock_widget(
                region_dock, area='right', name="Region Explorer",
                add_vertical_stretch=False)
        except TypeError:
            added = self.viewer.window.add_dock_widget(
                region_dock, area='right', name="Region Explorer")
        ontology_tree_ui.set_dock_width(added, PANEL_START_WIDTH_PX)

    def on_point_size_change(self, val):
        for layer in self.viewer.layers:
            if layer.name.startswith("Cell:") or layer.name == ">> Highlight Cells <<":
                layer.size = val
            elif layer.name == "🚩 Flagged Cells":
                layer.size = val * 2.2

    def on_mode_change(self, text):
        if not text: return
        if "[Native]" in text:
            self.mode = "Native"
            self.load_sample_native_view()
        elif "[Atlas ]" in text:
            self.mode = "Atlas_Sample"
            self.load_sample_atlas_view()
    
    def on_cell_check_toggle(self, name, state):
        layer_name = f"Cell: {name}"
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                layer.visible = (state == Qt.Checked)
                break
        self.perform_search() 

    def on_bg_toggle(self, state):
        """当勾选/取消背景细胞显示时触发，重新渲染细胞图层"""
        # 1. 移除旧的细胞图层
        layers_to_remove = [layer.name for layer in self.viewer.layers if layer.name.startswith("Cell:")]
        for name in layers_to_remove:
            self.viewer.layers.remove(name)
            
        # 2. 找到当前的 Atlas 标签图层（用于赋予对应的脑区颜色）
        labels_layer = None
        for layer in self.viewer.layers:
            if layer.name in ["Atlas Regions", "Atlas Anatomy"]:
                labels_layer = layer
                break
                
        # 3. 根据当前设置重新渲染并恢复搜索高亮
        self.render_cells_from_df(self.current_cells_df, labels_layer)
        self.perform_search()

def main():
    parser = argparse.ArgumentParser(
        description="napari QC viewer for one registered sample "
                    "(sample/atlas views + cell points + region search)")
    local_config.add_config_arg(parser, "single_sample")
    args = parser.parse_args()

    # 就地更新，而不是重新绑定：MainController 的方法里直接引用模块级 CONFIG，
    # 换成新 dict 的话那些引用还指着旧的空 dict。
    CONFIG.update(local_config.load_config(
        "single_sample", cli_path=args.config, required=_REQUIRED_KEYS))

    viewer = napari.Viewer(title="Spatial Explorer (Single Workspace)")
    controller = MainController(viewer)  # noqa: F841 -- 必须留个引用，否则 Qt 面板会被 GC 掉
    napari.run()


if __name__ == "__main__":
    main()