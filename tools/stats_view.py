"""交互式浏览组间统计结果的 3D 热图。

加载配准用的图谱标注（默认只取右半球）和 ../Registration_ants 那边跑出来的
`region_stats.csv`，把每个脑区按选定的统计量着色，在 napari 里滑动查看。
可以随时切换 ontology level、细胞类别、指标、统计量。

三件需要先讲清楚的事：

* **颜色是逐脑区的，不是逐体素的。** 一个区是一整片同色，图说明的是差异落在
  哪个解剖结构上，不是结构内部的分布。这不是体素级统计图，不能当成那个读。
* **切换 level 靠的是"每个体素读它在该层的祖先"**：标着 CA1（level 8）的体素在
  level 5 的图上显示 HPF 的值。否则 level 5 的图只会点亮那些自身标签恰好在
  level 5 的零星体素，那不是"level 5 的结果"。
* **灰色 = 没有结果**，不是"没有差异"。可能是没被检验、被覆盖率/最小计数过滤掉、
  或者门控下祖先不显著所以根本没往下测。

配置：configs/stats_view.yaml（照 configs/stats_view.example.yaml 复制一份）。

    conda activate antsreg
    python tools/stats_view.py
    python tools/stats_view.py configs/stats_view.tsc.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from shared.local_config import load_config  # noqa: E402

# napari/Qt 都是懒加载：--selftest 和 --list 不需要显示器，也不该为了跑一遍
# 参数检查就付 napari 启动那几秒。
napari = None
QtWidgets = None


def _import_gui():
    global napari, QtWidgets
    if napari is not None:
        return
    import napari as _napari
    from qtpy import QtWidgets as _QtWidgets
    napari, QtWidgets = _napari, _QtWidgets


def _import_stats(ants_root):
    """把 ../Registration_ants 挂上 sys.path，复用它的 stats.region_maps ——
    level 折叠和取值的逻辑必须和静态图那边完全一致，不能各写一份。"""
    root = str(Path(ants_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from stats import region_maps
    return region_maps


VALUE_COLUMNS = {
    "log2fc": ("log2 fold change (B/A)", True),
    "hedges_g": ("Hedges' g (B - A)", True),
    "neglog10p": ("-log10 p (uncorrected)", False),
    "mean_a": ("group A mean", False),
    "mean_b": ("group B mean", False),
}


class StatsHeatmap:
    """把 region_stats.csv 渲染成体积，并在选项变化时重画。

    昂贵的一步（体素 -> 唯一标签下标）只做一次，之后换 level/类别/指标都只是
    一次 fancy index，所以在 6400 万体素的半脑上也是即时的。"""

    def __init__(self, annotation_path, ontology_path, stats_csv, region_maps,
                 hemisphere="right", downsample=1):
        self.rm = region_maps
        annot = region_maps.load_annotation(annotation_path)
        annot, self.ml_offset = region_maps.hemisphere_slice(annot, hemisphere)
        if downsample > 1:
            annot = annot[::downsample, ::downsample, ::downsample]
        self.ontology = region_maps.load_ontology(ontology_path)
        self.volume = region_maps.RegionVolume(annot, self.ontology)
        self.stats = pd.read_csv(stats_csv)
        with np.errstate(divide="ignore"):
            self.stats["neglog10p"] = -np.log10(self.stats["p_value"].clip(lower=1e-300))
        self.brain_mask = np.isfinite(
            self.volume.paint(np.ones(self.ontology.n), level=None))

    # ---- what the dropdowns can offer -------------------------------------
    def levels(self):
        return sorted(int(v) for v in self.stats["level"].unique())

    def classes(self, level=None):
        sub = self.stats if level is None else self.stats[self.stats["level"] == level]
        return sorted(sub["class_name"].unique())

    def metrics(self, level=None, class_name=None):
        sub = self.stats
        if level is not None:
            sub = sub[sub["level"] == level]
        if class_name is not None:
            sub = sub[sub["class_name"] == class_name]
        return sorted(sub["metric"].unique())

    def is_exploratory(self, level, class_name, metric):
        sub = self.stats[(self.stats["level"] == level)
                         & (self.stats["class_name"] == class_name)
                         & (self.stats["metric"] == metric)]
        if sub.empty or "exploratory" not in sub.columns:
            return False
        return bool(sub["exploratory"].iloc[0])

    # ---- rendering ---------------------------------------------------------
    def render(self, level, class_name, metric, value="log2fc",
               significant_only=True, alpha=0.05):
        vals, n = self.rm.values_by_order(
            self.stats, self.ontology, level, class_name, metric, value,
            significant_only=significant_only, alpha=alpha)
        vol = self.volume.paint(vals, level=level)
        if VALUE_COLUMNS[value][1]:
            lo, hi = self.rm.symmetric_limits(vol)
        else:
            finite = vol[np.isfinite(vol)]
            lo, hi = (0.0, 1.0) if finite.size == 0 else (float(finite.min()),
                                                          float(finite.max()))
        return vol, (lo, hi), n

    def region_names(self, level):
        """order -> 名称，给 hover 用。"""
        ids = self.volume.region_id_volume(level=level)
        return ids


def build_viewer(hm, cfg):
    _import_gui()
    state = {
        "level": cfg.get("level", hm.levels()[len(hm.levels()) // 2]),
        "value": cfg.get("value", "log2fc"),
        "significant_only": bool(cfg.get("significant_only", True)),
        "alpha": float(cfg.get("alpha", 0.05)),
    }
    state["class_name"] = cfg.get("class_name") or hm.classes(state["level"])[0]
    state["metric"] = cfg.get("metric") or hm.metrics(state["level"], state["class_name"])[0]

    viewer = napari.Viewer(title="Group stats heatmap")
    # 灰色脑轮廓垫在底下：留白的区是"没有结果"，那必须看得见，不能读成背景
    viewer.add_image(hm.brain_mask.astype(np.float32), name="brain", colormap="gray",
                     contrast_limits=(0, 3), opacity=1.0, blending="additive")
    vol, clim, n = hm.render(**state)
    layer = viewer.add_image(vol, name="statistic", colormap="RdBu_r",
                            contrast_limits=clim, opacity=0.9, blending="translucent")

    panel = QtWidgets.QWidget()
    form = QtWidgets.QFormLayout(panel)
    widgets = {}

    def _combo(items, current):
        c = QtWidgets.QComboBox()
        c.addItems([str(i) for i in items])
        if str(current) in [str(i) for i in items]:
            c.setCurrentText(str(current))
        return c

    widgets["level"] = _combo(hm.levels(), state["level"])
    widgets["class_name"] = _combo(hm.classes(state["level"]), state["class_name"])
    widgets["metric"] = _combo(hm.metrics(state["level"], state["class_name"]), state["metric"])
    widgets["value"] = _combo(sorted(VALUE_COLUMNS), state["value"])
    sig = QtWidgets.QCheckBox("only p_adj < alpha")
    sig.setChecked(state["significant_only"])
    widgets["significant_only"] = sig
    info = QtWidgets.QLabel("")
    info.setWordWrap(True)

    form.addRow("ontology level", widgets["level"])
    form.addRow("cell class", widgets["class_name"])
    form.addRow("metric", widgets["metric"])
    form.addRow("statistic", widgets["value"])
    form.addRow("", sig)
    form.addRow(info)

    updating = {"busy": False}

    def refresh(repopulate=False):
        if updating["busy"]:
            return
        updating["busy"] = True
        try:
            state["level"] = int(widgets["level"].currentText())
            if repopulate:
                # level 变了，可用的类别/指标也可能变（门控下深层只剩少数分支）
                for key, options in (("class_name", hm.classes(state["level"])),
                                     ("metric", None)):
                    if key == "metric":
                        options = hm.metrics(state["level"], state["class_name"])
                    w = widgets[key]
                    keep = w.currentText()
                    w.blockSignals(True)
                    w.clear()
                    w.addItems([str(o) for o in options])
                    if keep in [str(o) for o in options]:
                        w.setCurrentText(keep)
                    w.blockSignals(False)
            state["class_name"] = widgets["class_name"].currentText()
            state["metric"] = widgets["metric"].currentText()
            state["value"] = widgets["value"].currentText()
            state["significant_only"] = sig.isChecked()

            vol, clim, n = hm.render(**state)
            layer.data = vol
            layer.contrast_limits = clim
            layer.colormap = "RdBu_r" if VALUE_COLUMNS[state["value"]][1] else "viridis"
            label = VALUE_COLUMNS[state["value"]][0]
            warn = ("\n[EXPLORATORY level: uncorrected. 着色的区没有通过任何校正，"
                    "只能当线索读。]"
                    if hm.is_exploratory(state["level"], state["class_name"],
                                         state["metric"]) else "")
            info.setText(f"{label}\n{n} regions coloured   range "
                         f"[{clim[0]:.3g}, {clim[1]:.3g}]\n"
                         f"灰色 = 没有结果（未检验/被过滤/门控未到达），不是没有差异。{warn}")
        finally:
            updating["busy"] = False

    widgets["level"].currentTextChanged.connect(lambda _: refresh(repopulate=True))
    widgets["class_name"].currentTextChanged.connect(lambda _: refresh(repopulate=True))
    for key in ("metric", "value"):
        widgets[key].currentTextChanged.connect(lambda _: refresh())
    sig.stateChanged.connect(lambda _: refresh())
    refresh()

    viewer.window.add_dock_widget(panel, area="right", name="stats")
    return viewer


def selftest():
    """无显示器也能跑：只验证渲染管线，不启动 napari。"""
    import json
    import tempfile
    ok = 0
    with tempfile.TemporaryDirectory() as tmp:
        onto = {"msg": [{"id": 1, "name": "root", "acronym": "root", "children": [
            {"id": 10, "name": "Area A", "acronym": "A", "children": [
                {"id": 100, "name": "A1", "acronym": "A1", "children": []}]},
            {"id": 20, "name": "Area B", "acronym": "B", "children": []}]}]}
        op = Path(tmp) / "onto.json"
        op.write_text(json.dumps(onto))

        import tifffile
        annot = np.zeros((4, 4, 8), dtype=np.uint32)
        annot[..., :4] = 100      # A1, level 2
        annot[..., 4:] = 20       # B,  level 1
        ap = Path(tmp) / "annot.tif"
        tifffile.imwrite(ap, annot)

        stats = pd.DataFrame([
            {"level": 1, "id": 10, "class_name": "X", "metric": "Density",
             "log2fc": 2.0, "p_value": 0.001, "p_adj": 0.01, "exploratory": False},
            {"level": 1, "id": 20, "class_name": "X", "metric": "Density",
             "log2fc": -1.0, "p_value": 0.4, "p_adj": 0.9, "exploratory": False},
        ])
        sp = Path(tmp) / "stats.csv"
        stats.to_csv(sp, index=False)

        rm = _import_stats(_HERE.parent.parent / "Registration_ants")
        hm = StatsHeatmap(ap, op, sp, rm, hemisphere="both")

        vol, clim, n = hm.render(1, "X", "Density", "log2fc", significant_only=False)
        assert n == 2, n
        # A1 的体素在 level 1 上必须读出它的祖先 A 的值 —— 这是整个 level 切换的关键
        assert np.allclose(vol[..., :4], 2.0), vol[..., :4]
        assert np.allclose(vol[..., 4:], -1.0), vol[..., 4:]
        assert clim[0] == -clim[1], clim
        ok += 1
        print("  ok: level 折叠 —— 子区体素读出祖先的值")

        vol2, _, n2 = hm.render(1, "X", "Density", "log2fc", significant_only=True)
        assert n2 == 1, n2
        assert np.allclose(vol2[..., :4], 2.0)
        assert np.isnan(vol2[..., 4:]).all(), "不显著的区必须留白，不能画成 0"
        ok += 1
        print("  ok: significant_only 把不显著的区留白（NaN 而不是 0）")

        assert hm.levels() == [1] and hm.classes() == ["X"]
        ok += 1
        print("  ok: 下拉选项来自表格本身")
    print(f"selftest: {ok} 项通过")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    cfg = load_config("stats_view", args.config,
                      required=("ants_root", "annotation_path", "ontology_path",
                                "region_stats_csv"))
    rm = _import_stats(cfg["ants_root"])
    print(f"Loading {cfg['annotation_path']} ...")
    hm = StatsHeatmap(cfg["annotation_path"], cfg["ontology_path"],
                      cfg["region_stats_csv"], rm,
                      hemisphere=cfg.get("hemisphere", "right"),
                      downsample=int(cfg.get("downsample", 1)))
    print(f"levels={hm.levels()}  classes={hm.classes()}")
    build_viewer(hm, cfg)
    _import_gui()
    napari.run()


if __name__ == "__main__":
    main()
