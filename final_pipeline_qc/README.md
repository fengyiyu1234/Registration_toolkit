# qc/ — 质控

三套互相独立的东西：

- **配准后的细胞质控**（`view_region_cells.py` + `region_cells.py`）：按脑区选细胞，
  回到原始数据上看它们是不是真在那个区里，逐个判定。不是盲的。见文末
  「配准后质控：按脑区回看原图」。
- **全链路质控**（`view_detection_qc.py` + `detection_boxes.py`）：同样按脑区选点，
  但把 brain_detector 的**每一级**检测框也铺开 —— 原始 2D、z-link 去重后、共定位
  后 —— 所以能回答「类别比例是在哪一步变的」，而不只是「这个细胞归区对不对」。
  不是盲的。见文末「全链路质控：每一级检测框 + 分级漏斗」。
- **检测漏检率的穷举标注**（下面这一大段）：盲的，切 crop、人工从零标。

三套的输出目录不要混用。前两套共用 `verdicts.csv` 的格式，但判定词表不同
（`rc.VerdictStore(..., vocab=...)`），所以也各给各的 `out_dir`。

---

# 检测漏检率的穷举标注

回答一个问题：**统计结果里的 Sox9 共定位细胞，有多少是检测漏掉的？**

三个脚本，只有中间那个有界面。

```
qc/cut_crops.py       命令行。在有全分辨率 tile 的机器上跑一次，切出盲的 crop
qc/annotate_crop.py   唯一的 GUI。napari，一次开一个 crop，人在这里干活
qc/score_crops.py     命令行。全部标完之后跑一次，出混淆矩阵
qc/crop_geometry.py   被上面两个 import，不单独跑
```

配置是 `configs/qc_crops.yaml`，`cut_crops.py` 和 `score_crops.py` 共用。
`annotate_crop.py` 不读配置，只接受一个 crop 目录 —— 见下面「盲法」一节。

## 为什么不是稀疏抽样

从检测结果里抽一批细胞逐个核对，测到的只有 precision。一个被漏掉的细胞进入
样本的概率恒等于零，所以真正存疑的那个量在这个设计里原理上看不见。

穷举标注换的是**抽样单位**：抽体积不抽细胞，在 crop 里把每个细胞从零标一遍，
分母于是从「pipeline 说有的」变成「实际有的」，recall 才有定义。

代价是误差棒的单位也跟着变成 crop。同一个 crop 里的细胞共享深度、tile、染色和
局部密度，不是独立观测，所以置信区间在 `score_crops.py` 里是对 **crop** 做
cluster bootstrap，不是对细胞算二项分布。这也是为什么要切十几个中等大小的
crop，而不是两个大的。

## 为什么 Sox9 是重点

细胞只在 GFP 和 RFP 两个 soma 通道上被检出。Sox9 不是独立检出的细胞，它是
brain_detector 用严格的 3D bbox 包含关系贴到 soma 上的属性。所以 **Sox9 假阴性
不丢细胞，只是把细胞从 `X_Sox9` 挪到 `X`**。后果分三条：

- `all_cells` / `GFP_any` / `RFP_any` 完全免疫，细胞还在，只是改了名。
- Sox9 类的 `Percentage` 在漏检率空间均匀时基本抵消，分子分母同比缩放。
- Sox9 类的 `Density` 和 `RegionProportion` 直接按局部命中率衰减。

而且如果两组漏得一样多，效应只是被拉向 null，那是统计假阴性，不是假阳性。
**唯一能凭空造出结论的是两组漏检率不同**，所以 `score_crops.py` 把组间
sensitivity 之差的置信区间单独打出来，那是第一个该看的数。

## 盲法是结构保证的，不是靠自觉

`out_dir` 里只有图像。crop 目录用随机 id 命名，`crop.json` 里只有尺寸和体素
大小，没有样本、没有分组、没有脑区、没有任何预测。

样本、分组、脑区、深度带和该 crop 内的全部预测，写在 `manifest_path` 那份封存
清单里。`cut_crops.py` 拒绝把清单写进 `out_dir`。`annotate_crop.py` 这个文件里
根本没有读清单的代码路径。

**把 `out_dir` 整个交给标注者，清单留在自己这里。**

## 反向映射：不需要 inverse warp

选 crop 要从图谱脑区反推回原始 tile。这一步比看上去简单，因为反向映射已经在
磁盘上了：`<run>/*_labels_in_sample.nii.gz` 就是图谱标签 warp 进样本空间的结果，
20 µm 各向同性，origin 为 0，落在**未裁剪**的 fine 网格上。整条流水线共用一个
物理坐标系，origin 0、spacing 就是微米，所以两个坐标系之间只差一次缩放：

```
label_voxel = global_px * cell_voxel_um / 20.0     # 0.65/20 = 0.0325, 8/20 = 0.4
global_px   = label_voxel * 20.0 / cell_voxel_um   # 反过来乘 30.77 / 30.77 / 2.5
```

不需要把图谱 warp 到 0.65 µm 网格，那是约 960 亿个体素，uint32 存下来 384 GB。
坐标算术是唯一可行的做法。

从全局像素到 tile 的那一步由 `crop_geometry.TileGrid` 做，它照抄
brain_detector `visualize.py` 的 `_get_tile_offset` 约定，因为
`cell_registration.csv` 里的全局坐标就是在那个约定下写出来的：

```
tile_x0 = ABS_H - min(ABS_H)
tile_y0 = ABS_V - min(ABS_V)
tile_z0 = max(ABS_D) - ABS_D
local_z_0idx = global_z - 1 + tile_z0
```

抄而不是 import：这个仓库依赖 `registration_ants`，不依赖 `brain_detector`。

## signal check：一定要看

每切一个 crop，`cut_crops.py` 都会打印一个比值：预测的细胞中心处的平均强度，
除以同一个 crop 里随机位置的平均强度。**切对了这个数远大于 1。**

接近 1 意味着读出来的框和预测描述的不是同一块组织 —— z 起始约定错了、某个通道
的 tile 有偏移、用错了 merging xml。这类错误没有任何其它症状：crop 里仍然是真
组织，看起来仍然像脑子，后面所有 recall 数字全是垃圾但不会有任何东西报错。

低于 1.5 会打 `<-- CHECK`。先修好再让任何人开始标注。

## 已知的坑

- **s18 跑的是重定位过的 run。** 那批细胞的原始像素坐标没变，但归区用的是搬过
  之后的位置，所以碎片上的组织从脑区反推回原图不是纯缩放。让 s18 的 crop 避开
  碎片区域。
- **`crop_for_registration` 之外没有标签。** 裁剪范围外的 `labels_in_sample`
  被清零，那里的皮层选不出 crop。这是正确行为，不是 bug。
- **通道之间的残余对齐偏移**既是切图要处理的事，也正是 Sox9 假阴性的机制来源
  之一。`channels` 下的 `offset_px` 可以手动补偿，但填之前先确认它是系统性的。

## 跑法

```bash
conda activate antsreg
cp configs/qc_crops.example.yaml configs/qc_crops.yaml   # 一次，然后改路径
python qc/cut_crops.py --dry-run          # 只选点、数细胞，不碰 tile
python qc/cut_crops.py                    # 真的切图
python qc/annotate_crop.py <out_dir>/<crop_id>           # 每个 crop 开一次
python qc/score_crops.py                  # 全部标完之后
python tests/test_qc_crops_smoke.py       # 自测，headless
```

`--dry-run` 只需要 `run_dir`，在没挂载全分辨率数据的机器上也能跑，用来先确认
选点位置和每个 crop 的细胞数量级。

## 先验一件事再投入

在切满之前，先切 2 个 crop，让同一个人隔几天盲标两遍，或者两个人各标一遍，看
Sox9 判读的一致性。如果人眼自己前后都不一致，就不存在 ground truth，recall 测
出来只是标注噪声。`annotation.meta.json` 里记了 annotator 和时间，就是为了算
这个。

`uncertain` 那一层同理：它的比例本身就是「Sox9 能不能用人眼判」的答案。比例高
的话诚实的结论是判不了，退回去比较 Sox9 阴性 soma 里 730 nm 通道的强度分布是
不是双峰，那条路不需要任何人工判读。

---

# 配准后质控：按脑区回看原图

```bash
cp configs/region_qc.example.yaml configs/region_qc.yaml   # 改 run_dir / out_dir
python qc/view_region_cells.py --summary --frame-check      # 先看数字，约 10 秒
python qc/view_region_cells.py --snapshot                   # 每个 site 一张 PNG
python qc/view_region_cells.py                              # napari 逐个判定
python tests/test_region_qc_smoke.py                        # 自测，headless
```

- **选细胞**：`regions` 写名字 / 缩写 / id，自动含后代；`classes` 按文件夹名通配。
  按细胞表第 9 列选，和统计用的是同一个归区。`prefer: boundary` 只挑离边界近的。
- **读图**：`source: volume` 用配准网格的整脑 tif（哪台机器都能跑，看组织边缘和
  分层）；`source: tiles` 用 0.65 µm tile（挂着 Y: 的机器，看单个细胞），tile
  拼接复用 `crop_geometry.TileGrid`。
- **每个 site 给两个归区答案**：细胞表的（细胞推进图谱空间查的）和
  `labels_in_sample` 的（图谱拉回样本空间查的），外加到所选区域边界的有符号
  距离（µm，正 = 区内）。不一致的几乎都在边界 20 µm 以内。
- **判定**写进 `<out_dir>/verdicts.csv`，按 run_dir + 细胞 id（`类别:行号`）记，
  可以关掉再接着判。键盘 `1-4` 判定、`n/p` 翻页、`u` 跳到下一个未判的。

## frame check：细胞坐标和原图对不对得上

`--frame-check`（仅 volume 模式）在整脑图上比较 marker+ 和 marker- 细胞处的平均
强度，扫 z 平移，逐 tile 给出峰值位置。两组细胞都在组织里，组织/背景的差异抵消，
剩下的只有 marker 本身，所以比「细胞 vs 框内随机点」灵敏得多（后者在 2.6 µm 图上
错开 50 µm 几乎不变）。555 的配准图用 `RFP`，730 用 `Sox9`。

2026-09-17 在六个 TSC run 上实测（RFP，555 图）：

| run | dz=0 比值 | 最佳 dz | 逐 tile |
|---|---|---|---|
| s8 DeMBA_0904 | 2.5 | −20 µm | 19/20 个 tile 都是 −20 |
| s11 DeMBA_0902 | 2.1 | −20 µm | 26/33 个是 −20，其余 −12~−32 |
| s18 DeMBA_0902_repos | 1.9 | −20 µm | — |
| s12t DeMBA_0915 | 1.4 | −24 µm | −16~−36 |
| **s12q DeMBA_0830_mask2** | **1.0** | −80 µm | **−44~−148，各 tile 不一致** |
| **s12t DeMBA_0828** | **1.0** | −120 µm | **−32~−140，各 tile 不一致** |
| **s10 DeMBA_0904** | **1.0** | −68 µm | 没有 tile_name，只有整体 |

正常 run 统一在 −20 µm，这和「配准图第 k 层 = 全局第 4k+1~4k+4 层的平均」一致
（从偏移反推的，没核对降采样代码）。也就是流水线把所有细胞放深了约 20 µm，
各样本相同。各 tile 散开几十微米的，是 cell_centroids 本身的 tile z 起点有问题。

---

# 全链路质控：每一级检测框 + 分级漏斗

```bash
cp configs/detection_qc.example.yaml configs/detection_qc.yaml   # 改 run_dir / detection_dir / out_dir
python qc/view_detection_qc.py --funnel      # 先跑这个：数字，不读图
python qc/view_detection_qc.py --snapshot    # 每个 site 一张 PNG
python qc/view_detection_qc.py               # napari 逐个看、逐个判定
python tests/test_detection_qc_smoke.py      # 自测，headless
```

## 它比 `view_region_cells.py` 多了什么

那个工具画的是 `cell_registration.csv`，一个细胞一个点 —— 整条链路的**最后一步**。
点画对了只说明归区没错，说明不了这个细胞是怎么来的。这个工具把 brain_detector 的
中间产物也铺开：

| 图层 | 来自 | 一行是什么 |
|---|---|---|
| `[s2] <ch>` | `2_global_2d_raw/{ch}_2d_global.csv` | 一个 2D 框在一层上 |
| `[s3] <ch>` | `3_channel_3d/{ch}_3d_tracked.csv` | z-link 之后的一个细胞 |
| `[coloc] GFP+Sox9` | `4_colocalization/coloc_result.csv` | 共定位判定后的最终类别 |
| `region cells` | `cell_registration/*/cell_registration.csv` | 统计真正数的那一批 |

没有 `2_global_2d_raw/` 时 `[s2]` 自动回退到逐 tile 的 `1_tile_2d_filtered/`
（stage 3 自己读的就是它），那时才需要 `channels` 里的拼接 xml 来还原偏移。

coloc 那几层和 `visualize.py` 一样是**超集**匹配：GFP+RFP+Sox9 的细胞同时出现在
GFP+RFP 层和 GFP+Sox9 层里。几层叠起来是一张 Venn 图，不是互斥分区 —— 各层框数
加起来超过细胞数是对的。

## 先跑 `--funnel`

`--funnel` 一张图都不读，两趟扫盘，回答的是**类别比例是在哪一级变的**：

```
检测分级漏斗 —— Isocortex（归区口径：labels_in_sample 回查框中心）
  s2  原始 2D（全局）：GFP 1,223   RFP 525   Sox9 628
  s3  z-link 去重后：GFP 415   RFP 177   Sox9 214
  s4  共定位后：592
        glia_GFP_Sox9      214
        ...
        Sox9+ 占比       36.15%
  z-link 压缩比（2D 框·层 ÷ 细胞，≈ 每个细胞跨几层）：GFP 2.95  RFP 2.97  Sox9 2.93
  细胞表 cell_registration/（第 9 列归区）：592   Sox9+ 占比 36.15%
```

看两个数：每一级各有多少，以及 **Sox9+ 占比**。占比是这套数据里唯一跨样本可比的
量（标记效率是动物级的，这个设计没有内参），所以它在哪一级变的，就是该去查哪一级。
逐个看图是用来解释漏斗里那个跳变的，不是用来发现它的。

`z-link 压缩比` 是一个细胞平均跨几层。同一个样本里各通道差很多，或者两组之间差
很多，说明 `z_linker` 的 `iou_thresh` / `max_cell_z_span` 对某个通道不合适 ——
过并会吃掉细胞数，欠并会凭空多出细胞。

## s4 → cell_registration 对账：唯一一处应该精确相等的接缝

`run_inference.py` 写质心用的是 `cx = (x1 + x2) / 2`、`cy = (y1 + y2) / 2`、`z`
原样；`cell_points.py` 把这三个数原样抄进 `cell_registration.csv` 的 0-2 列
（reposition 也不动它们）。所以这两份文件**应该逐行精确相等**：

```
s4 → cell_registration 对账（(x1+x2)/2, (y1+y2)/2, z, class 精确相等）：
  coloc_result.csv 1,200 行，细胞表 1,200 行，匹配上 1,200（100.00%）
  ✅ 完全一致
```

对不上不是精度问题（坐标全程 float64，就是为了这个检查），是细胞表和这份
`coloc_result.csv` 不是同一次跑出来的、或者中间丢了行。**先修好再看任何按类别
或按区的统计**，因为这两份东西一旦错位，分组比较里的差异可以完全是它造出来的。

## 判定词表和归区质控不一样

| 键 | 含义 |
|---|---|
| `ok` | 都对 |
| `over_merge` | 该分没分（z-link 过并） |
| `under_merge` | 该并没并（一个细胞数成多个） |
| `sox9_wrong` | Sox9 判错 |
| `not_cell` | 不是细胞 / 看不清 |

写进 `<out_dir>/verdicts.csv`，格式和 `view_region_cells.py` 的一样（按
run_dir + 细胞 id），所以两边的判定可以并在一张表里读，但**词表不同**，不要共用
同一个 `out_dir`。

## 几件要知道的事

- **框只有在 `source: tiles` 下才对得上单个细胞。** 框本来就是在 0.65 µm 网格上
  画出来的；`source: volume` 的 2.6 × 2.6 × 32 µm 网格上一个细胞不到一个体素，
  框会挤成一团。volume 模式适合 `--funnel` 和看框的整体分布。
- **`--funnel` 的归区口径和统计的不完全一样。** 漏斗里每一级都是用
  `labels_in_sample` 回查框中心（四级和细胞表统一口径，否则少掉的那几个说不清是
  真少了还是换了个查法）；细胞表那一行用的是它自己第 9 列（细胞推进图谱空间查的
  那个答案），和统计一致。两者差几个百分点是正常的，差很多要先查配准。
- **内存和文件大小无关。** `2_global_2d_raw` 是一个框一层一行，整脑几千万行；
  这里所有读取都是分块的，看图时只留落在 site 框里的行，对账时一行都不留。
- **s18 跑的是重定位过的 run。** 碎片上的细胞，画出来的脑区边界是合拢后的位置，
  和原图对不上（session 会警告）。选区按第 9 列仍然正确，判碎片上的细胞时不要看
  边界。


## 从全局坐标追溯原始文件

`view_detection_qc.py` 右侧的定位框接受完整 `cell_id`（如 `neuron_RFP:88`），
或细胞表前 3 列对应的全局像素 `x, y, z`。坐标输入按物理距离查找**最近的已配准
细胞**，显示距离，并以该细胞为中心打开局部图像；不是任意坐标裁图，也不能用来
抽样没有被检出的细胞。查找覆盖整张细胞表，不受 `regions` / `classes` 抽样过滤
限制，原有区域图层仍保留当前配置的选区。新位置加入本次会话的翻页列表，判定
仍按真实 cell_id 保存。新位置需要扫描检测 CSV，大文件可能等待较久。

使用 `source: tiles` 并配置各通道 `channels` 后，右侧同时显示目标点对应的
原始 TIFF 路径及 tile 局部 `x, y, z`（均从 0 起，z 是最近切片的文件序号）。
重叠 tile 会全部列出；通道 `offset_px` 也参与追溯。路径可选中复制。
这里的输入不是图谱坐标，也不是配准降采样图的体素坐标。

建议先用 `--funnel` 检查共定位结果与最终细胞表对账，再在窗口逐层切换 s2、s3、
coloc 和脑区边界，核对检测、合并、共定位和归区。碎片重定位样本的边界限制仍然
适用：原始文件定位使用原始细胞坐标，但当前 QC 尚未把脑区边界复位到碎片上。
