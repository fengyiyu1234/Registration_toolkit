# QC visualization implementation status

更新时间：2026-09-21

本文记录当前工作区中针对 `QC_VISUALIZATION_PLAN.md` 已实现的内容、验证情况和仍存在的问题。它是对 `QC_VISUALIZATION_PROGRESS.yaml` 的人类可读补充；YAML 仍作为 Codex 追踪状态的主文件。

## 已实现

### 1. 坐标契约接入

- `qc/coordinate_contract.py` 支持标量和 `N×3` 坐标，并统一处理 `origin`、各向异性 `spacing`、`direction`。
- `qc/crop_geometry.py` 的 global-pixel ↔ label-voxel 转换复用该坐标契约，同时保留旧调用的默认 origin-zero、identity direction 行为。
- `qc/region_cells.py::LabelVolume` 从 NIfTI affine 提取 LPS 下的 origin、spacing、direction。
- `LabelVolume.lookup`、`box`、`resample_to` 和 session 的重采样列检查改为使用完整坐标变换。
- 新增非零 origin、各向异性 spacing、非默认 direction 的合成测试。
- Napari label layer 使用与标签元数据对应的 affine，而不是只使用 scale/translate。

涉及文件：

- `qc/coordinate_contract.py`
- `qc/crop_geometry.py`
- `qc/region_cells.py`
- `qc/view_detection_qc.py`
- `qc/view_region_cells.py`
- `tests/test_region_qc_smoke.py`

### 2. 任意全局坐标作为直接查看目标

- `DetectionSession.locate_coordinate()` 支持不依赖已有 cell id 的全局坐标。
- 直接坐标 target 保存用户输入的精确坐标；最近细胞只作为上下文信息，不替换裁图中心。
- GUI 定位框支持 `cell_id` 或 `x, y, z` 全局像素坐标。
- 直接坐标 target 会进入现有 detection crop、标签和来源 tile 查询路径。
- CLI 对指定坐标输出每个 source grid 的覆盖位置以及物理查看窗口。

涉及文件：

- `qc/view_detection_qc.py`
- `qc/qc_visualization.py`
- `tests/test_detection_qc_smoke.py`

### 3. 三维人工判定统一写入 verdicts.csv

- `VerdictStore` 保留旧版 CSV 读取能力，并补充 schema/version、target key、source 信息及三列判定：
  - `detection_verdict`
  - `colocalization_verdict`
  - `registration_verdict`
- 保留旧的 `verdict` 字段，兼容已有读取逻辑。
- 同一 target 重判时按稳定 target key 更新，而不是无限追加重复记录。
- coordinate target 使用稳定坐标键，不依赖空的 cell id。
- GUI 新增三维判定控件，并把备注、来源和旧版 verdict 一起写入 CSV。
- CSV 更新采用临时文件替换，降低写入中断导致半文件的风险。

涉及文件：

- `qc/region_cells.py`
- `qc/view_detection_qc.py`
- `tests/test_detection_qc_smoke.py`

### 4. per-tile alignment shift 测试覆盖

- 保留远端已有的 `load_tile_shifts()`、`TileGrid(..., tile_shifts=...)` 和各 QC 调用方接入。
- 新增合成 tile、alignment JSON、`locate()` 和 `cut()` 的联动测试，验证 `dx/dy/dz` 在来源像素和裁图结果中一致。

涉及文件：

- `qc/crop_geometry.py`
- `tests/test_qc_crops_smoke.py`

## 当前验证结果

已执行：

- `conda run -n antsreg python -m py_compile qc/*.py`：通过。
- `conda run -n antsreg python tests/test_region_qc_smoke.py`：7 passed。
- `conda run -n antsreg python tests/test_detection_qc_smoke.py`：11 passed。

crop smoke 在补充测试首次运行时发现测试文件缺少 `json` 导入；该测试夹具问题已补上，尚未在本次文档更新前重新运行。

完整 `tests/test_gui_smoke.py` 的已知失败不属于本次 QC 改动：失败点是 `paint_mask` 控件最小高度断言（当前 68，测试期望不超过 48）。之前同一测试进程中的 detection QC 定位部分已通过。

## 仍存在的问题

### 高优先级

1. 坐标约定需要补充文档化验证。

   当前实现已接入完整 affine，但 LPS/RAS 转换、数组轴顺序和 Napari world axis 的约定仍主要依赖代码与测试。需要在计划文档或 README 中写出一个完整的坐标示例，并补充旋转标签在 Napari 中的 GUI 验证。

2. `qc_visualization.py` 的 CLI 仍是定位/证据清单入口，不是独立的图形查看器。

   GUI 直达 coordinate target 已实现；CLI 目前输出 source locations 和物理窗口，但不会自行保存 crop 图像，也不会自动启动 Napari。若计划要求命令行本身产出静态 crop，需要继续增加明确的输出参数和 headless crop writer。

3. `tile_shifts` 的输入校验和配置说明还不完整。

   当前合成测试覆盖了正确 JSON 的 `dx/dy/dz`，但加载时尚未系统验证字段缺失、非有限值、非整数值；`qc_crops`、`region_qc`、`detection_qc` 的配置模板/README 也需要统一说明单位、符号和 `align_key`。

### 中优先级

4. 需要完成 GUI 端到端验证。

   应单独验证 coordinate target 在 detection viewer 中的裁图中心、来源 tile 列表、labels affine 和三维判定回填。当前已有 headless 覆盖，完整 GUI smoke 仍被 paint_mask 独立回归阻断。

5. `VerdictStore` 的旧 CSV 兼容需要针对更多历史变体补测试。

   当前覆盖旧字段读取、三维写入和重复 target 更新；还应覆盖缺列旧 CSV、coordinate target 重启后读取、并发/异常替换失败时的恢复策略。

6. 需要清理遗留的旧 JSON verdict 接口。

   `qc/qc_visualization.py` 中的旧 `append_verdict` 仍存在时，应明确标记为兼容接口、改为委托 `VerdictStore`，或删除并更新所有引用，避免后续重新产生第二套判定格式。

## 建议的下一步

1. 重新运行 crop smoke，确认新增 tile shift 合成测试通过。
2. 增加 `tile_shifts` 输入校验和三个配置入口的文档。
3. 把 CLI 是否需要实际写出 crop 图像作为计划决策；若需要，补 `--output-dir`/headless writer。
4. 在 GUI smoke 中单独跑 detection QC 相关测试，或先修复 `paint_mask` 的独立尺寸回归后再跑完整套件。
5. 完成后同步更新 `QC_VISUALIZATION_PROGRESS.yaml` 中各 issue 的 status、证据和最终验收结果。

