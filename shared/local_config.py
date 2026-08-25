"""Shared helper: 每个工具的真实路径都放在 configs/<tool>.yaml 里，既不写死在
代码里、也不散落在仓库根目录 —— 换一个 sample 只改 yaml，不会变成 git diff。

约定（所有工具一致）：

    configs/<tool>.example.yaml    模板，提交进 git
    configs/<tool>.yaml            真实路径，gitignored，每台设备各自一份

每个工具都接受一个可选的位置参数指向别的 config，所以同一台机器上可以给不同
样本各留一份，互相不覆盖：

    python single_sample.py configs/single_sample.local.yaml
    python tools/atlas_view.py configs/atlas_view.devccf.yaml

configs/ 和 .dialog_state/ 都在仓库根目录，不在这个包里 —— 下面的路径锚点用的是
`parents[1]`，所以 tools/ 里的工具和根目录的主脚本找到的是同一份配置。

字符串值里的 `~` 和 `${VAR}` 会被展开，方便同一份 config 在数据盘挂载点不同的
两台机器之间共用。未定义的 `${VAR}` 原样保留（不会静默变成空字符串），这样出错
时报出来的是那个没展开的路径本身，一眼能看出是哪个变量没设。

弹表单的交互式工具（tools/edit_sample_labels.py）走 `resolve_inputs()`：配置是
**可选**的，有就用来预填表单（优先于 .dialog_state/ 里上次用过的值），`--no-form`
则完全不弹窗、直接用配置。其余工具（single_sample.py / paint_mask.py /
tools/atlas_view.py / tools/qc_guide_mask.py）直接用 `load_config()`。

Not runnable on its own -- imported by paint_mask.py, single_sample.py and the
tools in tools/.
"""
import os
import sys
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def example_path(tool):
    return CONFIGS_DIR / f"{tool}.example.yaml"


def default_path(tool):
    return CONFIGS_DIR / f"{tool}.yaml"


def add_config_arg(parser, tool):
    """给一个 argparse.ArgumentParser 加上所有工具共用的可选 config 位置参数。"""
    parser.add_argument(
        "config", nargs="?", default=None,
        help=f"配置文件路径（默认 configs/{tool}.yaml，模板见 configs/{tool}.example.yaml）")


def add_no_form_arg(parser):
    """给弹表单的交互式工具加上 --no-form（直接按配置跑，不弹窗）。"""
    parser.add_argument(
        "--no-form", action="store_true",
        help="不弹表单，直接用配置文件里的值（此时配置文件必须存在且填全）")


def _expand(value):
    """递归展开字符串里的 ~ 和 ${VAR}；其它类型原样返回。"""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def _invocation(tool):
    """错误信息里那条"再跑一遍"的命令。工具分散在仓库根目录和 tools/ 两处，写死
    `{tool}.py` 会给出一条 tools/ 里的工具根本跑不起来的命令，所以优先用实际的
    argv[0]。"""
    argv0 = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if argv0 and argv0.suffix == ".py":
        try:
            return argv0.resolve().relative_to(CONFIGS_DIR.parent)
        except ValueError:
            return argv0
    return f"{tool}.py"


def _missing_message(tool, legacy_paths=()):
    default = default_path(tool)
    example = example_path(tool)
    lines = [
        f"找不到配置文件 {default}",
        "",
        f"    cp {example.relative_to(example.parent.parent)} {default.relative_to(default.parent.parent)}",
        "",
        "然后在里面改成本机的真实路径。configs/*.yaml 是 gitignored 的，只有",
        "*.example.yaml 会进 git，所以换样本改路径不会产生 git diff。",
        f"也可以显式指定别的配置：python {_invocation(tool)} path/to/other.yaml",
    ]
    if legacy_paths:
        lines.append(f"（旧位置 {', '.join(str(p) for p in legacy_paths)} 也不存在。）")
    return "\n".join(lines)


def resolve_config_path(tool, cli_path=None, legacy_paths=()):
    """按 显式路径 > configs/<tool>.yaml > 旧位置 的顺序找配置文件。

    都找不到时返回 None（由 load_config 决定是报错还是当作"没有配置"）。
    显式给了却不存在则直接报错 —— 那是打错了路径，静默回退到默认配置会让人
    以为自己指定的那份生效了。
    """
    if cli_path:
        path = Path(cli_path)
        if not path.exists():
            raise FileNotFoundError(f"指定的配置文件不存在: {path}")
        return path

    path = default_path(tool)
    if path.exists():
        return path

    for legacy in legacy_paths:
        legacy = Path(legacy)
        if legacy.exists():
            print(f"⚠️  [config] {legacy} 是旧位置，仍然能读，但建议迁移到统一目录：\n"
                  f"    mv {legacy} {default_path(tool)}")
            return legacy
    return None


def load_config(tool, cli_path=None, required=(), legacy_paths=(), optional=False):
    """读取并校验 <tool> 的配置，返回 dict。

    optional=True 时，没有配置文件就返回 {}（给那些本来就有 GUI 表单、config
    只是用来预填的工具用）；显式指定了路径的话仍然必须存在。
    """
    path = resolve_config_path(tool, cli_path, legacy_paths)
    if path is None:
        if optional:
            return {}
        raise FileNotFoundError(_missing_message(tool, legacy_paths))

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} 的顶层内容应该是 key: value 映射，实际是 {type(cfg).__name__}")

    cfg = _expand(cfg)
    missing = [k for k in required if cfg.get(k) in (None, "")]
    if missing:
        raise ValueError(
            f"{path} 缺少必填项: {', '.join(missing)}\n"
            f"（每一项的含义见 {example_path(tool)}）")

    print(f"[config] {path}")
    return cfg


def _field_enabled(field, values):
    """enabled_when=(other_key, expected) —— 另一个字段不等于 expected 时本字段
    不生效（跟 form_dialog.run_form 里灰掉输入框是同一套语义）。"""
    cond = field.get("enabled_when")
    return cond is None or values.get(cond[0]) == cond[1]


def values_from_config(tool, fields, cfg):
    """不弹表单时，直接把配置铺成 {key: value}，并做表单本来会做的必填校验。"""
    if not cfg:
        raise FileNotFoundError(
            _missing_message(tool) + "\n\n（--no-form 跳过了表单，所以配置文件必须存在。）")

    values = {f["key"]: cfg.get(f["key"], f.get("default")) for f in fields}
    missing = []
    for f in fields:
        if not _field_enabled(f, values):
            values[f["key"]] = None       # 跟表单里灰掉的字段一样解析成 None
        elif not f.get("optional") and values[f["key"]] in (None, ""):
            missing.append(f["key"])
    if missing:
        raise ValueError(
            f"--no-form 需要配置里填全这些项: {', '.join(missing)}\n"
            f"（每一项的含义见 {example_path(tool)}）")

    unknown = set(cfg) - {f["key"] for f in fields}
    if unknown:
        print(f"[config] 忽略了 {tool} 用不到的字段: {', '.join(sorted(unknown))}")
    return values


def resolve_inputs(tool, title, fields, cli_path=None, no_form=False):
    """所有交互式工具取参数的统一入口：配置文件（可选）→ 表单 → {key: value}。

    有 configs/<tool>.yaml 时，里面写了的字段**优先于** .dialog_state/ 里上次
    用过的值 —— 否则改了 config 却看不出效果。没有配置文件时行为跟以前完全一样
    （空表单 + 上次的值）。--no-form 则完全不弹窗，直接用配置里的值。
    """
    cfg = load_config(tool, cli_path, optional=True)
    if no_form:
        return values_from_config(tool, fields, cfg)
    from shared.form_dialog import run_form   # 惰性导入：--no-form 这条路不需要 PyQt5
    return run_form(tool, title, fields, overrides=cfg)
