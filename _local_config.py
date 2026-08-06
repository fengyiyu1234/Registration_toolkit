"""Shared helper: 每个工具的真实路径都放在 configs/<tool>.yaml 里，既不写死在
代码里、也不散落在仓库根目录 —— 换一个 sample 只改 yaml，不会变成 git diff。

约定（所有工具一致）：

    configs/<tool>.example.yaml    模板，提交进 git
    configs/<tool>.yaml            真实路径，gitignored，每台设备各自一份

每个工具都接受一个可选的位置参数指向别的 config，所以同一台机器上可以给不同
样本各留一份，互相不覆盖：

    python single_sample.py configs/single_sample.s12t.yaml

字符串值里的 `~` 和 `${VAR}` 会被展开，方便同一份 config 在数据盘挂载点不同的
两台机器之间共用。未定义的 `${VAR}` 原样保留（不会静默变成空字符串），这样出错
时报出来的是那个没展开的路径本身，一眼能看出是哪个变量没设。

Not runnable on its own -- imported by single_sample.py, paint_mask.py,
place_landmarks.py, edit_sample_labels.py.
"""
import os
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def example_path(tool):
    return CONFIGS_DIR / f"{tool}.example.yaml"


def default_path(tool):
    return CONFIGS_DIR / f"{tool}.yaml"


def add_config_arg(parser, tool):
    """给一个 argparse.ArgumentParser 加上所有工具共用的可选 config 位置参数。"""
    parser.add_argument(
        "config", nargs="?", default=None,
        help=f"配置文件路径（默认 configs/{tool}.yaml，模板见 configs/{tool}.example.yaml）")


def _expand(value):
    """递归展开字符串里的 ~ 和 ${VAR}；其它类型原样返回。"""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


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
        f"也可以显式指定别的配置：python {tool}.py path/to/other.yaml",
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

    with open(path) as f:
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
