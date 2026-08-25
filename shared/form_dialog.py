"""Shared helper: a single Qt form window for collecting a script's inputs
(file paths, folders, numbers, choices) instead of CLI arguments. Every
field is pre-filled with whatever value was used the last time that script
ran, so re-running with the same paths is just "click OK" -- state is kept
per-script in .dialog_state/<script_name>.json (gitignored, so it
never shows up as a git diff).

Not runnable on its own, and normally not imported directly either: the
interactive tools call local_config.resolve_inputs(), which layers
configs/<tool>.yaml on top of the remembered values and calls run_form() here
(or skips it entirely for --no-form). Import this module directly only if a
tool wants the form with no config layer at all.
"""
import json
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
    QMessageBox, QPushButton, QSpinBox, QWidget,
)

_STATE_DIR = Path(__file__).resolve().parents[1] / ".dialog_state"

# Held here for the life of the process -- QApplication has no Python
# reference otherwise and gets garbage-collected right after the `or`
# expression in run_form(), aborting the next widget construction with
# "QWidget: Must construct a QApplication before a QWidget".
_app = None


def _state_path(script_name):
    return _STATE_DIR / f"{script_name}.json"


def _load_state(script_name):
    p = _state_path(script_name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(script_name, state):
    _STATE_DIR.mkdir(exist_ok=True)
    _state_path(script_name).write_text(json.dumps(state, indent=2))


class _PathField(QWidget):
    """A QLineEdit + "Browse..." button, for open/save file or folder picks."""

    def __init__(self, initial, mode, filter_str):
        super().__init__()
        self.mode = mode
        self.filter_str = filter_str
        self.edit = QLineEdit(initial)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit)
        layout.addWidget(browse)

    def _browse(self):
        start = self.edit.text() or str(Path.home())
        if self.mode == "open":
            path, _ = QFileDialog.getOpenFileName(self, "Select file", start, self.filter_str)
        elif self.mode == "save":
            path, _ = QFileDialog.getSaveFileName(self, "Select output file", start, self.filter_str)
        else:
            path = QFileDialog.getExistingDirectory(self, "Select folder", start)
        if path:
            self.edit.setText(path)

    def value(self):
        return self.edit.text().strip()


def run_form(script_name, title, fields, overrides=None):
    """Show one form window with all `fields` and return {key: value}.

    Each field dict has: key, label, type (one of "open_file", "save_file",
    "directory", "text", "choice", "int", "float", "checkbox"), plus
    type-specific keys: filter (path types), placeholder (text),
    options/default (choice), default/minimum/maximum/decimals (int/float),
    default (checkbox), optional (skip the required-value check),
    enabled_when=(other_key, expected_value) to grey the field out unless
    another field currently equals expected_value.

    `overrides` (normally configs/<script_name>.yaml, passed in by
    local_config.resolve_inputs) pre-fills fields ahead of the remembered
    last-used values -- editing the config has to be visible in the form, or
    there would be no point editing it. Fields absent from `overrides` still
    fall back to .dialog_state/ and then to the field's own default.

    Disabled fields resolve to None in the result. Raises SystemExit if the
    user cancels the dialog.
    """
    global _app
    state = _load_state(script_name)
    # `state` stays the thing that gets written back, so config values do not
    # leak into .dialog_state/; `prefill` is only what the widgets start at.
    prefill = {**state, **{k: v for k, v in (overrides or {}).items() if v is not None}}
    _app = QApplication.instance() or QApplication([])

    dialog = QDialog()
    dialog.setWindowTitle(title)
    dialog.setMinimumWidth(560)
    layout = QFormLayout(dialog)

    widgets = {}
    for f in fields:
        key = f["key"]
        remembered = prefill.get(key, f.get("default"))
        ftype = f["type"]
        if ftype in ("open_file", "save_file", "directory"):
            mode = {"open_file": "open", "save_file": "save", "directory": "dir"}[ftype]
            w = _PathField(remembered or "", mode, f.get("filter", "All files (*)"))
        elif ftype == "text":
            # Free text, for the values that are neither a path nor a single
            # number -- e.g. a "2.6 2.6 32.0" voxel size. Parsing is the
            # caller's; this only collects the string.
            w = QLineEdit("" if remembered is None else str(remembered))
            if f.get("placeholder"):
                w.setPlaceholderText(f["placeholder"])
        elif ftype == "choice":
            w = QComboBox()
            w.addItems(f["options"])
            if remembered in f["options"]:
                w.setCurrentText(remembered)
        elif ftype == "int":
            w = QSpinBox()
            w.setRange(f.get("minimum", 0), f.get("maximum", 1_000_000))
            w.setValue(int(remembered if remembered is not None else f.get("default", 0)))
        elif ftype == "float":
            w = QDoubleSpinBox()
            w.setRange(f.get("minimum", 0.0), f.get("maximum", 1_000_000.0))
            w.setDecimals(f.get("decimals", 2))
            w.setValue(float(remembered if remembered is not None else f.get("default", 0.0)))
        elif ftype == "checkbox":
            w = QCheckBox()
            w.setChecked(bool(remembered if remembered is not None else f.get("default", False)))
        else:
            raise ValueError(f"Unknown field type: {ftype!r}")
        widgets[key] = w
        layout.addRow(f["label"], w)

    def current_value(widget):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        return None

    def apply_conditions():
        for f in fields:
            cond = f.get("enabled_when")
            if cond:
                other_key, expected = cond
                widgets[f["key"]].setEnabled(current_value(widgets[other_key]) == expected)

    for f in fields:
        cond = f.get("enabled_when")
        if cond:
            other = widgets[cond[0]]
            signal = other.toggled if isinstance(other, QCheckBox) else other.currentTextChanged
            signal.connect(lambda _=None: apply_conditions())
    apply_conditions()

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    layout.addRow(buttons)

    def on_accept():
        for f in fields:
            w = widgets[f["key"]]
            if f.get("optional") or not w.isEnabled():
                continue
            blank = ((isinstance(w, _PathField) and not w.value())
                     or (isinstance(w, QLineEdit) and not w.text().strip()))
            if blank:
                QMessageBox.warning(dialog, "Missing value", f"'{f['label']}' is required.")
                return
        dialog.accept()

    buttons.accepted.connect(on_accept)
    buttons.rejected.connect(dialog.reject)

    if dialog.exec_() != QDialog.Accepted:
        raise SystemExit("Cancelled -- no changes made.")

    result = {}
    for f in fields:
        key = f["key"]
        w = widgets[key]
        if not w.isEnabled():
            result[key] = None
            continue
        if isinstance(w, _PathField):
            value = w.value() or None
        elif isinstance(w, QLineEdit):
            value = w.text().strip() or None
        elif isinstance(w, QComboBox):
            value = w.currentText()
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            value = w.value()
        elif isinstance(w, QCheckBox):
            value = w.isChecked()
        result[key] = value
        state[key] = value

    _save_state(script_name, state)
    return result
