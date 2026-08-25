"""Modules that are imported, never run: the pieces more than one tool needs.

    local_config      configs/<tool>.yaml -> a dict, plus the form-window path
    form_dialog       the Qt input form local_config falls back to
    landmark_io       the landmark CSV format the tools pass between each other
    atlas_reference   GUI-free atlas loading + ontology math
    ontology_tree_ui  the searchable Qt ontology tree both region panels use

Nothing here has a main(); the runnable tools live in the repo root
(paint_mask.py / single_sample.py / registration_eval.py) and in tools/.

Anything __file__-relative in here must anchor on `parents[1]` -- the repo
root, one level up from this package -- not on this directory. configs/ and
.dialog_state/ both live at the root, and a tool in tools/ has to find the
same ones a tool in the root does.
"""
