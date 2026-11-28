"""
Component ids, option lists and shared style constants for the Explorer page.

Imports no other Explorer module, so every layer can import from it.
"""

from __future__ import annotations

# URL path for the Explorer page.
EXPLORER_PATH = "/explorer"
# The Explorer's entry in the sidebar search
EXPLORER_SEARCH_ENTRY = {
    "label": "Explorer",
    "kind": "Page",
    "href": EXPLORER_PATH,
    "search_text": "Explorer model finder report card head-to-head compare",
}

# Component ids: the layout renders them, register_explorer_callbacks wires them.
REPORT_MODEL = "explorer-report-model"
REPORT_SUMMARY = "explorer-report-summary"
REPORT_PROFILE = "explorer-report-profile"
REPORT_TABLE = "explorer-report-table"
STRUCT_VIEWER = "explorer-struct-viewer"
STRUCT_TABLE = "explorer-struct-table"
H2H_MODEL_A = "explorer-h2h-model-a"
H2H_MODEL_B = "explorer-h2h-model-b"
H2H_GRAPH = "explorer-h2h-graph"
H2H_DETAIL_CATEGORY = "explorer-h2h-detail-category"
H2H_DETAIL_GRAPH = "explorer-h2h-detail-graph"
H2H_SUMMARY = "explorer-h2h-summary"
COVERAGE_GRAPH = "explorer-coverage-graph"
COVERAGE_GRANULARITY = "explorer-coverage-granularity"
FINDER_ELEMENTS = "explorer-finder-elements"
FINDER_CATEGORIES = "explorer-finder-categories"
FINDER_TOLERANCE = "explorer-finder-tolerance"
FINDER_PRESET = "explorer-finder-preset"
FINDER_GRAPH = "explorer-finder-graph"
FINDER_PICKS = "explorer-finder-picks"
STRUCT_FAILS_CATEGORIES = "explorer-fails-categories"
STRUCT_FAILS_MODEL = "explorer-fails-model"
STRUCT_FAILS_SORT = "explorer-fails-sort"
STRUCT_FAILS_GRAPH = "explorer-fails-graph"
STRUCT_FAILS_LIST = "explorer-fails-list"
SIMILARITY_GRAPH = "explorer-similarity-graph"

# Pattern-matched id types for the finder pick buttons and failure list buttons
FINDER_PICK_TYPE = "explorer-finder-pick"
FAILS_PICK_TYPE = "explorer-fails-pick"

# Granularity for the coverage rose: one wedge per category, or one per benchmark.
COVERAGE_GRANULARITY_OPTIONS = [
    {"label": "By category", "value": "category"},
    {"label": "By benchmark", "value": "benchmark"},
]

# Field-wide failure rankings; ignored when a model is picked
STRUCT_FAILS_SORT_OPTIONS = [
    {"label": "Hardest (most models fail)", "value": "hardest"},
    {"label": "Most contested (models disagree)", "value": "contested"},
]

# Level-of-theory classes, in severity order. Red and green are kept for scores.
LOT_CLASS_STYLE = {
    # Mirrors --mlpeg-accent (theme.css): Plotly cannot read CSS variables server-side,
    # so keep the two in sync.
    "in_domain": {"color": "#0d8ff0", "label": "in-domain"},
    "dft": {"color": "#8b5cf6", "label": "functional mismatch"},
    "high_level": {"color": "#d946ef", "label": "higher-level reference"},
    "experimental": {"color": "#64748b", "label": "vs experiment"},
    "none": {"color": "#9aa0a6", "label": "unclassified"},
}
LOT_CLASSES = tuple(LOT_CLASS_STYLE)

# Head-to-head winner colours (A = blue, B = amber, tie = grey). Semantic data colours,
# so theme_plots.js leaves them alone; each clears 3:1 on both plot backgrounds.
H2H_COLOUR_A = "#4c6ef5"
H2H_COLOUR_B = "#f59f00"
H2H_COLOUR_TIE = "#b0b6bd"

FINDER_TOLERANCE_OPTIONS = [
    {"label": "In-domain evidence only", "value": "in_domain"},
    {"label": "Allow DFT-functional mismatch", "value": "dft"},
    {"label": "Use all evidence", "value": "any"},
]

# Finder experiment presets, each selecting a bundle of category directories
FINDER_EXPERIMENTS = [
    {"label": "Anything", "value": "any"},
    {"label": "Bulk materials & crystals", "value": "bulk"},
    {"label": "Surfaces, defects & reactions", "value": "surfaces"},
    {"label": "Molecules & non-covalent chemistry", "value": "molecular"},
    {"label": "Metals & f-block chemistry", "value": "metals"},
]
FINDER_EXPERIMENT_CATEGORIES: dict[str, list[str]] = {
    "any": [],
    "bulk": [
        "bulk_crystal",
        "molecular_crystal",
        "molecular_dynamics",
        "alloy_metallurgy",
        "amorphous_materials",
        "porous_materials",
        "electrolytes",
        "fracture",
    ],
    "surfaces": ["surfaces", "defects", "nebs", "molecular_reactions"],
    "molecular": [
        "molecular",
        "conformers",
        "isomers",
        "thermochemistry",
        "non_covalent_interactions",
        "supramolecular",
        "biomolecules",
    ],
    "metals": ["tm_complexes", "f_block", "alloy_metallurgy"],
}
