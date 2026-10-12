"""Benthic classes used in the study.

Class IDs and names follow the pointcoral label crosswalk. IDs 3 and 9-14 are pointcoral
classes that were excluded from the analysis, so the eight retained IDs are not consecutive.
"""

CLASS_NAMES = {
    0: "CORAL (C)",
    1: "GORGONIANS (G)",
    2: "SPONGES (S)",
    4: "MACROALGAE (MA)",
    5: "PEYSSONNELIACEAE",
    6: "CALCAREOUS GREEN",
    7: "CORALLINE ALGAE (CA)",
    8: "SAND, PAVEMENT, RUBBLE (SPR)",
}

CLASS_COLORS = {
    0: "#ff4d4d",
    1: "#ff944d",
    2: "#3399ff",
    4: "#33cc66",
    5: "#cc3366",
    6: "#7fbf3f",
    7: "#8b0000",
    8: "#d8c28a",
}

EXCLUDED_ML_CLASSES = {
    "ZOANTHIDS (Z)",
    "DISEASED CORALS (DC)",
    "OTHER LIVE (OL)",
    "DEAD CORAL WITH ALGAE (DCA)",
    "ASCIDIAN",
    "TAPE, WAND, SHADOW (TWS)",
    "UNKNOWNS (U)",
}

EXCLUDED_CLASS_IDS = {3, 9, 10, 11, 12, 13, 14}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
