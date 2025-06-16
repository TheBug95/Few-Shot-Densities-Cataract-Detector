from pathlib import Path
from enum import Enum, auto

# ─── rutas base ────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent

DATASETS_CATARACT_DIR   = ROOT_DIR / "datasets" / "cataract_detection"
DATASETS_PROCESSED_CATARACT_DIR = ROOT_DIR / "datasets" / "cataract_detection" / "processed"

MODELS_BACKBONES_DIR = ROOT_DIR / "models" / "backbones"
MODELS_PROTOTYPES_DIR = ROOT_DIR / "models" / "prototypes"
MODELS_SAM_ZOO_DIR = ROOT_DIR / "models" / "sam_zoo"

# ─── parámetros de entrenamiento ──────────────────────────────────
INPUT_SIZE   = 224
BATCH_SIZE   = 32
NUM_CLASSES  = 2
RANDOM_SEED  = 42

# ─── enumeraciones útiles ─────────────────────────────────────────
class Backbone(str, Enum):
    RESNET18 = "r18"
    RESNET34 = "r34"
    VIT_B16  = "vit_b_16"
    
class BinningMethod(str, Enum):
    SCOTT = "scott"
    STURGES = "sturges"
    FREEDMAN_DIACONIS = "freedman_diaconis"
    AUTO = "auto"
    FIXED_20 = "fixed_20"
    FIXED_30 = "fixed_30"