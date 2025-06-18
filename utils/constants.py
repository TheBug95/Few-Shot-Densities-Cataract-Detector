from pathlib import Path
from enum import Enum, auto
from torchvision import models
import torch

# ─── rutas base ────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent

DATASETS_CATARACT_DIR   = ROOT_DIR / "datasets" / "cataract_detection"
DATASETS_PROCESSED_CATARACT_DIR = ROOT_DIR / "datasets" / "cataract_detection" / "processed"

MODELS_BACKBONES_DIR = ROOT_DIR / "models" / "backbones"
MODELS_PROTOTYPES_DIR = ROOT_DIR / "models" / "prototypes"
MODELS_SAM_ZOO_DIR = ROOT_DIR / "models" / "sam zoo"

# ─── parámetros de entrenamiento ──────────────────────────────────
INPUT_SIZE   = 224
BATCH_SIZE   = 32
NUM_CLASSES  = 2
RANDOM_SEED  = 42

# ─── rutas de entrenamiento y validación ─────────────────────────
CATARACT_TRAIN_SPLIT  = ROOT_DIR / "datasets" / "cataract_detection" / "train"
CATARACT_VALID_SPLIT  = ROOT_DIR / "datasets" / "cataract_detection" / "valid"
CATARACT_TEST_SPLIT   = ROOT_DIR / "datasets" / "cataract_detection" / "test"

# ─── rutas de segmentaciones COCO ─────────────────────────────
CATARACT_COCO_TRAIN_SPLIT  = ROOT_DIR / "datasets" / "cataract_detection" / "train" / "_annotations.coco.json"
CATARACT_COCO_VALID_SPLIT  = ROOT_DIR / "datasets" / "cataract_detection" / "valid" / "_annotations.coco.json"
CATARACT_COCO_TEST_SPLIT   = ROOT_DIR / "datasets" / "cataract_detection" / "test" / "_annotations.coco.json"

# ─── categorias dataset  ──────────────────────
NORMAL_CAT_ID = 2
MILD_CAT_ID = 1
SEVERE_CAT_ID = 3

# ─── dispositivo ─────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    
class BandwidthMethod(str, Enum):
    SCOTT = "scott"
    SILVERMAN = "silverman"
    
class BackbonesWeights(str, Enum):
    RESNET18 = models.ResNet18_Weights.DEFAULT
    RESNET34 = models.ResNet34_Weights.DEFAULT
    VIT_B16  = models.ViT_B_16_Weights.DEFAULT
    
class DatasetCataractSplit(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST  = "test"
    