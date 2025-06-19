from pathlib import Path
from enum import Enum, auto
from torchvision import models
import torch

# ─── rutas base ────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent

DATASETS_CATARACT_DIR   = ROOT_DIR / "datasets" / "cataract_detection"
DATASETS_PROCESSED_CATARACT_DIR = ROOT_DIR / "datasets" / "cataract_detection" / "processed" / "images" 

MODELS_BACKBONES_DIR = ROOT_DIR / "models" / "backbones"
MODELS_PROTOTYPES_DIR = ROOT_DIR / "models" / "prototypes"
MODELS_SAM_ZOO_DIR = ROOT_DIR / "models" / "sam zoo"

# ─── parámetros de entrenamiento ──────────────────────────────────
INPUT_SIZE   = 224
BATCH_SIZE   = 32
NUM_CLASSES  = 2
RANDOM_SEED  = 42
ITERATIONS   = 10

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
    FREEDMAN_DIACONIS = "fd"
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

class ProtosFileNames(str, Enum):
    KDE_AUTO_R34 = "kde_univar_protos_auto_r34_seed42.pkl"
    KDE_SCOTT_R34 = "kde_univar_protos_scott_r34_seed42.pkl"
    KDE_FIXED_30_VIT_B_16 = "kde_univar_protos_fixed_30_vit_b_16_seed42.pkl"
    KDE_STURGES_R34 = "kde_univar_protos_sturges_r34_seed42.pkl"
    KDE_FIXED_30_R34 = "kde_univar_protos_fixed_30_r34_seed42.pkl"
    KDE_STURGES_VIT_B_16 = "kde_univar_protos_sturges_vit_b_16_seed42.pkl"
    KDE_FD_R18 = "kde_univar_protos_fd_r18_seed42.pkl"
    KDE_FD_R34 = "kde_univar_protos_fd_r34_seed42.pkl"
    KDE_FD_VIT_B_16 = "kde_univar_protos_fd_vit_b_16_seed42.pkl"
    KDE_FIXED_20_VIT_B_16 = "kde_univar_protos_fixed_20_vit_b_16_seed42.pkl"
    KDE_AUTO_VIT_B_16 = "kde_univar_protos_auto_vit_b_16_seed42.pkl"
    KDE_SCOTT_VIT_B_16 = "kde_univar_protos_scott_vit_b_16_seed42.pkl"
    KDE_SCOTT_R18 = "kde_univar_protos_scott_r18_seed42.pkl"
    KDE_AUTO_R18 = "kde_univar_protos_auto_r18_seed42.pkl"
    KDE_FIXED_30_R18 = "kde_univar_protos_fixed_30_r18_seed42.pkl"
    KDE_FIXED_20_R34 = "kde_univar_protos_fixed_20_r34_seed42.pkl"
    KDE_FIXED_20_R18 = "kde_univar_protos_fixed_20_r18_seed42.pkl"
    KDE_STURGES_R18 = "kde_univar_protos_sturges_r18_seed42.pkl"
    
"""# ─── Variables Fijas para la Inferencia con Resnet-18 ───────────

EXPERS_R18 = [
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_FIXED_30_R18.value,
        extractor = train_resnet_18_fixed_30._get_embedding,
        excel   = "fixed_30_r18",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_FIXED_20_R18.value,
        extractor = train_resnet_18_fixed_20._get_embedding,
        excel   = "fixed_20_r18",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_SCOTT_R18.value,
        extractor = train_resnet_18_scott._get_embedding,
        excel   = "scott_r18",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_STURGES_R18.value,
        extractor = train_resnet_18_sturges._get_embedding,
        excel   = "sturges_r18",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_AUTO_R18.value,
        extractor = train_resnet_18_auto._get_embedding,
        excel   = "auto_r18",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_FD_R18.value,
        extractor = train_resnet_18_fd._get_embedding,
        excel   = "fd_r18",
    )
]

# ─── Variables Fijas para la Inferencia con Resnet-34 ───────────

EXPERS_R34 = [
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_FIXED_30_R34.value,
        extractor = train_resnet_34_fixed_30._get_embedding,
        excel   = "fixed_30_r34",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_FIXED_20_R34.value,
        extractor = train_resnet_34_fixed_20._get_embedding,
        excel   = "fixed_20_r34",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_SCOTT_R34.value,
        extractor = train_resnet_34_scott._get_embedding,
        excel   = "scott_r34",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_STURGES_R34.value,
        extractor = train_resnet_34_sturges._get_embedding,
        excel   = "sturges_r34",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_AUTO_R34.value,
        extractor = train_resnet_34_auto._get_embedding,
        excel   = "auto_r34",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_FD_R34.value,
        extractor = train_resnet_34_fd._get_embedding,
        excel   = "fd_r34",
    )
]

EXPERS_VIT_B_16 = [
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_FIXED_30_VIT_B_16.value,
        extractor = train_vit_b_16_fixed_30._get_embedding,
        excel   = "fixed_30_vit_b_16",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_FIXED_20_VIT_B_16.value,
        extractor = train_vit_b_16_fixed_20._get_embedding,
        excel   = "fixed_20_vit_b_16",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_SCOTT_VIT_B_16.value,
        extractor = train_vit_b_16_scott._get_embedding,
        excel   = "scott_vit_b_16",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_STURGES_VIT_B_16.value,
        extractor = train_vit_b_16_sturges._get_embedding,
        excel   = "sturges_vit_b_16",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_AUTO_VIT_B_16.value,
        extractor = train_vit_b_16_auto._get_embedding,
        excel   = "auto_vit_b_16",
    ),
    dict(
        proto_path   = MODELS_BACKBONES_DIR / ProtosFileNames.KDE_FD_VIT_B_16.value,
        extractor = train_vit_b_16_fd._get_embedding,
        excel   = "fd_vit_b_16",
    )
]"""