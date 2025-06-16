import random, pickle, pathlib, numpy as np
from sklearn.neighbors import KernelDensity
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
from utils.constants import (
    DATASETS_PROCESSED_CATARACT_DIR, 
    RANDOM_SEED, 
    INPUT_SIZE,
    MODELS_BACKBONES_DIR,
    NORMAL_CAT_ID,
    CATARACT_TRAIN_SPLIT,
    CATARACT_COCO_TRAIN_SPLIT,
    Backbone,
    BackbonesWeights,
    BandwidthMethod, 
    BinningMethod
)
from torchvision import models
import torch, torchvision.transforms as T
import warnings


class FewShotDensityTrainerKDELeaveOneOut:

    def __init__(self,
                 weights_models: BackbonesWeights = BackbonesWeights.RESNET18,
                 proc_dir: str = DATASETS_PROCESSED_CATARACT_DIR,
                 backbone: Backbone = Backbone.R18,
                 pad: int = 10,
                 device: str = None,
                 binning_strategy: BinningMethod = BinningMethod.SCOTT,
                 random_seed: int = RANDOM_SEED):

        # Configurar semilla para reproducibilidad
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

        # —–– COCO y paths —––––––––––––––––––––––––––––
        self.coco       = COCO(CATARACT_COCO_TRAIN_SPLIT)
        self.proc_dir   = pathlib.Path(proc_dir)
        self.proc_dir.mkdir(exist_ok=True, parents=True)

        # —–– Parámetros —––––––––––––––––––––––––––––––
        self.pad        = pad
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone   = backbone.lower()
        self.weights_models = weights_models
        self.binning_strategy = binning_strategy
        self.random_seed = random_seed
        
        # —–– Transformación para ResNet —––––––––––––––
        self.tf = T.Compose([
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        # —–– Backbone sin FC final —–––––––––––––––––––
        if self.backbone == Backbone.R18:
            net = models.resnet18(weights=self.weights_models)
            self.model = torch.nn.Sequential(*list(net.children())[:-1])

        elif self.backbone == Backbone.R34:
            net = models.resnet34(weights=self.weights_models)
            self.model = torch.nn.Sequential(*list(net.children())[:-1])

        elif self.backbone == Backbone.VIT_B16:
            # ViT-B/16: quitamos la cabeza de clasificación
            net = models.vit_b_16(weights=self.weights_models)
            net.heads = torch.nn.Identity()
            self.model = net

        else:
            raise ValueError(f"Backbone no reconocido: {self.backbone!r}")

        self.model = self.model.to(self.device).eval()

        # Dimensión de embedding (se infiere con un tensor dummy)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE).to(self.device)
            feat  = self.model(dummy)
            self.feat_dim = int(feat.reshape(1, -1).shape[1])

        # —–– Pre-cache IDs positivos y normales —–––––––
        all_ids = self.coco.getImgIds()
        self.pos_ids  = [iid for iid in all_ids
                         if any(a["category_id"]!=NORMAL_CAT_ID
                                for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[iid])))]

        print(f"Inicializado con {len(self.pos_ids)} imágenes positivas")
        print(f"Estrategia de binning: {self.binning_strategy}")
        print(f"Semilla aleatoria: {self.random_seed}")

    def _get_embedding(self, crop: Image.Image) -> np.ndarray:
        """Extrae embedding 512-D de un crop PIL."""
        try:
            x = self.tf(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                f = self.model(x)
            return f.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Error extrayendo embedding: {e}")
            return np.zeros(self.feat_dim)

    def _scott_bw(self, n: int) -> float:
        """Scott's rule univariante: n^(-1/(d+4)), d=1."""
        return n**(-1/(1+4))

    def _load_and_crop(self, iid: int, positive: bool) -> Image.Image:
        """
        Si `positive`: recorta por la primera segmentación de catarata;
        si `negative`: devuelve la imagen entera (ya redimensionada).
        """
        try:
            info = self.coco.loadImgs(iid)[0]
            img_path = CATARACT_TRAIN_SPLIT / info["file_name"]

            if not img_path.exists():
                raise FileNotFoundError(f"Imagen no encontrada: {img_path}")

            img = Image.open(img_path).convert("RGB")
            H, W = info["height"], info["width"]

            if positive:
                ann = next(a for a in self.coco.loadAnns(
                            self.coco.getAnnIds(imgIds=[iid]))
                        if a["category_id"] != NORMAL_CAT_ID)
                rle = maskUtils.frPyObjects(ann["segmentation"], H, W)
                rle = maskUtils.merge(rle)
                m = maskUtils.decode(rle).astype(bool)
                ys, xs = np.where(m)

                if len(xs) == 0 or len(ys) == 0:
                    # Fallback si no hay píxeles en la máscara
                    crop = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
                else:
                    x0, x1 = max(xs.min()-self.pad, 0), min(xs.max()+self.pad, W)
                    y0, y1 = max(ys.min()-self.pad, 0), min(ys.max()+self.pad, H)
                    crop = img.crop((x0, y0, x1, y1))
            else:
                crop = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

            return crop

        except Exception as e:
            print(f"Error cargando imagen {iid}: {e}")
            # Fallback: imagen negra
            return Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), (0, 0, 0))

    # ==================== MÉTODOS DE BINNING ====================
    def _theta_from_hist(self, d: np.ndarray, method) -> float:
        """Método 2: Métodos automáticos de numpy"""
        if len(d) == 0:
            return 0.0

        try:
            hist, edges = np.histogram(d, bins=method)
            nz = np.nonzero(hist)[0]

            if len(nz) == 0:
                return float(np.min(d)), float(np.max(d))

            return float(edges[nz[0]]), float(edges[nz[-1] + 1])
        except Exception as e:
            warnings.warn(f"Error con método {method}: {e}. Usando fallback.")
            return float(np.min(d)), float(np.max(d))


    def _get_theta(self, d: np.ndarray):
        """Dispatcher para diferentes estrategias de binning"""
        if self.binning_strategy == BinningMethod.FIXED_20:
            return self._theta_from_hist(d, 20)
        elif self.binning_strategy == BinningMethod.FIXED_30:
            return self._theta_from_hist(d, 30)
        elif self.binning_strategy == BinningMethod.AUTO:
            return self._theta_from_hist(d, "auto")
        elif self.binning_strategy == BinningMethod.SCOTT:
            return self._theta_from_hist(d, "scott")
        elif self.binning_strategy == BinningMethod.FD:
            return self._theta_from_hist(d, "fd")
        elif self.binning_strategy == BinningMethod.STURGES:
            return self._theta_from_hist(d, "sturges")
        else:
            raise ValueError(f"Estrategia no reconocida: {self.binning_strategy}")

    # ------------------------------- TRAIN ------------------------------
    def train(self, ks=None):
        """Entrena el modelo con la estrategia de binning especificada"""
        ks = ks or list(range(3, 43, 3))
        prototypes = {}

        print(f"\n🚀 Iniciando entrenamiento con estrategia: {self.binning_strategy}")
        print(f"Valores de k a probar: {ks}")

        for i, k in enumerate(ks):
            print(f"\n--- Entrenando con k={k} ({i+1}/{len(ks)}) ---")

            # Validar que hay suficientes muestras
            if len(self.pos_ids) < k:
                print(f"⚠️  Solo hay {len(self.pos_ids)} muestras positivas, pero se necesitan {k}")
                continue

            try:
                # 1) Sample support
                s_pos = random.sample(self.pos_ids, k)
                print(f"Seleccionadas {len(s_pos)} muestras positivas")

                # 2) Extraer embeddings
                print("Extrayendo embeddings...")
                S_pos = []
                for j, img_id in enumerate(s_pos):
                    crop = self._load_and_crop(img_id, True)
                    emb = self._get_embedding(crop)
                    S_pos.append(emb)
                    if (j + 1) % 5 == 0:
                        print(f"  Procesadas {j+1}/{k} imágenes")

                S_pos = np.stack(S_pos)
                print(f"Shape de embeddings: {S_pos.shape}")

                # 3) Leave-One-Out Cross Validation
                print("Realizando Leave-One-Out CV...")
                logp_loo = []

                for idx in range(k):
                    # Separar train/test
                    S_train = np.delete(S_pos, idx, axis=0)  # k-1
                    S_test = S_pos[idx:idx+1]                # 1

                    # Entrenar KDEs
                    kdes_idx = [
                        KernelDensity(kernel="gaussian", bandwidth=BandwidthMethod.SCOTT).fit(S_train[:, d:d+1])
                        for d in range(self.feat_dim)
                    ]

                    # Calcular log-probabilidad
                    logp = np.sum([
                        kde.score_samples(S_test[:, d:d+1])
                        for d, kde in enumerate(kdes_idx)
                    ], axis=0)[0]

                    logp_loo.append(logp)

                logp_loo = np.array(logp_loo)
                print(f"Log-probabilidades LOO: min={logp_loo.min():.3f}, max={logp_loo.max():.3f}, mean={logp_loo.mean():.3f}")

                # 4) Calcular threshold con la estrategia especificada
                theta_min, theta_max = self._get_theta(logp_loo)
                print(f"Threshold calculado: {theta_min:.6f}")

                # 5) KDE definitivo con todas las muestras
                print("Entrenando KDE definitivo...")
                kdes = [
                    KernelDensity(kernel="gaussian", bandwidth=BandwidthMethod.SCOTT).fit(S_pos[:, d:d+1])
                    for d in range(self.feat_dim)
                ]

                prototypes[k] = {
                    "kdes": kdes,
                    "theta_min": theta_min,
                    "theta_max": theta_max,
                    "logp_loo_stats": {
                        "min": float(logp_loo.min()),
                        "max": float(logp_loo.max()),
                        "mean": float(logp_loo.mean()),
                        "std": float(logp_loo.std())
                    },
                    "sample_ids": s_pos  # Para reproducibilidad
                }

                print(f"✅ Completado k={k}")

            except Exception as e:
                print(f"❌ Error con k={k}: {e}")
                continue

        # Guardar prototipos
        suffix = f"{self.binning_strategy}_{self.backbone}_seed{self.random_seed}"
        out = MODELS_BACKBONES_DIR / f"kde_univar_protos_{suffix}.pkl"

        with open(out, "wb") as fp:
            pickle.dump({
                "prototypes": prototypes,
                "config": {
                    "binning_strategy": self.binning_strategy,
                    "backbone": self.backbone,
                    "random_seed": self.random_seed,
                    "bw_method": BandwidthMethod.SCOTT,
                    "input_size": INPUT_SIZE,
                    "pad": self.pad
                }
            }, fp)

        print(f"\n✔ Guardados prototipos en {out}")
        print(f"✔ Entrenamiento completado para {len(prototypes)} valores de k")

        return prototypes