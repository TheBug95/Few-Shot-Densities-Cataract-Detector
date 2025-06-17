import random, pickle, pathlib, numpy as np
from sklearn.neighbors import KernelDensity
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
import torch, torchvision.transforms as T
from torchvision import models

class FewShotDensityTrainerKDELeaveOneOut:

    def __init__(self,
                 coco_ann_train: str,
                 images_dir_train: str,
                 weights_models: str,
                 proc_dir: str = "processed",
                 backbone: str = "r18",
                 input_size: int = 224,
                 pad: int = 10,
                 bw_method: str = "scott",
                 normal_cat_id: int = 2,
                 device: str = None):
        # —–– COCO y paths —––––––––––––––––––––––––––––
        self.coco       = COCO(coco_ann_train)
        self.images_dir = pathlib.Path(images_dir_train)
        self.proc_dir   = pathlib.Path(proc_dir)
        self.proc_dir.mkdir(exist_ok=True, parents=True)

        # —–– Parámetros —––––––––––––––––––––––––––––––
        self.pad        = pad
        self.input_size = input_size
        self.bw_method  = bw_method
        self.normal_id  = normal_cat_id
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone   = backbone.lower()
        self.weights_models = weights_models

        # —–– Transformación para ResNet —––––––––––––––
        self.tf = T.Compose([
            T.Resize((input_size,input_size)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        # —–– Backbone sin FC final —–––––––––––––––––––
        if self.backbone == "r18":
            net = models.resnet18(weights=self.weights_models)
            self.model = torch.nn.Sequential(*list(net.children())[:-1])

        elif self.backbone == "r34":
            net = models.resnet34(weights=self.weights_models)
            self.model = torch.nn.Sequential(*list(net.children())[:-1])

        elif self.backbone == "vit":
            # ViT-B/16: quitamos la cabeza de clasificación
            net = models.vit_b_16(weights=self.weights_models)
            net.heads = torch.nn.Identity()
            self.model = net

        else:
            raise ValueError(f"Backbone no reconocido: {self.backbone!r}")

        self.model = self.model.to(self.device).eval()

        # Dimensión de embedding (se infiere con un tensor dummy)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size).to(self.device)
            feat  = self.model(dummy)
            self.feat_dim = int(feat.reshape(1, -1).shape[1])

        # —–– Pre-cache IDs positivos y normales —–––––––
        all_ids = self.coco.getImgIds()
        self.pos_ids  = [iid for iid in all_ids
                         if any(a["category_id"]!=self.normal_id
                                for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[iid])))]

    def _get_embedding(self, crop: Image.Image) -> np.ndarray:
        """Extrae embedding 512-D de un crop PIL."""
        x = self.tf(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.model(x)        # (1,512,1,1)
        return f.squeeze().cpu().numpy()

    def _scott_bw(self, n: int) -> float:
        """Scott’s rule univariante: n^(-1/(d+4)), d=1."""
        return n**(-1/(1+4))

    def _load_and_crop(self, iid:int, positive:bool) -> Image.Image:
        """
        Si `positive`: recorta por la primera segmentación de catarata;
        si `negative`: devuelve la imagen entera (ya redimensionada).
        """
        info = self.coco.loadImgs(iid)[0]
        img  = Image.open(self.images_dir/info["file_name"]).convert("RGB")
        H,W  = info["height"], info["width"]

        if positive:
            ann = next(a for a in self.coco.loadAnns(
                        self.coco.getAnnIds(imgIds=[iid]))
                    if a["category_id"]!=self.normal_id)
            rle  = maskUtils.frPyObjects(ann["segmentation"], H,W)
            rle  = maskUtils.merge(rle)
            m    = maskUtils.decode(rle).astype(bool)
            ys,xs= np.where(m)
            x0,x1 = max(xs.min()-self.pad,0), min(xs.max()+self.pad,W)
            y0,y1 = max(ys.min()-self.pad,0), min(ys.max()+self.pad,H)
            crop = img.crop((x0,y0,x1,y1))
        else:
            crop = img.resize((self.input_size,self.input_size), Image.BILINEAR)

        return crop

    @staticmethod
    def _theta_from_hist(d: np.ndarray, bins="auto") -> tuple[float,float]:
        """
        Histograma de d → (θ_min, θ_max) = bordes exteriores de los bins
        con conteo>0.
        """
        hist, edges = np.histogram(d, bins=bins)
        nz          = np.nonzero(hist)[0]
        return float(edges[nz[0]]), float(edges[nz[-1] + 1])

    # ------------------------------- TRAIN ------------------------------
    def train(self, ks=None):
        ks = ks or list(range(3, 43, 3))
        prototypes = {}

        for k in ks:
            # 1) sample support ------------------------------------------------
            s_pos = random.sample(self.pos_ids, k)

            # 2) embeddings ----------------------------------------------------
            S_pos = np.stack([self._get_embedding(self._load_and_crop(i, True))
                              for i in s_pos])

            # 3) KDE idéntico
            logp_loo = []
            bw   = self._scott_bw(k-1) if self.bw_method=="scott" else float(self.bw_method)
            kdes = []
            for idx in range(k):
                # a) separa conjunto de entrenamiento y test
                S_train = np.delete(S_pos, idx, axis=0)          # k-1
                S_test  = S_pos[idx:idx+1]                       # 1

                # b) KDE con k-1 muestras
                kdes_idx = [
                    KernelDensity(
                        kernel="gaussian",
                        bandwidth=bw
                    ).fit(
                        S_train[:, d:d+1]
                    )
                    for d in range(self.feat_dim)
                ]

                # c) log-p del ejemplo dejado fuera
                logp = np.sum([kde.score_samples(S_test[:, d:d+1])
                            for d, kde in enumerate(kdes_idx)], axis=0)[0]
                logp_loo.append(logp)                # shape: (k,)

            # 5)
            theta_min, theta_max = self._theta_from_hist(logp_loo, bins=30)

            # 6) KDE DEFINITIVO - Durante inferencia se quiere la mejor estimación de densidad posible
            # si añadimos el ejemplo que antes estuvo omitido se puede enriquece mas el modelo.
            kdes = [
                KernelDensity(
                    kernel="gaussian", bandwidth=bw
                ).fit(S_pos[:, d:d+1])
                for d in range(self.feat_dim)
            ]

            prototypes[k] = {
                "kdes":      kdes,
                "bw":        bw,
                "theta_min": theta_min,
                "theta_max": theta_max
            }

        # 8) guardar -----------------------------------------------------------
        out = self.proc_dir / f"kde_univar_protos_{self.backbone}.pkl"
        with open(out, "wb") as fp:
            pickle.dump(prototypes, fp)

        print(f"\n✔ Guardado prototipos en {out}")
        return prototypes