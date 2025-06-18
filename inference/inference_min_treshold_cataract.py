# inference_kde.py
from __future__ import annotations
import pickle, random, numpy as np
from pathlib import Path
from typing import List, Dict, Any
from pycocotools.coco import COCO
from PIL import Image
from tqdm.auto import tqdm


class FewShotKDEInferencer:
    """
    Inferencia Few-Shot con prototipos KDE + histograma θ.
    Compatible con los .pkl guardados por el trainer antiguo y el nuevo.
    """

    # ----------------------------------------------------------
    def __init__(
        self,
        proto_path: str | Path,
        mask_generator,
        feature_extractor,
        backbone_name: str,
        root: str | Path,
        splits: List[str],
        normal_cat_id: int = 2,
    ):
        self.proto_path      = Path(proto_path)
        self.mask_generator  = mask_generator
        self.feature_extr    = feature_extractor
        self.backbone_name   = backbone_name
        self.root            = Path(root)
        self.splits          = splits
        self.normal_cat_id   = normal_cat_id

        self._load_prototypes()          # ← detecta formato

        print(f"✓ Cargados {len(self.ks)} valores de k "
              f"desde '{self.proto_path.name}'")

    # ----------------------------------------------------------
    def _load_prototypes(self) -> None:
        """Detecta formato (antiguo vs. nuevo) y carga prototipos."""
        with open(self.proto_path, "rb") as fp:
            data: Any = pickle.load(fp)

        if isinstance(data, dict) and "prototypes" in data:
            # ⇢ Formato NUEVO
            self.protos: Dict[int, dict] = data["prototypes"]
            self.train_config: Dict[str, Any] = data.get("config", {})
        else:
            # ⇢ Formato ANTIGUO
            self.protos = data
            self.train_config = {}

        self.ks = sorted(self.protos.keys())

    # ----------------------------------------------------------
    def _predict_image(self, img) -> Dict[int, int]:
        """Devuelve {k: 0/1} usando θ_min (y θ_max si está guardado)."""
        masks = self.mask_generator.generate(np.asarray(img))
        preds = {}

        for k in self.ks:
            proto   = self.protos[k]
            kdes    = proto["kdes"]
            t_min   = proto.get("theta_min", -np.inf)
            t_max   = proto.get("theta_max",  np.inf)   # por si lo necesitas

            img_pred = 0
            for m in masks:
                ys, xs = np.where(m["segmentation"])
                x0, x1 = xs.min(), xs.max()
                y0, y1 = ys.min(), ys.max()
                crop   = img.crop((x0, y0, x1, y1))

                emb = self.feature_extr(crop)
                logp = sum(
                    kde.score_samples([[emb[d]]])[0]
                    for d, kde in enumerate(kdes)
                )

                if t_min <= logp <= t_max:
                    img_pred = 1
                    break

            preds[k] = img_pred
        return preds

    # ----------------------------------------------------------
    def infer(self, sample_n: int | None = None) -> Dict[int, List[int]]:
        """Procesa `sample_n` imágenes (o todas) y devuelve {k: [preds]}."""
        print(f"\n=== Inferencia usando {self.backbone_name} ===")
        preds_by_k = {k: [] for k in self.ks}

        for split in self.splits:
            coco    = COCO(self.root / f"{split}/_annotations.coco.json")
            img_dir = self.root / split

            img_ids = coco.getImgIds()
            if sample_n is not None:
                img_ids = random.sample(img_ids, min(sample_n, len(img_ids)))

            for iid in tqdm(img_ids, desc=f"[{split}]"):
                info = coco.loadImgs(iid)[0]
                img  = Image.open(img_dir / info["file_name"]).convert("RGB")
                img_preds = self._predict_image(img)
                for k, p in img_preds.items():
                    preds_by_k[k].append(p)

        return preds_by_k
