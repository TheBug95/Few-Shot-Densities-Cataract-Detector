# inference_kde.py
from __future__ import annotations
import pickle, random, numpy as np
from pathlib import Path
from typing import List, Dict

import torch          # solo si tu extractor usa GPU
from pycocotools.coco import COCO
from PIL import Image
from tqdm.auto import tqdm


class FewShotKDEInferencer:
    """
    Clase para inferencia Few-Shot con prototipos KDE + histograma θ_min.
    Mantiene la lógica exacta del script original.
    """

    def __init__(
        self,
        proto_path: Path | str,
        mask_generator,
        feature_extractor,
        backbone_name: str,
        root: Path | str,
        splits: List[str],
        normal_cat_id: int = 2,
    ):
        self.proto_path = Path(proto_path)
        self.mask_generator = mask_generator
        self.feature_extractor = feature_extractor
        self.backbone_name = backbone_name
        self.root = Path(root)
        self.splits = splits
        self.normal_cat_id = normal_cat_id

        # ── cargar prototipos una sola vez ───────────────────────────────
        with open(self.proto_path, "rb") as fp:
            self.protos: Dict[int, dict] = pickle.load(fp)
        self.ks = sorted(self.protos.keys())

        print(f"✓ Cargados prototipos de '{self.proto_path.name}' "
              f"({len(self.ks)} valores de k)")

    # ---------------------------------------------------------------------
    def _predict_image(self, img) -> Dict[int, int]:
        """
        Devuelve un dict {k: 0/1} para una imagen PIL.
        0 = normal, 1 = catarata (según umbral θ_min).
        """
        masks = self.mask_generator.generate(np.asarray(img))
        preds = {}

        for k in self.ks:
            proto = self.protos[k]
            kdes = proto["kdes"]
            t_min = proto["theta_min"]

            img_pred = 0
            for m in masks:
                ys, xs = np.where(m["segmentation"])
                x0, x1 = xs.min(), xs.max()
                y0, y1 = ys.min(), ys.max()
                crop = img.crop((x0, y0, x1, y1))

                emb = self.feature_extractor(crop)

                # log-densidad total KDE
                logp = sum(
                    kde.score_samples([[emb[d]]])[0]
                    for d, kde in enumerate(kdes)
                )

                if t_min <= logp:
                    img_pred = 1
                    break

            preds[k] = img_pred

        return preds

    # ---------------------------------------------------------------------
    def infer(self, sample_n: int | None = None) -> Dict[int, List[int]]:
        """
        Recorre los splits, procesa sample_n imágenes (o todas) y
        devuelve un dict {k: [pred_0, pred_1, …]}.
        """
        print(f"\n=== Inferencia usando backbone {self.backbone_name} ===")
        preds_by_k = {k: [] for k in self.ks}

        for split in self.splits:
            coco = COCO(self.root / f"{split}/_annotations.coco.json")
            img_dir = self.root / split

            img_ids = coco.getImgIds()
            if sample_n is not None:
                img_ids = random.sample(img_ids, min(sample_n, len(img_ids)))

            for iid in tqdm(img_ids, desc=f"[{split}]"):
                info = coco.loadImgs(iid)[0]
                img = Image.open(img_dir / info["file_name"]).convert("RGB")

                img_preds = self._predict_image(img)
                for k, p in img_preds.items():
                    preds_by_k[k].append(p)

        return preds_by_k
