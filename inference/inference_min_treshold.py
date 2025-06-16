# %% Inferencia Few-Shot KDE + Histograma
import pickle, random, numpy as np, torch
from pathlib import Path
from tqdm.auto import tqdm
from pycocotools.coco import COCO
from PIL import Image

#SEED = 240915
#random.seed(SEED); np.random.seed(SEED)
#torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ─── rutas y constantes ──────────────────────────────────────────────
PROC_DIR      = Path("processed")
SPLITS        = ["valid"]
ROOT          = Path("/content/cataract-seg.v2-with-augmentation.coco-segmentation")
NORMAL_CAT_ID = 2

def infer_with_proto(proto_path, backbone_name, feature_extractor, sample_n = None):
    """
    Evalúa accuracy usando los prototipos con umbral θ_min / θ_max
    obtenidos vía histograma de log-scores KDE.
    """
    print(f"\n=== Inferencia usando backbone {backbone_name} ===")

    # 1) cargar prototipos ----------------------------------------------------
    with open(proto_path, "rb") as fp:
        protos = pickle.load(fp)
    ks = sorted(protos.keys())

    # 2) recorrer cada split --------------------------------------------------
    for split in SPLITS:
        coco    = COCO(str(ROOT/f"{split}/_annotations.coco.json"))
        img_dir = ROOT/split

        #print(len(y_true))
        preds_by_k = {k: [] for k in ks}

        # --- elegir sub-muestra sin repetición -----------------------
        img_ids = coco.getImgIds()
        if sample_n is not None:
            img_ids = random.sample(img_ids, min(sample_n, len(img_ids)))

        # 3) recorrer imágenes ------------------------------------------------
        for iid in tqdm(img_ids, desc=f"[{split}]"):
            info = coco.loadImgs(iid)[0]
            img  = Image.open(img_dir/info["file_name"]).convert("RGB")

            #masks = mobile_mask_generator.generate(np.asarray(img))
            #masks = hq_mask_generator.generate(np.asarray(img))
            masks = mask_generator.generate(np.asarray(img))

            # evaluar cada valor de k
            for k in ks:
                proto   = protos[k]
                kdes    = proto["kdes"]
                t_min   = proto["theta_min"]
                #t_max   = proto["theta_max"]

                img_pred = 0
                for m in masks:
                    ys, xs = np.where(m["segmentation"])
                    x0, x1 = xs.min(), xs.max()
                    y0, y1 = ys.min(), ys.max()
                    crop   = img.crop((x0, y0, x1, y1))

                    emb = feature_extractor(crop)

                    # log-densidad total KDE
                    logp = sum(
                        kde.score_samples([[emb[d]]])[0]
                        for d, kde in enumerate(kdes)
                    )

                    if t_min <= logp:
                        img_pred = 1
                        break

                preds_by_k[k].append(img_pred)

    return preds_by_k