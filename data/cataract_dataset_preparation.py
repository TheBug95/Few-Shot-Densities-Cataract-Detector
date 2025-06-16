from dataclasses import dataclass, field
from typing import List, Optional
from tqdm.auto import tqdm
import glob, pathlib, json

@dataclass
class CataractDatasetPrep:
    root: str
    out_dir: str = "processed"
    cat_ids: Optional[List[int]] = None

    def _process_img(self, src: str, dst: str):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        cv2.imwrite(dst, img)

    def _img_out_dir(self, split: str) -> pathlib.Path:
        d = pathlib.Path(self.out_dir)/"images"/split
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _find_json(self, split_dir: pathlib.Path) -> str:
        js = glob.glob(str(split_dir/"*.json"))
        if not js:
            raise FileNotFoundError(f"No JSON en {split_dir}")
        return js[0]

    def run(self):
        index = []
        for split in ("train","valid","test"):
            split_dir = pathlib.Path(self.root)/split
            ann_file  = self._find_json(split_dir)
            coco      = COCO(ann_file)
            img_out   = self._img_out_dir(split)

            for info in tqdm(coco.loadImgs(coco.getImgIds()), desc=f"[{split}]"):
                if self.cat_ids:
                    ann_ids = coco.getAnnIds(imgIds=[info["id"]],
                                              catIds=self.cat_ids, iscrowd=False)
                    if not ann_ids:
                        continue
                src = split_dir/info["file_name"]
                dst = img_out/info["file_name"]

                img = cv2.imread(str(src))
                cv2.imwrite(str(dst), img)
                index.append({
                    "id": info["id"],
                    "file_name": f"{split}/{info['file_name']}",
                    "split": split
                })

        pathlib.Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{self.out_dir}/index.json","w") as fp:
            json.dump({"images": index}, fp, indent=2)
        print(f"✔ {len(index)} imágenes listadas en «{self.out_dir}/images»")
