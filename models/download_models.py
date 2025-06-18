from pathlib import Path
import subprocess, sys, re
from huggingface_hub import hf_hub_download


class SAMModelManager:
    """
    Gestor unificado para SAM, MobileSAM, HQ-SAM/Light-HQ-SAM y MedSAM(+2).
    """
    _MODELS = {
        # 1) Segment Anything original ---------------------------------------
        "sam": {
            "pip": "git+https://github.com/facebookresearch/segment-anything.git",
            "variants": {
                "vit_b": ("url", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
                "vit_l": ("url", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
                "vit_h": ("url", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
            },
        },

        # 2) MobileSAM --------------------------------------------------------
        #   *Peso oficial en HF: repo «dhkim2810/MobileSAM», archivo mobile_sam.pt*
        "mobilesam": {
            "pip": "git+https://github.com/ChaoningZhang/MobileSAM.git",
            "variants": {
                "vit_t": ("dhkim2810/MobileSAM", "mobile_sam.pt"),  # 40 MB:contentReference[oaicite:0]{index=0}
            },
        },

        # 3) HQ-SAM / Light-HQ-SAM -------------------------------------------
        "hq_sam": {
            "pip": "git+https://github.com/SysCV/sam-hq.git",
            "variants": {
                "vit_l": ("url ", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"),
                "vit_t": ("url", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_t.pth"),
                "vit_h": ("url", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"),
                "vit_b": ("url", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"),
                "hiera_l_2.1": ("url", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt"),
            },
        },

        # 4) MedSAM y MedSAM2 -------------------------------------------------
        "medsam": {
            "pip": "git+https://github.com/bowang-lab/MedSAM.git",
            "variants": {
                "medsam1": ("url", "https://zenodo.org/records/10689643/files/medsam_vit_b.pth"),
                "medsam2_latest": ("url", "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"),
            },
        },
    }

    # -----------------------------------------------------------------------
    def __init__(self, root_dir: str):
        self.root = Path(root_dir).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------- API pública ------------------------------
    def install_repo(self, family: str) -> None:
        pkg = self._MODELS[family]["pip"]
        if not self._pkg_installed(pkg):
            print(f"Instalando {family}…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        else:
            print(f"{family} ya estaba instalado.")

    def download_variant(self, family: str, variant: str, force: bool = False) -> Path:
        repo_or_url, filename = self._MODELS[family]["variants"][variant]
        dest = self.root / Path(filename).name

        if dest.exists() and not force:
            print(f"{dest.name} ya existe → se omite descarga.")
            return dest

        # --- descarga ------------------------------------------------------
        if repo_or_url == "url" or repo_or_url.startswith("http"):
            import requests
            url = filename if repo_or_url == "url" else repo_or_url
            print(f"Descargando desde URL directa:\n  {url}")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        else:
            print(f"Descargando desde Hugging Face ({repo_or_url})…")
            dest_path = hf_hub_download(
                repo_id=repo_or_url,
                filename=filename,
                cache_dir=str(self.root),
                force_download=force,
            )
            dest = Path(dest_path)

        return dest

    def setup(self, family: str, variant: str, install: bool = True, force: bool = False) -> Path:
        if install:
            self.install_repo(family)
        return self.download_variant(family, variant, force)

    def list_supported(self) -> None:
        for fam, info in self._MODELS.items():
            print(f"{fam}: {', '.join(info['variants'])}")

    # ---------------------- utilidades internas ---------------------------
    @staticmethod
    def _pkg_installed(pip_spec: str) -> bool:
        import pkg_resources
        # nombre heurístico del proyecto
        name = re.sub(r".*/|\.git$", "", pip_spec)
        try:
            pkg_resources.get_distribution(name)
            return True
        except pkg_resources.DistributionNotFound:
            return False



#USAGE EXAMPLE
"""mgr = SAMModelManager("Models")

# 1) Instalar y descargar el SAM base (vit_h)
sam_ckpt = mgr.setup("sam", "vit_h")

# 2) Light-HQ-SAM (vit_tiny)  → instala y descarga
path_hqsam_tiny = mgr.setup("hq_sam", "vit_tiny")

# 3) Instalar y descargar MobileSAM (tiny)
mobile_ckpt = mgr.setup("mobilesam", "vit_t")

# 4) MedSAM2 (latest)
medsam2_ckpt = mgr.setup("medsam", "medsam2_latest")

# 5) Listar todo lo soportado
mgr.list_supported()"""