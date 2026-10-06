import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Cainiao-AI/LaDe-D"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "delivery"

# Paths copied verbatim from the Hugging Face dataset viewer snippet
SPLIT_FILES = {
    "cq": "data/delivery_cq-00000-of-00001-465887add76aeabc.parquet",
    "hz": "data/delivery_hz-00000-of-00001-8090c86f64781f71.parquet",
    "jl": "data/delivery_jl-00000-of-00001-a4fbefe3c368583c.parquet",
    "sh": "data/delivery_sh-00000-of-00001-ad9a4b1d79823540.parquet",
    "yt": "data/delivery_yt-00000-of-00001-cc85c1fcb1d10955.parquet",
}


def downloadSplit(cityCode: str, repoPath: str) -> Path:
    """Download one city split and copy it into data/raw/delivery."""
    targetPath = RAW_DIR / f"delivery_{cityCode}.parquet"
    if targetPath.exists():
        print(f"[skip] {targetPath.name} already exists")
        return targetPath

    # Streams to the HF cache first, so memory stays O(1)
    cachedPath = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=repoPath
    )
    shutil.copyfile(cachedPath, targetPath)
    print(f"[done] {targetPath.name} ({targetPath.stat().st_size / 1e6:.1f} MB)")
    return targetPath


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for cityCode, repoPath in SPLIT_FILES.items():
        downloadSplit(cityCode, repoPath)


if __name__ == "__main__":
    main()
