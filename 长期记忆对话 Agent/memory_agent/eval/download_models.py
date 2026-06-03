import argparse
from pathlib import Path


DEFAULT_GENERATION_MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local models before running baselines.")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--generation_model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--hf_endpoint", default=None, help="Example: https://hf-mirror.com")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_embed", action="store_true")
    args = parser.parse_args()

    if args.hf_endpoint:
        import os

        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Please install huggingface_hub first.") from exc

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_generation:
        download_one(snapshot_download, args.generation_model, models_dir / local_name(args.generation_model))
    if not args.skip_embed:
        download_one(snapshot_download, args.embed_model, models_dir / local_name(args.embed_model))


def download_one(snapshot_download, repo_id: str, target_dir: Path) -> None:
    print(f"[download] {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
    )
    print(f"[done] {target_dir}")


def local_name(repo_id: str) -> str:
    return repo_id.split("/")[-1]


if __name__ == "__main__":
    main()
