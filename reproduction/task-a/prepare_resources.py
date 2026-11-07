"""Prepare pinned evaluation source files and model weights for this reproduction."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_verified(url, destination, expected_prefix):
    if destination.exists():
        if not digest(destination).startswith(expected_prefix):
            raise ValueError(f"Existing file has an unexpected checksum: {destination}")
        print("Verified", destination.relative_to(ROOT))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            request = Request(url, headers={"User-Agent": "ChordEdit-reproduction"})
            with urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if not digest(temporary).startswith(expected_prefix):
            raise ValueError(f"Downloaded file has an unexpected checksum: {url}")
        temporary.replace(destination)
        print("Downloaded and verified", destination.relative_to(ROOT))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_sources():
    entries = json.loads((ROOT / "evaluation-sources.json").read_text())
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unexpected dependency path: {relative}")
        download_verified(entry["url"], ROOT / relative, entry["sha256"])


def prepare_models():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "stabilityai/sd-turbo",
        revision="b261bac6fd2cf515557d5d0707481eafa0485ec2",
        local_dir=ROOT / "sd-turbo",
        allow_patterns=[
            "LICENSE.md", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
            "text_encoder/model.safetensors", "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors", "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ],
        token=False,
    )
    clip_revision = "32bd64288804d66eefd0ccbe215aa642df71cc41"
    cache = ROOT / "evaluation-cache/hf"
    snapshot_download(
        "openai/clip-vit-large-patch14",
        revision=clip_revision,
        cache_dir=cache,
        allow_patterns=[
            "config.json", "merges.txt", "model.safetensors", "preprocessor_config.json",
            "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
        ],
        token=False,
    )
    refs = cache / "models--openai--clip-vit-large-patch14/refs"
    refs.mkdir(parents=True, exist_ok=True)
    temporary_ref = refs / "main.tmp"
    temporary_ref.write_text(clip_revision)
    os.replace(temporary_ref, refs / "main")

    checkpoints = ROOT / "evaluation-cache/torch/hub/checkpoints"
    download_verified(
        "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
        checkpoints / "dino_vitbase8_pretrain.pth",
        "575f0efc4838938314afa09897a62ed2b87919928b5edcd133abb907328995eb",
    )
    download_verified(
        "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
        checkpoints / "squeezenet1_1-b8a52dc0.pth",
        "b8a52dc0",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["sources", "models"])
    args = parser.parse_args()
    if args.action == "sources":
        prepare_sources()
    else:
        prepare_models()
