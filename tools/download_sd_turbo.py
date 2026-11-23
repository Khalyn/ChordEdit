"""Download the sd-turbo weights needed by ChordEdit.

The app expects a model root that contains:
    unet/ scheduler/ text_encoder/ tokenizer/ vae/

Usage (any Python 3.7+):
    python tools/download_sd_turbo.py                 # -> models/sd-turbo
    python tools/download_sd_turbo.py --dest D:/sd-turbo
    python tools/download_sd_turbo.py --endpoint https://hf-mirror.com   # mainland mirror

By default only the fp16 checkpoints are fetched (about 2.2 GB instead of
4.3 GB), which is what fits comfortably on a 6 GB laptop GPU. The fp16 files
are additionally hard-linked to the canonical file names
(`diffusion_pytorch_model.safetensors` / `model.safetensors`) so that
`from_pretrained()` finds them without needing `variant="fp16"`.
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

REPO = "stabilityai/sd-turbo"

FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "text_encoder/config.json",
    ("text_encoder/model.fp16.safetensors", "text_encoder/model.safetensors"),
    "unet/config.json",
    ("unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.safetensors"),
    "vae/config.json",
    ("vae/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.safetensors"),
]


def human(num):
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return "%.1f%s" % (num, unit)
        num /= 1024.0


def download(url, dest, retries=5):
    """Download url -> dest with HTTP range resume."""
    tmp = dest + ".part"
    for attempt in range(1, retries + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = {"User-Agent": "chordedit-setup/1.0"}
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=60)
            total = resp.headers.get("Content-Length")
            total = int(total) + have if total is not None else None
            mode = "ab" if have else "wb"
            if have and resp.status != 206:
                # server ignored the range request -> restart from scratch
                have, mode = 0, "wb"
            done = have
            last = time.time()
            with open(tmp, mode) as fh:
                while True:
                    chunk = resp.read(1024 * 512)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last > 2.0:
                        speed = (done - have) / max(0.001, now - last)
                        msg = "  %s / %s  @ %s/s" % (
                            human(done),
                            human(total) if total else "?",
                            human(speed),
                        )
                        sys.stdout.write("\r" + msg + "      ")
                        sys.stdout.flush()
                        last = now
            sys.stdout.write("\r" + " " * 70 + "\r")
            if total is not None and os.path.getsize(tmp) != total:
                raise IOError("incomplete download %d/%d" % (os.path.getsize(tmp), total))
            os.replace(tmp, dest)
            return
        except (urllib.error.URLError, IOError, OSError) as exc:
            wait = min(30, 2 * attempt)
            sys.stdout.write("\r  retry %d/%d in %ds (%s)\n" % (attempt, retries, wait, str(exc)[:120]))
            sys.stdout.flush()
            time.sleep(wait)
    raise SystemExit("download failed: %s" % url)


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)  # hard link: no extra disk space
    except OSError:
        import shutil

        shutil.copyfile(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sd-turbo"))
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--fp32", action="store_true", help="also download the full fp32 checkpoints")
    args = parser.parse_args()

    dest = os.path.abspath(args.dest)
    endpoint = args.endpoint.rstrip("/")
    print("repo    :", REPO)
    print("endpoint:", endpoint)
    print("dest    :", dest)
    os.makedirs(dest, exist_ok=True)

    items = list(FILES)
    if args.fp32:
        items += [
            "text_encoder/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ]

    links = []
    for item in items:
        remote = item if isinstance(item, str) else item[0]
        target = remote if isinstance(item, str) else item[1]
        out = os.path.join(dest, target.replace("/", os.sep))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        url = "%s/%s/resolve/main/%s" % (endpoint, REPO, remote)
        print("[%d/%d] %s" % (len(links) + 1, len(items), target))
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print("  already present, skipping")
        else:
            download(url, out)
        if not isinstance(item, str):
            links.append((out, os.path.join(dest, remote.replace("/", os.sep))))

    for src, dst in links:
        # remote == canonical name and already downloaded, nothing to do
        if os.path.normcase(src) != os.path.normcase(dst):
            link_or_copy(src, dst)
            print("linked:", os.path.relpath(dst, dest))

    print("\ndone. model root:", dest)


if __name__ == "__main__":
    main()
