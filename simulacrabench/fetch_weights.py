"""Download the two pinned checkpoints next to main.py and verify their sha256.

Standard library only. Usage: python fetch_weights.py
"""
import hashlib
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

WEIGHTS = [
    ("Prior-Labs/tabpfn_3", "24a16a89d245878b846555110985634aa2e656d7",
     "tabpfn-v3-classifier-v3_default.ckpt",
     "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"),
    ("jingang/TabICL", "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
     "tabicl-classifier-v2-20260212.ckpt",
     "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    for repo, rev, name, digest in WEIGHTS:
        dst = os.path.join(HERE, name)
        if not (os.path.exists(dst) and sha256(dst) == digest):
            url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
            print("downloading", url)
            urllib.request.urlretrieve(url, dst + ".part")
            os.replace(dst + ".part", dst)
        got = sha256(dst)
        if got != digest:
            raise SystemExit(f"sha256 mismatch for {name}: {got}")
        print("ok", name, got)


if __name__ == "__main__":
    main()
