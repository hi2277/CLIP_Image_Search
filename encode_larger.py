import csv
import hashlib
import time
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image


def read_names(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(dict.fromkeys(
            row["image"] for row in csv.DictReader(f)
        ))


def main():
    root = Path(__file__).resolve().parent
    folder = root / "data" / "flickr8k"
    subset_100 = folder / "subset_100.csv"

    names_100 = read_names(subset_100)
    names_500 = read_names(folder / "subset_500.csv")
    names_1000 = read_names(folder / "subset_1000.csv")

    if (
        len(names_100) != 100
        or len(names_500) != 500
        or len(names_1000) != 1000
        or names_1000[:500] != names_500
        or names_1000[:100] != names_100
    ):
        raise ValueError("Image lists do not match.")

    with np.load(folder / "features_100.npz") as cache:
        if cache["model_name"].item() != "ViT-B/32":
            raise ValueError("Unexpected model.")

        current_hash = hashlib.sha256(
            subset_100.read_bytes()
        ).hexdigest()
        if cache["subset_sha256"].item() != current_hash:
            raise ValueError("Original subset changed.")

        cached_names = cache["image_names"].tolist()
        cached_features = cache["image_features"].copy()

    if cached_names != names_100:
        raise ValueError("Original feature order does not match.")

    if cached_features.shape != (100, 512):
        raise ValueError("Unexpected original feature shape.")

    features = {
        name: feature
        for name, feature in zip(cached_names, cached_features)
    }

    print("Reusing 100 saved image features.", flush=True)
    print("Loading CLIP model...", flush=True)

    model, preprocess = clip.load(
        "ViT-B/32",
        device="cpu",
        download_root=str(root / "models"),
    )
    model.eval()

    new_names = names_1000[100:]
    start = time.perf_counter()

    with torch.inference_mode():
        for i, name in enumerate(new_names, start=1):
            with Image.open(folder / "Images" / name) as image:
                inputs = preprocess(
                    image.convert("RGB")
                ).unsqueeze(0)

            feature = model.encode_image(inputs).float()
            feature = feature / feature.norm(
                dim=-1, keepdim=True
            )
            features[name] = feature.cpu().numpy()[0]

            if i == 1 or i % 50 == 0:
                print(
                    f"New images processed: {i}/{len(new_names)}",
                    flush=True,
                )

    elapsed = time.perf_counter() - start

    for size, names in [(500, names_500), (1000, names_1000)]:
        matrix = np.stack([features[name] for name in names])

        if not np.isfinite(matrix).all():
            raise ValueError("Invalid feature values.")

        subset = folder / f"subset_{size}.csv"
        output = folder / f"features_{size}.npz"

        np.savez_compressed(
            output,
            image_names=np.array(names),
            image_features=matrix,
            model_name=np.array("ViT-B/32"),
            subset_sha256=np.array(
                hashlib.sha256(subset.read_bytes()).hexdigest()
            ),
        )

        print(f"\nSaved: {output}")
        print("Feature shape:", matrix.shape)

    print(f"\nNew 900 images extraction time: {elapsed:.2f} seconds")
    print("Completed!")


if __name__ == "__main__":
    main()