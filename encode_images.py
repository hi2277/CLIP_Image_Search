import csv
import hashlib
import time
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image


def main():
    root = Path(__file__).resolve().parent
    folder = root / "data" / "flickr8k"
    subset = folder / "subset_100.csv"

    with subset.open(encoding="utf-8", newline="") as f:
        names = list(dict.fromkeys(
            row["image"] for row in csv.DictReader(f)
        ))

    if len(names) != 100:
        raise ValueError(f"Expected 100 images, got {len(names)}")

    print("Loading CLIP model...", flush=True)
    model, preprocess = clip.load(
        "ViT-B/32",
        device="cpu",
        download_root=str(root / "models"),
    )
    model.eval()

    features = []
    start = time.perf_counter()

    with torch.inference_mode():
        for i, name in enumerate(names, start=1):
            with Image.open(folder / "Images" / name) as image:
                inputs = preprocess(
                    image.convert("RGB")
                ).unsqueeze(0)

            feature = model.encode_image(inputs).float()
            feature = feature / feature.norm(
                dim=-1, keepdim=True
            )
            features.append(feature.cpu().numpy())

            if i == 1 or i % 10 == 0:
                print(
                    f"Processed {i}/{len(names)} images",
                    flush=True,
                )

    elapsed = time.perf_counter() - start
    features = np.concatenate(features, axis=0)

    if not np.isfinite(features).all():
        raise ValueError("Invalid feature values.")

    output = folder / "features_100.npz"
    np.savez_compressed(
        output,
        image_names=np.array(names),
        image_features=features,
        model_name=np.array("ViT-B/32"),
        subset_sha256=np.array(
            hashlib.sha256(subset.read_bytes()).hexdigest()
        ),
        extraction_seconds=np.array(elapsed),
    )

    print("\nFeature extraction completed!")
    print("Feature shape:", features.shape)
    print(f"Extraction time: {elapsed:.2f} seconds")
    print("Saved to:", output)


if __name__ == "__main__":
    main()