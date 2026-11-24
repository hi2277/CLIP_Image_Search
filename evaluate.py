import csv
import hashlib
import json
import time
from pathlib import Path

import clip
import numpy as np
import torch


def main():
    root = Path(__file__).resolve().parent
    folder = root / "data" / "flickr8k"
    subset = folder / "subset_100.csv"

    with np.load(folder / "features_100.npz") as cache:
        names = cache["image_names"].tolist()
        features = torch.from_numpy(
            cache["image_features"].copy()
        ).float()
        saved_hash = cache["subset_sha256"].item()

    current_hash = hashlib.sha256(
        subset.read_bytes()
    ).hexdigest()
    if current_hash != saved_hash:
        raise ValueError("Subset changed. Re-run encode_images.py.")

    with subset.open(encoding="utf-8", newline="") as f:
        queries = list(csv.DictReader(f))

    if len(names) != 100 or len(queries) != 500:
        raise ValueError("Expected 100 images and 500 captions.")

    name_to_index = {name: i for i, name in enumerate(names)}
    if set(row["image"] for row in queries) != set(names):
        raise ValueError("Images and captions do not match.")

    print("Loading CLIP model...", flush=True)
    model, _ = clip.load(
        "ViT-B/32",
        device="cpu",
        download_root=str(root / "models"),
    )
    model.eval()

    hits = {1: 0, 5: 0, 10: 0}
    times = []
    details = []

    with torch.inference_mode():
        # Warm up before measuring query time.
        model.encode_text(clip.tokenize(["a photo of a dog"]))

        for i, row in enumerate(queries, start=1):
            start = time.perf_counter()

            tokens = clip.tokenize([row["caption"]])
            text = model.encode_text(tokens).float()
            text = text / text.norm(dim=-1, keepdim=True)
            scores = (features @ text.T).squeeze(1)
            order = torch.argsort(
                scores, descending=True, stable=True
            ).tolist()

            target = name_to_index[row["image"]]
            rank = order.index(target) + 1
            seconds = time.perf_counter() - start
            times.append(seconds)

            for k in hits:
                hits[k] += int(rank <= k)

            details.append({
                "query": row["caption"],
                "correct_image": row["image"],
                "correct_rank": rank,
                "top1_image": names[order[0]],
                "top1_similarity": float(scores[order[0]]),
                "query_seconds": seconds,
            })

            if i % 50 == 0:
                print(f"Evaluated {i}/{len(queries)}", flush=True)

    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    summary = {
        "model": "ViT-B/32",
        "device": "cpu",
        "candidate_images": len(names),
        "queries": len(queries),
        "subset_sha256": current_hash,
        "recall": {
            f"R@{k}": hits[k] / len(queries) for k in hits
        },
        "mean_query_seconds": float(np.mean(times)),
    }

    with (results_dir / "metrics_100.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (results_dir / "query_results_100.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)

    print("\nEvaluation completed!")
    for k in hits:
        print(
            f"Recall@{k}: {hits[k] / len(queries):.2%}"
            f" ({hits[k]}/{len(queries)})"
        )
    print(f"Mean query time: {np.mean(times) * 1000:.2f} ms")
    print("Results saved to:", results_dir)


if __name__ == "__main__":
    main()