import csv
import hashlib
import json
import time
from pathlib import Path

import clip
import numpy as np
import torch


def load_features(folder, size):
    subset = folder / f"subset_{size}.csv"

    with np.load(folder / f"features_{size}.npz") as cache:
        names = cache["image_names"].tolist()
        features = cache["image_features"].copy()
        saved_hash = cache["subset_sha256"].item()
        model_name = cache["model_name"].item()

    current_hash = hashlib.sha256(
        subset.read_bytes()
    ).hexdigest()

    if saved_hash != current_hash:
        raise ValueError(f"Subset {size} changed.")

    if model_name != "ViT-B/32":
        raise ValueError("Unexpected model.")

    if features.shape != (size, 512):
        raise ValueError(f"Unexpected feature shape for {size}.")

    if len(set(names)) != size or not np.isfinite(features).all():
        raise ValueError(f"Invalid features for {size}.")

    return names, torch.from_numpy(features).float()


def main():
    root = Path(__file__).resolve().parent
    folder = root / "data" / "flickr8k"
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    query_file = folder / "subset_100.csv"
    with query_file.open(encoding="utf-8-sig", newline="") as f:
        queries = list(csv.DictReader(f))

    if len(queries) != 500:
        raise ValueError("Expected 500 fixed queries.")

    datasets = {
        size: load_features(folder, size)
        for size in [100, 500, 1000]
    }

    names_100, features_100 = datasets[100]

    if set(row["image"] for row in queries) != set(names_100):
        raise ValueError("Queries and original images do not match.")

    for size in [500, 1000]:
        names, features = datasets[size]
        if names[:100] != names_100:
            raise ValueError("Original image order does not match.")
        if not torch.allclose(features[:100], features_100):
            raise ValueError("Original image features do not match.")

    if datasets[1000][0][:500] != datasets[500][0]:
        raise ValueError("500-image and 1000-image lists do not match.")

    print("Loading CLIP model...", flush=True)
    model, _ = clip.load(
        "ViT-B/32",
        device="cpu",
        download_root=str(root / "models"),
    )
    model.eval()

    summaries = []

    with torch.inference_mode():
        for size in [100, 500, 1000]:
            names, features = datasets[size]
            name_to_index = {
                name: index for index, name in enumerate(names)
            }
            hits = {1: 0, 5: 0, 10: 0}
            times = []
            details = []

            print(f"\nEvaluating {size} candidate images...", flush=True)

            # Warm up before timing this experiment.
            warm_text = model.encode_text(
                clip.tokenize(["a photo of a dog"])
            ).float()
            warm_text = warm_text / warm_text.norm(
                dim=-1, keepdim=True
            )
            torch.argsort(
                (features @ warm_text.T).squeeze(1),
                descending=True,
                stable=True,
            )

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

                if i % 100 == 0:
                    print(
                        f"{size} images: {i}/500 queries completed",
                        flush=True,
                    )

            summary = {
                "candidate_images": size,
                "queries": len(queries),
                "recall_at_1": hits[1] / len(queries),
                "recall_at_5": hits[5] / len(queries),
                "recall_at_10": hits[10] / len(queries),
                "mean_query_ms": float(np.mean(times) * 1000),
            }
            summaries.append(summary)

            output = results_dir / f"query_results_fixed500_{size}.csv"
            with output.open(
                "w", encoding="utf-8-sig", newline=""
            ) as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(details[0])
                )
                writer.writeheader()
                writer.writerows(details)

            print(f"Saved: {output.name}", flush=True)

    with (results_dir / "scale_comparison_fixed500.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summaries[0])
        )
        writer.writeheader()
        writer.writerows(summaries)

    report = {
        "model": "ViT-B/32",
        "device": "cpu",
        "query_source": "subset_100.csv",
        "query_sha256": hashlib.sha256(
            query_file.read_bytes()
        ).hexdigest(),
        "protocol": "Same 500 queries for all candidate sizes",
        "results": summaries,
    }
    with (results_dir / "scale_comparison_fixed500.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Comparison: same 500 queries ===")
    for row in summaries:
        print(f"\nCandidate images: {row['candidate_images']}")
        print(f"Recall@1:  {row['recall_at_1']:.2%}")
        print(f"Recall@5:  {row['recall_at_5']:.2%}")
        print(f"Recall@10: {row['recall_at_10']:.2%}")
        print(f"Mean query time: {row['mean_query_ms']:.2f} ms")

    print("\nResults saved to:", results_dir)
    print("Completed!")


if __name__ == "__main__":
    main()