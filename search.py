from pathlib import Path

import clip
import torch
from PIL import Image


def main():
    project_dir = Path(__file__).resolve().parent
    image_dir = project_dir / "data" / "images"
    model_dir = project_dir / "models"

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if not image_paths:
        print("No images found.")
        return

    print("Loading CLIP model...")
    model, preprocess = clip.load(
        "ViT-B/32",
        device="cpu",
        download_root=str(model_dir)
    )
    model.eval()

    features = []
    valid_paths = []

    print("Processing images...")

    with torch.no_grad():
        for path in image_paths:
            try:
                with Image.open(path) as image:
                    image_input = preprocess(
                        image.convert("RGB")
                    ).unsqueeze(0)
            except OSError as error:
                print(f"Cannot open {path.name}: {error}")
                continue

            feature = model.encode_image(image_input)
            feature = feature / feature.norm(dim=-1, keepdim=True)

            features.append(feature)
            valid_paths.append(path)

        if not features:
            print("No readable images found.")
            return

        image_features = torch.cat(features, dim=0)

    print(f"Ready! {len(valid_paths)} images loaded.")

    while True:
        query = input("\nEnter English text (or q to quit): ").strip()

        if query.lower() == "q":
            break

        if not query:
            continue

        with torch.no_grad():
            text_input = clip.tokenize([query], truncate=True)
            text_feature = model.encode_text(text_input)
            text_feature = text_feature / text_feature.norm(
                dim=-1, keepdim=True
            )

            similarities = (
                image_features @ text_feature.T
            ).squeeze(1)

            scores, indices = similarities.topk(
                min(5, len(valid_paths))
            )

        print("\nSearch results:")

        for rank, (score, index) in enumerate(
            zip(scores.tolist(), indices.tolist()), start=1
        ):
            print(
                f"{rank}. {valid_paths[index].name}"
                f" | similarity: {score:.4f}"
            )


if __name__ == "__main__":
    main()