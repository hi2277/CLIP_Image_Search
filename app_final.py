import hashlib
import time
from pathlib import Path

import clip
import numpy as np
import streamlit as st
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "flickr8k"

st.set_page_config(
    page_title="CLIP Image Search",
    layout="wide",
)

st.title("CLIP 이미지 검색")
st.write("영어 문장을 입력하면 관련 이미지를 찾아줍니다.")
st.caption("CLIP ViT-B/32 · CPU")


@st.cache_resource
def load_model():
    model, _ = clip.load(
        "ViT-B/32",
        device="cpu",
        download_root=str(ROOT / "models"),
    )
    model.eval()
    return model


@st.cache_data
def load_index(size, feature_modified, subset_modified):
    subset = DATA / f"subset_{size}.csv"

    with np.load(DATA / f"features_{size}.npz") as cache:
        names = cache["image_names"].tolist()
        features = cache["image_features"].copy()
        saved_hash = cache["subset_sha256"].item()
        model_name = cache["model_name"].item()

    current_hash = hashlib.sha256(
        subset.read_bytes()
    ).hexdigest()

    if saved_hash != current_hash:
        raise ValueError("이미지 목록과 저장된 특징이 일치하지 않습니다.")

    if model_name != "ViT-B/32":
        raise ValueError("모델 정보가 일치하지 않습니다.")

    if len(names) != size or features.shape != (size, 512):
        raise ValueError("이미지 수 또는 특징 크기가 올바르지 않습니다.")

    if not np.isfinite(features).all():
        raise ValueError("특징 데이터에 잘못된 값이 있습니다.")

    return names, features


with st.form("search"):
    size = st.selectbox(
        "검색할 이미지 수",
        [100, 500, 1000],
        index=0,
    )

    query = st.text_input(
        "검색 문장 (English)",
        value="two men jump for the basketball .",
    )

    top_k = st.selectbox(
        "결과 개수 (Top-k)",
        [1, 5, 10],
        index=1,
    )

    submitted = st.form_submit_button("검색")


if submitted:
    query = query.strip()

    if not query:
        st.warning("검색 문장을 입력하세요.")
        st.stop()

    try:
        feature_file = DATA / f"features_{size}.npz"
        subset_file = DATA / f"subset_{size}.csv"

        with st.spinner("이미지 데이터와 모델을 준비하고 있습니다..."):
            names, array = load_index(
                size,
                feature_file.stat().st_mtime_ns,
                subset_file.stat().st_mtime_ns,
            )
            features = torch.from_numpy(array).float()
            model = load_model()

    except (OSError, KeyError, ValueError, RuntimeError) as error:
        st.error(f"준비 중 오류가 발생했습니다: {error}")
        st.stop()

    with st.spinner("검색 중..."):
        try:
            start = time.perf_counter()

            with torch.inference_mode():
                tokens = clip.tokenize([query])
                text = model.encode_text(tokens).float()
                text = text / text.norm(dim=-1, keepdim=True)

                similarities = (features @ text.T).squeeze(1)
                indices = torch.argsort(
                    similarities,
                    descending=True,
                    stable=True,
                )[:top_k].tolist()

            elapsed_ms = (time.perf_counter() - start) * 1000

        except RuntimeError as error:
            st.error(
                "검색에 실패했습니다. 문장이 너무 길면 짧게 입력해 주세요."
            )
            st.caption(str(error))
            st.stop()

    st.subheader("검색 결과")
    st.write(f"검색 문장: {query}")

    col1, col2, col3 = st.columns(3)
    col1.metric("검색 이미지 수", f"{size}장")
    col2.metric("표시 결과", f"Top-{top_k}")
    col3.metric("검색 처리 시간", f"{elapsed_ms:.2f} ms")

    st.caption(
        "처리 시간에는 데이터 및 모델 로딩, 화면 표시 시간이 포함되지 않습니다."
    )
    st.caption("유사도 점수는 정확도나 확률이 아닙니다.")

    for offset in range(0, len(indices), 3):
        columns = st.columns(3)

        for column, index in zip(
            columns, indices[offset:offset + 3]
        ):
            rank = offset + list(
                indices[offset:offset + 3]
            ).index(index) + 1
            name = names[index]
            image_path = DATA / "Images" / name
            score = float(similarities[index])

            with column:
                st.markdown(
                    f"**{rank}위 · 유사도 {score:.4f}**"
                )

                if image_path.is_file():
                    st.image(
                        str(image_path),
                        use_container_width=True,
                    )
                else:
                    st.warning("이미지 파일을 찾을 수 없습니다.")

                st.caption(name)