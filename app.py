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
st.write("영어 문장을 입력하여 관련 이미지를 검색합니다.")
st.caption("CLIP ViT-B/32 · CPU · Flickr8k 이미지 100장")


@st.cache_resource
def load_model():
    model, _ = clip.load(
        "ViT-B/32",
        device="cpu",
        download_root=str(ROOT / "models"),
    )
    model.eval()
    return model


try:
    with np.load(DATA / "features_100.npz") as cache:
        names = cache["image_names"].tolist()
        features = torch.from_numpy(
            cache["image_features"].copy()
        ).float()
        saved_hash = cache["subset_sha256"].item()

    current_hash = hashlib.sha256(
        (DATA / "subset_100.csv").read_bytes()
    ).hexdigest()

    if current_hash != saved_hash:
        st.error("데이터가 변경되었습니다. encode_images.py를 다시 실행하세요.")
        st.stop()

except (OSError, KeyError, ValueError) as error:
    st.error(f"데이터를 불러올 수 없습니다: {error}")
    st.stop()


with st.form("search_form"):
    query = st.text_input(
        "검색 문장 (English)",
        value="A goalie tries to block the puck in a hockey game.",
    )
    top_k = st.selectbox("결과 개수 (Top-k)", [1, 5, 10], index=1)
    submitted = st.form_submit_button("검색")


if submitted:
    query = query.strip()
    if not query:
        st.warning("검색 문장을 입력하세요.")
        st.stop()

    with st.spinner("검색 중..."):
        model = load_model()

        try:
            start = time.perf_counter()

            with torch.inference_mode():
                tokens = clip.tokenize([query])
                text = model.encode_text(tokens).float()
                text = text / text.norm(dim=-1, keepdim=True)
                similarities = (features @ text.T).squeeze(1)
                scores, indices = similarities.topk(
                    min(top_k, len(names))
                )

            elapsed_ms = (time.perf_counter() - start) * 1000

        except RuntimeError as error:
            st.error(f"검색에 실패했습니다: {error}")
            st.stop()

    st.subheader("검색 결과")
    st.write(f"검색 문장: {query}")
    st.caption(f"검색 처리 시간: {elapsed_ms:.2f} ms · 모델 로딩 제외")
    st.caption("유사도 점수는 정확도나 확률이 아닙니다.")

    columns = st.columns(3)

    for rank, (score, index) in enumerate(
        zip(scores.tolist(), indices.tolist()), start=1
    ):
        name = names[index]

        with columns[(rank - 1) % 3]:
            st.markdown(f"**{rank}위 · 유사도 {score:.4f}**")
            st.image(
                str(DATA / "Images" / name),
                use_container_width=True,
            )
            st.caption(name)