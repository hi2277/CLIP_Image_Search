# CLIP을 활용한 이미지-텍스트 유사도 검색 시스템

ZHOU ZICHENG · 2022106059 · 지도교수: 최진우

영어 문장을 입력하면 CLIP ViT-B/32로 관련 이미지를 검색하는 졸업프로젝트입니다.
Intel Mac의 CPU 환경에서 실행했습니다. 이미지를 미리 특징으로 저장한 뒤 검색에 사용합니다.
웹에서 후보 이미지 수 100 / 500 / 1000과 Top-k 1 / 5 / 10을 선택할 수 있습니다.

## 현재 프로젝트 폴더에서 웹 실행

기존 Mac에 데이터와 가상환경이 있으면 다시 설치하거나 특징을 다시 만들 필요가 없습니다.

```bash
cd ~/Desktop/clip_image_search
./.venv311/bin/python -m streamlit run app_final.py --server.address 127.0.0.1 --server.fileWatcherType none
```

브라우저 주소는 http://localhost:8501 입니다. 종료할 때는 실행 중인 터미널에서 Control + C를 누릅니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| app_final.py | 100 / 500 / 1000장 검색 웹 화면 |
| app.py | 초기 100장 웹 화면 |
| search.py | 고양이·강아지·자동차 3장 터미널 테스트 |
| prepare_subsets.py | 원본 이미지·설명에서 고정 부분집합 목록 생성 |
| encode_images.py | 100장 특징 생성 |
| encode_larger.py | 기존 100장 특징을 재사용하고 900장 추가 처리 |
| evaluate.py | 초기 100장 평가 |
| evaluate_sizes.py | 동일한 500개 질의로 세 후보 규모 평가 |
| validate_results.py | 코드 문법, 목록, 저장 결과의 일치 여부 검사 |
| requirements.txt | 확인된 주요 패키지 버전과 CLIP 설치 경로 |
| environment_versions.txt | 사용자 Mac에서 수집한 설치 버전 |

## 새 환경에서 준비

Python 3.11과 Git이 필요합니다. 확인된 사용자 환경은 macOS 14.4 / Intel / Python 3.11.4입니다.
다른 운영체제나 아키텍처에서 동일한 설치가 되는지는 별도로 확인하지 않았습니다.

```bash
python3.11 -m venv .venv311
./.venv311/bin/python -m pip install -r requirements.txt
```

CLIP은 https://github.com/openai/CLIP 의 구현을 사용합니다.
기존 환경의 배포 버전은 1.0이며 설치 당시 Git 커밋은 수집되지 않았습니다.
requirements.txt의 Git 설치 항목은 설치 시점의 저장소 내용을 가져오므로 커밋까지 고정된 환경 명세는 아닙니다.

## 데이터 준비

사용한 배포본: https://www.kaggle.com/datasets/adityajn105/flickr8k

다운로드 후 아래 위치에 둡니다.

- `data/flickr8k/Images/`: 원본 JPG 이미지
- `data/flickr8k/captions.txt`: 첫 행이 `image,caption`인 설명 파일

코드 패키지에는 원본 이미지, 모델 가중치, 가상환경 및 NPZ 특징 파일이 들어 있지 않습니다.
기존 컴퓨터에서는 이미 해당 파일이 있으므로 그대로 사용하면 됩니다.
새 환경에서는 위 데이터를 준비한 뒤 다음 순서로 실행합니다.

```bash
./.venv311/bin/python prepare_subsets.py
./.venv311/bin/python encode_images.py
./.venv311/bin/python encode_larger.py
```

이미지마다 설명이 정확히 5개 있고 이미지 파일도 존재하는 항목을 사용합니다.
정렬한 목록에서 `random.Random(42)`로 100장을 고릅니다.
나머지 목록에서 별도의 `random.Random(42)`로 900장을 골라 뒤에 붙입니다.
500장 목록은 이 순서의 앞 500장, 1000장 목록은 전체입니다.
고정 CSV 목록이 패키지에 포함되어 있으며 기존 목록이 재생성 결과와 다르면 자동 덮어쓰지 않습니다.
파일 바이트의 SHA-256이 특징 파일과 다르면 평가·웹에서 오류를 표시합니다.
목록을 엑셀 등에서 다시 저장하면 해시가 바뀔 수 있습니다.

모델은 최초 특징 생성 또는 검색 때 `models/`로 다운로드됩니다.
검색 시에는 저장된 이미지 특징을 읽으므로 전체 이미지 인코딩을 반복하지 않습니다.

## 평가 재실행

```bash
./.venv311/bin/python evaluate_sizes.py
```

실행하면 `results/`의 해당 규모 비교 CSV/JSON이 새 결과로 갱신됩니다.
기록을 따로 보존하려면 재실행 전에 해당 결과 파일을 복사해 둡니다.

후보 집합만 100 / 500 / 1000장으로 바꾸고 질의는 `subset_100.csv`의 500개 설명으로 고정했습니다.
500장에 연결된 2500개 설명이나 1000장에 연결된 5000개 설명 전체를 평가한 실험은 아닙니다.
각 질의의 정답은 설명에 연결된 하나의 이미지입니다.
모델을 새로 학습하거나 미세 조정하지 않았습니다.

| 후보 수 | 질의 수 | Recall@1 | Recall@5 | Recall@10 | 평균 처리 시간 |
|---:|---:|---:|---:|---:|---:|
| 100 | 500 | 81.60% | 98.20% | 99.20% | 81.14 ms |
| 500 | 500 | 60.20% | 87.20% | 94.80% | 89.42 ms |
| 1000 | 500 | 52.60% | 80.60% | 89.20% | 93.33 ms |

출처는 `results/scale_comparison_fixed500.json`, 동일 이름의 CSV 및 세 개의 질의별 CSV입니다.
초기 `metrics_100.json`의 84.71 ms는 이전 실행값입니다. 두 실행의 시간값을 섞지 않습니다.
Recall은 정답 순위가 K 이하인 질의 수를 500으로 나눈 값입니다.
유사도는 L2 정규화한 이미지·문장 특징의 내적이며 정확도나 확률이 아닙니다.

평가 시간에는 토큰화, 텍스트 인코딩, 유사도 계산, 정렬, 정답 순위 확인이 포함됩니다.
모델·데이터 로딩은 제외하고 워밍업 후 측정했습니다.
웹의 단일 검색 시간은 정답 순위 확인과 화면 렌더링을 포함하지 않아 평가 평균과 직접 같은 측정값은 아닙니다.
세 규모는 100 → 500 → 1000 순서로 한 번씩 실행했습니다.
CPU 부하, 발열 및 실행 순서의 영향을 통제한 반복 시간 실험은 아닙니다.

## 확인한 사례

- `two men jump for the basketball .`: 정답 `3126752627_dc2d6674da.jpg`의 순위는 1 → 9 → 22입니다.
- `A goalie tries to block the puck in a hockey game .`: 정답 `3354489242_dd529ffa1f.jpg`의 순위는 1 → 1 → 2입니다.
- 웹에서 세 후보 규모, Top-5/Top-10 표시, 빈 질의 안내와 안내 후 정상 재검색을 실행 확인했습니다.
- Top-1 선택과 지나치게 긴 입력의 오류 처리는 코드에 있으나 별도 실행 증거를 기록하지 않았습니다.

## 저장 결과 확인

```bash
./.venv311/bin/python validate_results.py
```

이 검사는 Python 표준 라이브러리만 사용합니다.
모델을 실행하거나 웹 화면을 테스트하는 명령은 아닙니다.
소스 문법, 이미지 목록 포함 관계, 500개 질의의 일치 여부를 확인합니다.
질의별 CSV에서 Recall과 평균 시간을 다시 계산해 요약 CSV/JSON과 비교합니다.

## 실험 범위와 한계

한 번 정한 부분집합과 기존 100장에 연결된 500개 질의의 예비 실험입니다.
공식 Flickr8k 전체 테스트 분할의 벤치마크 결과가 아닙니다.
동일 장면의 유사한 다른 이미지가 있어도 지정된 정답 이미지와 다르면 오답으로 계산합니다.
사전 학습 데이터와의 중복 여부는 확인하지 않았습니다.
영어를 기본 입력으로 사용하며 한국어 비교 실험은 수행하지 않았습니다.

## 관련 자료

- Radford et al., Learning Transferable Visual Models From Natural Language Supervision, ICML 2021. https://proceedings.mlr.press/v139/radford21a.html
- Hodosh, Young, Hockenmaier, Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics, JAIR 47, 853-899, 2013. https://doi.org/10.1613/jair.3994
- CLIP 구현: https://github.com/openai/CLIP
