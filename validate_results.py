"""Check source syntax, fixed subsets and saved metrics; no model execution."""
import ast
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    root = Path(__file__).resolve().parent
    data = root / 'data' / 'flickr8k'
    results = root / 'results'
    sources = sorted(root.glob('*.py'))
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8'), filename=path.name)
    print(f'Python syntax: {len(sources)} files passed')

    subsets = {}
    candidates = {}
    for size in (100, 500, 1000):
        rows = read_csv(data / f'subset_{size}.csv')
        counts = Counter(row['image'] for row in rows)
        require(len(rows) == size * 5, f'{size}: caption count')
        require(len(counts) == size and set(counts.values()) == {5},
                f'{size}: images must have five captions each')
        require(all(row['caption'].strip() for row in rows), 'Empty caption')
        subsets[size] = rows
        candidates[size] = list(counts)
    require(subsets[500][:500] == subsets[100], '100 must prefix 500')
    require(subsets[1000][:2500] == subsets[500], '500 must prefix 1000')
    print('Subset sizes, captions and nesting: passed')

    report = json.loads((results / 'scale_comparison_fixed500.json').read_text())
    require(report['query_sha256'] == hashlib.sha256(
        (data / 'subset_100.csv').read_bytes()).hexdigest(), 'Query hash mismatch')
    require(report['model'] == 'ViT-B/32' and report['device'] == 'cpu',
            'Model/device metadata mismatch')
    summaries = {row['candidate_images']: row for row in report['results']}
    csv_summaries = {int(row['candidate_images']): row for row in read_csv(
        results / 'scale_comparison_fixed500.csv')}
    require(set(summaries) == set(csv_summaries) == {100, 500, 1000},
            'Missing candidate size')
    previous_ranks = None
    for size in (100, 500, 1000):
        rows = read_csv(results / f'query_results_fixed500_{size}.csv')
        require(len(rows) == 500, f'{size}: expected 500 results')
        require([(r['correct_image'], r['query']) for r in rows] ==
                [(r['image'], r['caption']) for r in subsets[100]],
                f'{size}: query order/content mismatch')
        ranks = [int(row['correct_rank']) for row in rows]
        require(all(1 <= rank <= size for rank in ranks), 'Invalid rank')
        for row, rank in zip(rows, ranks):
            require(row['top1_image'] in candidates[size], 'Unknown top image')
            require((rank == 1) == (row['top1_image'] == row['correct_image']),
                    'Top-1 identity/rank mismatch')
            score = float(row['top1_similarity'])
            require(math.isfinite(score) and -1.00001 <= score <= 1.00001,
                    'Invalid cosine similarity')
            seconds = float(row['query_seconds'])
            require(math.isfinite(seconds) and seconds >= 0, 'Invalid query time')
        if previous_ranks is not None:
            require(all(a <= b for a, b in zip(previous_ranks, ranks)),
                    'Rank improved unexpectedly in a nested candidate set')
        previous_ranks = ranks
        computed = {f'recall_at_{k}': sum(r <= k for r in ranks) / 500
                    for k in (1, 5, 10)}
        computed['mean_query_ms'] = statistics.mean(
            float(r['query_seconds']) for r in rows) * 1000
        require(summaries[size]['queries'] == 500 and
                int(csv_summaries[size]['queries']) == 500, 'Query count mismatch')
        for key, value in computed.items():
            require(math.isclose(value, float(summaries[size][key]), abs_tol=1e-9)
                    and math.isclose(value, float(csv_summaries[size][key]),
                                     abs_tol=1e-9), f'{size}: {key} mismatch')
        print(f'{size} images: CSV, JSON, Recall and mean time agree')
    print('Saved-data checks passed. Model inference was not run by this check.')


if __name__ == '__main__':
    main()
