"""Recreate the fixed nested Flickr8k subsets without overwriting other lists."""
import csv
import io
import random
from collections import defaultdict
from pathlib import Path


def build_subsets(folder):
    groups = defaultdict(list)
    with (folder / 'captions.txt').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        if next(reader) != ['image', 'caption']:
            raise ValueError('Expected image,caption header.')
        for row in reader:
            if len(row) < 2:
                continue
            name = row[0].strip()
            caption = ','.join(row[1:]).strip()
            if Path(name).name != name:
                raise ValueError('Image name must be a basename.')
            if caption and (folder / 'Images' / name).is_file():
                groups[name].append(caption)

    eligible = sorted(name for name in groups if len(groups[name]) == 5)
    if len(eligible) < 1000:
        raise ValueError('Need at least 1000 images with five captions each.')
    first = random.Random(42).sample(eligible, 100)
    first_set = set(first)
    remaining = [name for name in eligible if name not in first_set]
    names = first + random.Random(42).sample(remaining, 900)
    output = {}
    for size in (100, 500, 1000):
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['image', 'caption'])
        for name in names[:size]:
            for caption in groups[name]:
                writer.writerow([name, caption])
        output[folder / f'subset_{size}.csv'] = stream.getvalue().encode('utf-8')
    return output


def main():
    folder = Path(__file__).resolve().parent / 'data' / 'flickr8k'
    outputs = build_subsets(folder)
    # Check every conflict before writing any file.
    for path, contents in outputs.items():
        if path.exists() and path.read_bytes() != contents:
            raise ValueError(f'{path.name} differs; existing file was kept.')
    for path, contents in outputs.items():
        if path.exists():
            print(f'Unchanged: {path.name}')
        else:
            path.write_bytes(contents)
            print(f'Created: {path.name}')


if __name__ == '__main__':
    main()
