"""Build the ShapeNet-Part object-to-Gaussian filename mapping."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


SPLIT_FILES = (
    "shuffled_train_file_list.json",
    "shuffled_val_file_list.json",
    "shuffled_test_file_list.json",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partanno-root",
        type=Path,
        required=True,
        help="ShapeNet-Part root containing train_test_split/.",
    )
    parser.add_argument(
        "--gs-root",
        type=Path,
        required=True,
        help="Directory containing <category>-<object>.ply Gaussian assets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("split_to_org_gs_map.json"),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write null entries instead of failing when an asset is missing.",
    )
    return parser.parse_args()


def index_gaussians(gs_root):
    by_object = defaultdict(list)
    for path in gs_root.glob("*.ply"):
        if "-" not in path.stem:
            continue
        _, object_id = path.stem.split("-", 1)
        by_object[object_id].append(path.name)
    return by_object


def main():
    args = parse_args()
    split_root = args.partanno_root / "train_test_split"
    by_object = index_gaussians(args.gs_root)
    mapping = {}
    missing = []
    entries = []

    for filename in SPLIT_FILES:
        with (split_root / filename).open() as handle:
            entries.extend(json.load(handle))

    for item in tqdm(entries, desc="matching ShapeNet-Part assets"):
        category_id, object_id = item.rsplit("/", 2)[-2:]
        key = f"{category_id}-{object_id}"
        exact = f"{key}.ply"
        candidates = by_object.get(object_id, [])
        match = exact if exact in candidates else (candidates[0] if len(candidates) == 1 else None)
        mapping[key] = match
        if match is None:
            missing.append(key)

    if missing and not args.allow_missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Missing {len(missing)} Gaussian assets (first entries: {preview}). "
            "Use --allow-missing only when incomplete coverage is intentional."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(mapping, handle, indent=2, sort_keys=True)
    print(f"Wrote {len(mapping)} entries to {args.output}; missing={len(missing)}")


if __name__ == "__main__":
    main()
