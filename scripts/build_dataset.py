import os
import shutil
import argparse
from collections import defaultdict
import json

RAW_DIR = "dataset"
PROCESSED_DIR = "data/processed"

TIME_LABELS = ["day", "evening", "night"]
WEATHER_LABELS = ["clear", "cloudy", "partly_cloudy"]

TIME_DIR_MAP = {
    "day": "day",
    "evening": "evening",
    "night": "night (Nightvision)",
}

WEATHER_DIR_MAP = {
    "clear": "clear",
    "cloudy": "cloudy",
    "partly_cloudy": "partly_cloudy",
}


def analyze_dataset(raw_dir):
    stats = {}
    for root, dirs, files in os.walk(raw_dir):
        for d in dirs:
            path = os.path.join(root, d)
            count = len([f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            stats[d] = count
    return stats


def build_combined_dataset(raw_dir, output_dir, symlink=True):
    os.makedirs(output_dir, exist_ok=True)
    link_fn = os.symlink if symlink else shutil.copy2

    combined = defaultdict(list)

    for time_label, time_dir in TIME_DIR_MAP.items():
        time_path = os.path.join(raw_dir, time_dir)
        if not os.path.isdir(time_path):
            continue
        for fname in os.listdir(time_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            for weather_label in WEATHER_LABELS:
                combined[f"{time_label}_{weather_label}"].append(
                    os.path.join(time_path, fname)
                )

    for weather_label, weather_dir in WEATHER_DIR_MAP.items():
        weather_path = os.path.join(raw_dir, weather_dir)
        if not os.path.isdir(weather_path):
            continue
        for fname in os.listdir(weather_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            for time_label in TIME_LABELS:
                combined[f"{time_label}_{weather_label}"].append(
                    os.path.join(weather_path, fname)
                )

    for class_name, file_paths in combined.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        added = set()
        for src_path in file_paths:
            base = os.path.basename(src_path)
            dest = os.path.join(class_dir, base)
            if base in added:
                name, ext = os.path.splitext(base)
                dest = os.path.join(class_dir, f"{name}_{len(added)}{ext}")
            try:
                link_fn(src_path, dest)
            except FileExistsError:
                pass
            added.add(os.path.basename(dest))

    return {k: len(v) for k, v in combined.items()}


def main():
    parser = argparse.ArgumentParser(description="Build combined time+weather dataset")
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--output-dir", default=PROCESSED_DIR)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    parser.add_argument("--analyze", action="store_true", help="Only analyze, don't build")
    args = parser.parse_args()

    stats = analyze_dataset(args.raw_dir)
    print("=" * 50)
    print("Dataset Analysis (raw)")
    print("=" * 50)
    for k, v in sorted(stats.items()):
        print(f"  {k:35s}: {v:5d} images")
    total_raw = sum(stats.values())
    print(f"  {'TOTAL':35s}: {total_raw:5d} images")

    if args.analyze:
        return

    print("\nBuilding combined dataset...")
    class_counts = build_combined_dataset(args.raw_dir, args.output_dir, symlink=not args.copy)

    print("\n" + "=" * 50)
    print("Combined Dataset Classes")
    print("=" * 50)
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:30s}: {count:5d} images")
    total_combined = sum(class_counts.values())
    print(f"  {'TOTAL':30s}: {total_combined:5d} images")

    info = {
        "time_labels": TIME_LABELS,
        "weather_labels": WEATHER_LABELS,
        "classes": list(class_counts.keys()),
        "class_counts": class_counts,
        "total_images": total_combined,
    }
    with open(os.path.join(args.output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDataset saved to {args.output_dir}/")
    print("Use --copy to copy files instead of symlinks")


if __name__ == "__main__":
    main()
