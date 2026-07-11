import os
import shutil
import argparse
from datetime import datetime
import json

CAPTURES_DIR = "captures"
OUTPUT_DIR = "data/captures_labeled"

def infer_time_of_day(hour):
    if 6 <= hour < 12:
        return "day"
    elif 12 <= hour < 18:
        return "day"
    elif 18 <= hour < 21:
        return "evening"
    else:
        return "night"


def process_captures(captures_dir, output_dir, copy=False):
    os.makedirs(output_dir, exist_ok=True)
    stats = {"day": 0, "evening": 0, "night": 0, "unknown": 0, "skipped": 0}

    for fname in os.listdir(captures_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            stats["skipped"] += 1
            continue

        try:
            ts_part = fname.replace("capture_", "").replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
            dt = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
            time_label = infer_time_of_day(dt.hour)
        except (ValueError, IndexError):
            stats["unknown"] += 1
            continue

        time_dir = os.path.join(output_dir, time_label)
        os.makedirs(time_dir, exist_ok=True)

        src = os.path.join(captures_dir, fname)
        dst = os.path.join(time_dir, fname)
        if copy:
            shutil.copy2(src, dst)
        else:
            if not os.path.exists(dst):
                os.symlink(src, dst)

        stats[time_label] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Label captures by timestamp")
    parser.add_argument("--captures-dir", default=CAPTURES_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = parser.parse_args()

    stats = process_captures(args.captures_dir, args.output_dir, copy=args.copy)

    print("=" * 50)
    print("Captures Labeled by Time of Day")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k:15s}: {v:5d}")
    print(f"  {'TOTAL':15s}: {sum(v for k, v in stats.items() if k != 'skipped'):5d}")

    with open(os.path.join(args.output_dir, "label_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
