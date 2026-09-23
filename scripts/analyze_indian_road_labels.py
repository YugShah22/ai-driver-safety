"""
Label analysis script for thirdeyelabs/indian-road-dataset.

Downloads:
  - annotations/scene_attributes.json  (small — full parse)
  - annotations/detection.json         (1.31 GB — full ijson streaming pass)

Produces exact counts for scene / weather / timeofday / detection categories.
No images or tar shards are downloaded.
"""
import argparse, json, os, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path
from decimal import Decimal

import ijson
from huggingface_hub import hf_hub_download

REPO = "thirdeyelabs/indian-road-dataset"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tmpdir", default=None,
                    help="Reuse existing temp dir (avoids re-download)")
args = parser.parse_args()

if args.tmpdir and os.path.isdir(args.tmpdir):
    tmpdir = args.tmpdir
    print(f"[setup] Reusing temp dir: {tmpdir}")
else:
    tmpdir = tempfile.mkdtemp(prefix="irdd_labels_")
    print(f"[setup] Temp dir: {tmpdir}")

def dl(filename):
    return hf_hub_download(REPO, filename, repo_type="dataset",
                           local_dir=tmpdir, force_download=False)

# ─────────────────────────────────────────────────────────────────────────────
# 1. scene_attributes.json — full parse (small file)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/2] Loading scene_attributes.json …")
scene_path = dl("annotations/scene_attributes.json")
with open(scene_path, encoding="utf-8") as f:
    scene_data = json.load(f)

# Structure: { clip_uuid: { "weather": ..., "scene": ..., "timeofday": ... } }
weather_counter  = Counter()
scene_counter    = Counter()
tod_counter      = Counter()
clips_total      = len(scene_data)

for clip_id, attrs in scene_data.items():
    weather_counter[attrs.get("weather", "__MISSING__")] += 1
    scene_counter[attrs.get("scene",   "__MISSING__")] += 1
    tod_counter[attrs.get("timeofday", "__MISSING__")] += 1

print(f"  Clips in scene_attributes.json: {clips_total:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. detection.json — full streaming pass via ijson
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/2] Streaming detection.json (full pass — this may take a few minutes) …")
det_path = dl("annotations/detection.json")
file_size_gb = os.path.getsize(det_path) / 1e9
print(f"  File size: {file_size_gb:.2f} GB")

category_counter       = Counter()  # category -> detection count
frames_with_category   = defaultdict(set)  # category -> set of frame names
clips_with_category    = defaultdict(set)  # category -> set of clip UUIDs
frames_total           = 0
frames_empty           = 0  # frames with no labels
labels_total           = 0
confidence_below_05    = 0
confidence_total       = 0

with open(det_path, "rb") as f:
    for entry in ijson.items(f, "item"):
        frames_total += 1
        name   = entry.get("name", "")
        labels = entry.get("labels", [])

        # Derive clip_id from "clip_uuid/frame.jpg"
        parts = name.split("/")
        clip_id = parts[0] if len(parts) >= 2 else "__unknown__"

        if not labels:
            frames_empty += 1
            continue

        for lbl in labels:
            cat = lbl.get("category", "__MISSING__")
            category_counter[cat] += 1
            frames_with_category[cat].add(name)
            clips_with_category[cat].add(clip_id)
            labels_total += 1
            conf = lbl.get("confidence")
            if conf is not None:
                confidence_total += 1
                conf_f = float(conf) if isinstance(conf, Decimal) else conf
                if conf_f < 0.5:
                    confidence_below_05 += 1

        if frames_total % 50_000 == 0:
            print(f"  … {frames_total:,} frames processed …")

print(f"  Done. {frames_total:,} frames processed.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Compute per-category frames and clips
# ─────────────────────────────────────────────────────────────────────────────
frames_per_cat = {cat: len(s) for cat, s in frames_with_category.items()}
clips_per_cat  = {cat: len(s) for cat, s in clips_with_category.items()}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Print results
# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 70

print(f"\n{SEP}")
print("SCENE ATTRIBUTES (clip-level)")
print(SEP)

print(f"\n--- weather ({len(weather_counter)} unique values) ---")
for val, cnt in weather_counter.most_common():
    pct = cnt / clips_total * 100
    print(f"  {val!r:<20} {cnt:>6,} clips  ({pct:.1f}%)")

print(f"\n--- scene ({len(scene_counter)} unique values) ---")
for val, cnt in scene_counter.most_common():
    pct = cnt / clips_total * 100
    print(f"  {val!r:<25} {cnt:>6,} clips  ({pct:.1f}%)")

print(f"\n--- timeofday ({len(tod_counter)} unique values) ---")
for val, cnt in tod_counter.most_common():
    pct = cnt / clips_total * 100
    print(f"  {val!r:<20} {cnt:>6,} clips  ({pct:.1f}%)")

print(f"\n{SEP}")
print("DETECTION CATEGORIES (frame-level)")
print(SEP)
print(f"\nTotal frames in detection.json : {frames_total:,}")
print(f"Frames with no labels         : {frames_empty:,}  ({frames_empty/frames_total*100:.1f}%)")
print(f"Total label instances         : {labels_total:,}")
print(f"Confidence total / <0.5       : {confidence_total:,} / {confidence_below_05:,}  ({confidence_below_05/max(confidence_total,1)*100:.1f}%)")

print(f"\n{'Category':<25} {'Detections':>12} {'Frames':>10} {'Clips':>8}  {'Det%':>6}  {'Frm%':>6}")
print("-" * 75)
for cat, det_count in category_counter.most_common():
    frm = frames_per_cat.get(cat, 0)
    clp = clips_per_cat.get(cat, 0)
    det_pct = det_count / max(labels_total, 1) * 100
    frm_pct = frm / max(frames_total, 1) * 100
    print(f"  {cat:<23} {det_count:>12,} {frm:>10,} {clp:>8,}  {det_pct:>5.1f}%  {frm_pct:>5.1f}%")

print(f"\n{SEP}")
print(f"Temp dir: {tmpdir}")
