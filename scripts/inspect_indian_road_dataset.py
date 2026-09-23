"""
Inspection script for thirdeyelabs/indian-road-dataset.

Downloads ONLY:
  - annotations/detection.json      (1.31 GB — streamed, first N entries only)
  - annotations/scene_attributes.json (small)
  - gps/gps_tracks.json              (small)
  - The first tar shard (first --num-samples frames only)
"""
import argparse, io, json, os, sys, tarfile, tempfile
from decimal import Decimal
from pathlib import Path
from collections import Counter, defaultdict

from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np


class _DecimalEncoder(json.JSONEncoder):
    """Make ijson Decimal objects JSON-serialisable."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def jdump(obj, **kw):
    """json.dumps with Decimal support, ASCII-safe."""
    return json.dumps(obj, cls=_DecimalEncoder, ensure_ascii=True, **kw)

REPO = "thirdeyelabs/indian-road-dataset"

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--num-samples", type=int, default=5,
                    help="Number of frames to inspect from the first shard")
parser.add_argument("--tmpdir", default=None,
                    help="Reuse existing temp dir to avoid re-downloading")
args = parser.parse_args()
NUM_SAMPLES = args.num_samples

if args.tmpdir and os.path.isdir(args.tmpdir):
    tmpdir = args.tmpdir
    print(f"\n[setup] Reusing existing temp dir: {tmpdir}")
else:
    tmpdir = tempfile.mkdtemp(prefix="irdd_inspect_")
    print(f"\n[setup] temp dir: {tmpdir}")
print(f"[setup] inspecting {NUM_SAMPLES} samples\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Download small metadata files
# ─────────────────────────────────────────────────────────────────────────────

def dl(filename):
    return hf_hub_download(REPO, filename, repo_type="dataset",
                           local_dir=tmpdir, force_download=False)

print("=" * 60)
print("SECTION 1 — SCENE ATTRIBUTES JSON")
print("=" * 60)
scene_path = dl("annotations/scene_attributes.json")
with open(scene_path, encoding="utf-8") as f:
    scene_data = json.load(f)

print(f"Type: {type(scene_data)}")
if isinstance(scene_data, list):
    print(f"Entries: {len(scene_data)}")
    sample_entry = scene_data[0]
elif isinstance(scene_data, dict):
    keys = list(scene_data.keys())
    print(f"Top-level keys ({len(keys)} total): {keys[:10]}")
    sample_entry = scene_data[keys[0]]

print(f"\nFirst entry:\n{json.dumps(sample_entry, indent=2, ensure_ascii=False)[:1000]}")

# Collect attribute values
weather_vals, scene_vals, tod_vals = Counter(), Counter(), Counter()
entries = scene_data if isinstance(scene_data, list) else list(scene_data.values())
for entry in entries:
    if isinstance(entry, dict):
        attrs = entry.get("attributes", entry)
        weather_vals[attrs.get("weather", "MISSING")] += 1
        scene_vals[attrs.get("scene", "MISSING")] += 1
        tod_vals[attrs.get("timeofday", "MISSING")] += 1

print(f"\nWeather values: {dict(weather_vals.most_common(20))}")
print(f"Scene values:   {dict(scene_vals.most_common(20))}")
print(f"TimeOfDay vals: {dict(tod_vals.most_common(20))}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GPS tracks
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 — GPS TRACKS JSON")
print("=" * 60)
gps_path = dl("gps/gps_tracks.json")
with open(gps_path, encoding="utf-8") as f:
    gps_data = json.load(f)

print(f"Type: {type(gps_data)}")
if isinstance(gps_data, list):
    print(f"Entries: {len(gps_data)}")
    gps_sample = gps_data[0]
elif isinstance(gps_data, dict):
    keys = list(gps_data.keys())
    print(f"Top-level keys ({len(keys)} total), first 5: {keys[:5]}")
    gps_sample = gps_data[keys[0]]

print(f"\nFirst entry (truncated):\n{json.dumps(gps_sample, indent=2, ensure_ascii=False)[:1200]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. detection.json — stream just the first few entries (it's 1.2 GB)
#    We use ijson for streaming if available, else fall back to reading the file
#    after download (unavoidable since it's a single JSON array)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — DETECTION.JSON (first few entries only)")
print("=" * 60)

try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    print("[info] ijson not available — will download full file")

det_path = dl("annotations/detection.json")
file_size = os.path.getsize(det_path)
print(f"detection.json size: {file_size / 1e9:.2f} GB")

print(f"\nReading first {NUM_SAMPLES} annotation entries …")
det_entries = []
if HAS_IJSON:
    with open(det_path, "rb") as f:
        for i, item in enumerate(ijson.items(f, "item")):
            det_entries.append(item)
            if i + 1 >= NUM_SAMPLES:
                break
else:
    # Read first chunk looking for complete JSON objects
    with open(det_path, "r", encoding="utf-8") as f:
        # Skip opening '[', collect items until we have enough
        raw = f.read(1_000_000)  # read first 1 MB
    # Find the first NUM_SAMPLES complete objects via simple brace counting
    depth = 0
    start = None
    objects = []
    for i, ch in enumerate(raw):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(raw[start:i+1]))
                except json.JSONDecodeError:
                    pass
                start = None
            if len(objects) >= NUM_SAMPLES:
                break
    det_entries = objects

print(f"Got {len(det_entries)} entries\n")

all_categories = Counter()
has_track_id = False
track_id_examples = []
frame_name_examples = []

for entry in det_entries:
    frame_name_examples.append(entry.get("name", "N/A"))
    labels = entry.get("labels", [])
    for lbl in labels:
        cat = lbl.get("category", "MISSING")
        all_categories[cat] += 1
        if "track_id" in lbl:
            has_track_id = True
            track_id_examples.append({
                "frame": entry.get("name"),
                "category": cat,
                "track_id": lbl["track_id"],
            })

print("Frame names seen:")
for n in frame_name_examples:
    print(f"  {n}")

print(f"\n--- FIRST FULL ANNOTATION ENTRY ---")
print(jdump(det_entries[0], indent=2))

print(f"\nCategories in these {NUM_SAMPLES} entries: {dict(all_categories)}")
print(f"\ntrack_id present: {has_track_id}")
if track_id_examples:
    print("track_id examples:")
    for ex in track_id_examples[:3]:
        print(f"  {ex}")

# Label field structure
if det_entries and det_entries[0].get("labels"):
    lbl0 = det_entries[0]["labels"][0]
    print(f"\nLabel field keys: {list(lbl0.keys())}")
    print(f"Full first label:\n{jdump(lbl0, indent=2)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Stream first tar shard — inspect image + mask + per-frame JSON
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — TAR SHARD INSPECTION (first shard, first samples)")
print("=" * 60)

shard_path = hf_hub_download(
    REPO,
    "data/train-00000-of-00646.tar",
    repo_type="dataset",
    local_dir=tmpdir,
)
print(f"Downloaded shard: {shard_path}  ({os.path.getsize(shard_path)/1e6:.1f} MB)")

samples_seen = 0
file_groups = defaultdict(dict)  # stem -> {jpg, png, json}

with tarfile.open(shard_path, "r") as tar:
    members = tar.getmembers()
    print(f"Members in shard: {len(members)}")
    # Show first 15 member names
    print("First 15 member names:")
    for m in members[:15]:
        print(f"  {m.name}  ({m.size:,} bytes)")

    # Group by stem
    for m in members:
        stem, ext = os.path.splitext(m.name)
        file_groups[stem][ext] = m

    stems = sorted(file_groups.keys())
    print(f"\nUnique frame stems: {len(stems)}")
    print(f"First 5 stems: {stems[:5]}")

    # Inspect first NUM_SAMPLES frames
    print(f"\n--- Inspecting {NUM_SAMPLES} frames ---")
    for stem in stems[:NUM_SAMPLES]:
        group = file_groups[stem]
        print(f"\nStem: {stem}")
        print(f"  Files present: {list(group.keys())}")

        # Image (jpg)
        if ".jpg" in group:
            m = group[".jpg"]
            f = tar.extractfile(m)
            img_bytes = f.read()
            img = Image.open(io.BytesIO(img_bytes))
            print(f"  JPG: size={img.size}  mode={img.mode}  format={img.format}  bytes={len(img_bytes):,}")

        # Segmentation mask (png)
        if ".png" in group:
            m = group[".png"]
            f = tar.extractfile(m)
            png_bytes = f.read()
            mask = Image.open(io.BytesIO(png_bytes))
            mask_arr = np.array(mask)
            unique_vals = sorted(np.unique(mask_arr).tolist())
            print(f"  PNG: size={mask.size}  mode={mask.mode}  dtype={mask_arr.dtype}  unique_vals={unique_vals}")

        # Per-frame JSON annotation
        if ".json" in group:
            m = group[".json"]
            f = tar.extractfile(m)
            frame_ann = json.loads(f.read().decode("utf-8"))
            print(f"  JSON keys: {list(frame_ann.keys())}")
            labels = frame_ann.get("labels", [])
            print(f"  Labels count: {len(labels)}")
            if labels:
                print(f"  First label: {jdump(labels[0], indent=4)}")
        samples_seen += 1

# ─────────────────────────────────────────────────────────────────────────────
# 5. Cleanup temp dir (keep for inspection report)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)
print(f"Temp dir kept for now: {tmpdir}")
print("Run with --cleanup to delete it.")
