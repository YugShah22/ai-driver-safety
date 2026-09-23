# Indian Road Driving Dataset — Inspection Report

**Dataset:** `thirdeyelabs/indian-road-dataset`  
**Inspected:** 2026-09-23  
**Inspection method:** HuggingFace Hub authenticated streaming  
**Samples inspected:** 5 frames from shard `train-00000-of-00646.tar`; first 5 entries of `detection.json`  
**Files downloaded:** `annotations/detection.json` (1.31 GB), `annotations/scene_attributes.json`, `gps/gps_tracks.json`, `data/train-00000-of-00646.tar` (298.5 MB)

> **Legend**
> - **OBSERVED** — confirmed by direct file inspection
> - **DOCUMENTED** — stated in the dataset README only, not directly verified
> - **ASSUMED** — inferred but not confirmed

---

## 1. Dataset Access

| Item | Result |
|------|--------|
| Access method | HuggingFace Hub authenticated (`hf_hub_download`) |
| Authentication required | Yes — dataset is **gated** (free, instant access after accepting terms) |
| Streaming without auth | ❌ HTTP 401 on all files |
| ijson streaming of detection.json | ✅ Works — avoids loading 1.31 GB into RAM |
| WebDataset / tar streaming | ✅ Works via `tarfile` on downloaded shard |
| HF `datasets` library `load_dataset()` | ✅ DOCUMENTED to work (not tested directly) |

---

## 2. Repository File Structure

**OBSERVED:**

```
thirdeyelabs/indian-road-dataset/
├── README.md
├── .gitattributes
├── annotations/
│   ├── detection.json          # 1.31 GB — all 645,714 frames, BDD100K format
│   └── scene_attributes.json   # ~small — per-clip weather / scene / timeofday
├── gps/
│   └── gps_tracks.json         # per-clip GPS tracks
└── data/
    ├── train-00000-of-00646.tar
    ├── train-00001-of-00646.tar
    ⋮
    └── train-00645-of-00646.tar
```

- **Total files in repo:** 651 (646 tar shards + 5 metadata files)
- **Split prefixes observed:** `train` only — no `val` or `test` tar shards exist
- **Shard count:** 646

---

## 3. Dataset Statistics

**DOCUMENTED (from README):**

| Metric | Value |
|--------|-------|
| Total clips | 8,437 |
| Annotated frames | 645,714 |
| Object detections | 3,669,119 |
| Segmentation masks | 1,290,463 |
| Annotation format | BDD100K |
| Label provenance | Machine-generated (not human-verified) |
| Sensor | Monocular dashcam (CP Plus) |
| Location | Delhi NCR, India |

**OBSERVED:**
- `scene_attributes.json` has **8,434 entries** (3 fewer than documented 8,437 clips)
- `gps_tracks.json` has **5,689 entries** — not all clips have GPS
- First shard (`train-00000`) has **1,000 frames** (3,000 tar members / 3 files per frame)

---

## 4. Tar Shard Structure

**OBSERVED — from `train-00000-of-00646.tar`:**

```
{clip_id}_{frame:04d}.jpg     # dashcam image
{clip_id}_{frame:04d}.png     # segmentation mask
{clip_id}_{frame:04d}.json    # per-frame BDD100K annotations
```

Example (first 3 members of first shard):
```
000db725-8180-4770-8b6a-4eb74aeda9f9_0000.jpg    (676,668 bytes)
000db725-8180-4770-8b6a-4eb74aeda9f9_0000.png    (13,480 bytes)
000db725-8180-4770-8b6a-4eb74aeda9f9_0000.json   (254 bytes)
```

> ⚠️ **Important naming discrepancy:**  
> Inside tar: `{clip_id}_{frame:04d}` (underscore separator, zero-padded 4-digit frame)  
> Inside `detection.json` `.name` field: `{clip_id}/{frame:04d}.jpg` (slash separator)  
> The adapter must handle both conventions when cross-referencing.

---

## 5. Image Properties

**OBSERVED — all 5 inspected frames:**

| Property | Value |
|----------|-------|
| Format | JPEG |
| Dimensions | 1920 × 994 pixels |
| Mode | RGB |
| Approximate file size | 387 KB – 677 KB per frame |

> All 5 frames from the first clip share the same resolution (1920×994). Whether this is consistent across all clips is **ASSUMED** but not confirmed.

---

## 6. Segmentation Masks

**OBSERVED — all 5 inspected masks:**

| Property | Value |
|----------|-------|
| Format | PNG |
| Dimensions | **1920 × 994** (matches image exactly) |
| PIL mode | `L` (8-bit grayscale) |
| dtype | `uint8` |
| Pixel value range | 0–33 (observed across 5 frames) |

**Unique pixel values observed per frame:**

| Frame | Unique pixel values |
|-------|---------------------|
| `_0000` | 0, 2, 5, 12, 19, 20, 24, 26, 32, 33 |
| `_0001` | 2, 12, 19, 20, 26, 28, 32, 33 |
| `_0002` | 0, 2, 11, 12, 19, 20, 26, 32, 33 |
| `_0003` | 0, 2, 12, 19, 20, 23, 26, 29, 32, 33 |
| `_0004` | 0, 2, 5, 6, 9, 12, 19, 20, 23, 26, 28, 29, 32, 33 |

> ⚠️ **Semantic meanings of pixel values NOT confirmed.**  
> The mask values are integer class IDs, but the exact mapping (e.g., `2 = road`, `19 = vehicle`) was not found in any inspected file. The README states segmentation uses SegFormer trained on IDD, which uses the IDD segmentation label set — but we did NOT verify the exact pixel-to-class mapping from the dataset files themselves.  
> **Do not assign semantic class names to these values until the mapping is confirmed.**

---

## 7. Per-Frame JSON Annotation (inside tar)

**OBSERVED — structure of `{clip_id}_{frame:04d}.json` inside shard:**

```json
{
  "name": "000db725-8180-4770-8b6a-4eb74aeda9f9_0000",
  "labels": [
    {
      "id": 0,
      "category": "car",
      "box2d": { "x1": 0.0, "y1": 458.49, "x2": 99.67, "y2": 897.87 },
      "confidence": 0.6539,
      "attributes": { "occluded": false, "truncated": false },
      "track_id": 1
    }
  ]
}
```

**OBSERVED top-level keys in per-frame JSON:** `name`, `labels`  
**NOT present in per-frame JSON:** `timestamp`, `attributes` (weather/scene/timeofday)  

> Scene attributes (weather, scene type, time of day) are **not embedded per frame** in the tar JSONs. They are clip-level and stored separately in `annotations/scene_attributes.json`.

---

## 8. Detection.json Structure (bulk annotation file)

**OBSERVED — structure of entries in `annotations/detection.json`:**

Frame name format: `{clip_id}/{frame:04d}.jpg`

```json
{
  "name": "8707696d-63d7-42c8-9829-47a44338fe29/0000.jpg",
  "labels": [
    {
      "id": 0,
      "category": "car",
      "box2d": { "x1": 952.54, "y1": 252.76, "x2": 1188.84, "y2": 429.51 },
      "confidence": 0.4368,
      "attributes": { "occluded": false, "truncated": false },
      "track_id": 3
    },
    ...
  ]
}
```

**OBSERVED label field keys:** `id`, `category`, `box2d`, `confidence`, `attributes`, `track_id`  
**`box2d` sub-keys:** `x1`, `y1`, `x2`, `y2` (floats)  
**`attributes` sub-keys:** `occluded` (bool), `truncated` (bool)  

> ⚠️ **`confidence` is present** — this is a machine-generated dataset. Confidence scores are available per detection label. The README notes ~33% of detections are below 0.5 confidence.

> ⚠️ **`timestamp` is NOT present** in the observed detection.json entries — despite the README schema example showing it. The field may be absent or zero for all entries. This requires further verification.

---

## 9. Detection Categories

**DOCUMENTED (README — 12 classes):**

| Category string | Description |
|-----------------|-------------|
| `person` | Pedestrians |
| `rider` | Motorcyclists/cyclists with rider |
| `car` | Passenger cars |
| `truck` | Trucks and tempos |
| `bus` | Buses |
| `motorcycle` | Motorcycles (unridden) |
| `bicycle` | Bicycles |
| `autorickshaw` | Auto-rickshaws (tuk-tuks) |
| `animal` | Cattle, dogs, animals on road |
| `vehicle fallback` | Unclassified vehicles |
| `traffic light` | Traffic signals |
| `traffic sign` | Road signs and boards |

**OBSERVED in first 5 frames only:** `car`, `person`, `traffic sign`

> The full 12-class list is from the README. Only 3 classes appeared in the 5 inspected frames. The complete class list should be taken from the README until verified from the full `detection.json`.

---

## 10. Track IDs

**OBSERVED:**

- `track_id` field is present on **every label** in both `detection.json` and per-frame tar JSONs
- Data type: **integer**
- Example: frame `0000` has `track_id: 3`, frame `0001` of the same clip shows `track_id: 34` for a different detection
- Track ID `2` observed on the same category (`car`) across frames `_0001` through `_0004` of clip `000db725...` — consistent with ByteTrack persistent IDs

**DOCUMENTED:** Track IDs are ByteTrack IDs that persist across frames within a clip.

> ⚠️ Cross-frame persistence observed for `track_id: 2` in the inspected clip. Track IDs appear to be clip-scoped (not global across clips).

---

## 11. Scene Attributes

**OBSERVED — `annotations/scene_attributes.json` structure:**

```json
{
  "{clip_id}": {
    "weather": "clear",
    "scene": "tunnel",
    "timeofday": "daytime"
  },
  ...
}
```

- Keyed by **clip UUID** (same UUID format as in tar filenames and detection.json)
- **8,434 entries** (clip-level, not frame-level)

**OBSERVED value distributions:**

| Attribute | Values and counts |
|-----------|-------------------|
| `weather` | `clear`: 7,348 · `overcast`: 485 · `foggy`: 400 · `rainy`: 201 |
| `scene` | `residential road`: 2,996 · `city street`: 2,237 · `highway`: 1,277 · `tunnel`: 905 · `village road`: 680 · `parking lot`: 339 |
| `timeofday` | `daytime`: 5,102 · `night`: 2,439 · `dusk`: 781 · `dawn`: 112 |

---

## 12. GPS Information

**OBSERVED — `gps/gps_tracks.json` structure:**

```json
{
  "{clip_id}": [
    {
      "ts": 1773723043000,
      "lat": 28.432738,
      "lon": 77.014969,
      "speed_kmh": 9.7995,
      "course_deg": 307.788
    },
    ...
  ]
}
```

- Keyed by **clip UUID**
- Each entry is a **list of GPS waypoints**, one per second of the clip
- **5,689 clips** have GPS data (out of 8,434 clips in scene_attributes → **3,745 clips lack GPS**)
- Fields per waypoint: `ts` (Unix ms timestamp), `lat`, `lon`, `speed_kmh`, `course_deg`

> `speed_kmh` is available directly — no need to compute from coordinates.  
> Frame-to-GPS matching: frames are extracted at 1 fps (DOCUMENTED), GPS is 1 Hz → can align by frame index to GPS waypoint index.

---

## 13. Key Observations and Potential Issues

| # | Issue | Severity |
|---|-------|----------|
| 1 | **Naming mismatch** between tar (`clip_id_frame`) and detection.json (`clip_id/frame.jpg`) | 🟡 Medium — adapter must normalise |
| 2 | **No val/test splits** — only a `train` split exists in tar shards | 🟡 Medium — adapter must create synthetic splits |
| 3 | **Scene attributes are clip-level**, not frame-level | 🟡 Medium — adapter must propagate to frames |
| 4 | **GPS absent for ~44% of clips** | 🟡 Medium — GPS lookup must be optional |
| 5 | **Confidence scores present** — ~33% of labels below 0.5 | 🟡 Medium — adapter or training should filter by confidence |
| 6 | **Machine-generated labels** — not human-verified | 🟠 High — treat as pseudo-labels, not ground truth |
| 7 | **Segmentation class map unknown** — pixel values 0–33 observed, no legend found | 🟠 High — do not use masks until mapping confirmed |
| 8 | **`timestamp` field absent** in observed detection.json entries | 🟡 Medium — README example shows it, reality may differ |
| 9 | **1.31 GB detection.json** — cannot be loaded fully into RAM easily | 🟡 Medium — adapter must use ijson streaming |
| 10 | **Only 1 split in tar** — adapter cannot rely on separate val/test shards | 🟡 Medium — must deterministically split by clip UUID |

---

## 14. Recommended Adapter Input Format

Based on **OBSERVED** data only:

```
Input:  tar shards  OR  detection.json + scene_attributes.json + gps_tracks.json
        accessed via HuggingFace Hub (requires HF_TOKEN env var)
        OR pre-downloaded locally

Per sample delivered to FrameDataset:
  image_path:   Path to extracted .jpg frame
  label:        integer class index (from 12-class list, see Section 9)
  metadata: {
    "clip_id":     str   — UUID of the clip
    "frame_idx":   int   — 0-based frame index within clip
    "category":    str   — exact category string from annotation
    "box2d":       dict  — {x1, y1, x2, y2} floats
    "confidence":  float — detection confidence
    "track_id":    int   — ByteTrack ID (clip-scoped)
    "occluded":    bool
    "truncated":   bool
    "weather":     str | None  — from scene_attributes.json
    "scene":       str | None  — from scene_attributes.json
    "timeofday":   str | None  — from scene_attributes.json
    "lat":         float | None
    "lon":         float | None
    "speed_kmh":   float | None
  }
```

**Split strategy (ASSUMED — no official split exists):**
Deterministic train/val/test split by clip UUID hash (e.g., 80/10/10).  
This keeps all frames from a clip in the same split, preventing temporal data leakage.

---

## 15. Files Inspected

| File | Size | Status |
|------|------|--------|
| `README.md` | 7.0 KB | ✅ Fully read |
| `annotations/scene_attributes.json` | small | ✅ Fully parsed |
| `gps/gps_tracks.json` | small | ✅ Fully parsed (first entry shown) |
| `annotations/detection.json` | 1.31 GB | ✅ First 5 entries streamed via ijson |
| `data/train-00000-of-00646.tar` | 298.5 MB | ✅ First 5 frames inspected |

Temp files cleaned up after inspection.
