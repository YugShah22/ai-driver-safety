# IndianRoadAdapter — Design Document

**Status:** Skeleton complete. Core logic pending implementation.  
**Target:** Phase 5 scene-classification CNN  
**File:** [`ml/datasets/adapters/indian_road.py`](../../ml/datasets/adapters/indian_road.py)  
**Tests:** [`backend/tests/test_indian_road_adapter.py`](../../backend/tests/test_indian_road_adapter.py)

---

## 1. What the Adapter Receives

The adapter is constructed with a `DatasetConfig` and reads three annotation files from the dataset root:

| File | Size | Contents |
|------|------|----------|
| `annotations/scene_attributes.json` | Small | Dict keyed by clip UUID → `{weather, scene, timeofday}` |
| `annotations/detection.json` | 1.31 GB | List of per-frame BDD100K annotation entries |
| `gps/gps_tracks.json` | Small | Dict keyed by clip UUID → list of GPS waypoints |

Frames themselves come from one of three sources (selected via `config.extra["source"]`):

| Source | Description |
|--------|-------------|
| `"local"` | Pre-extracted JPEG frames on disk at `<root>/frames/<clip_uuid>/<frame:04d>.jpg` |
| `"tar"` | Local tar shards at `<root>/data/train-NNNNN-of-00646.tar` (unextracted) |
| `"hf_hub"` | Streamed from HuggingFace Hub — requires `HF_TOKEN` env var |

---

## 2. What the Adapter Must Return

Every adapter method that produces samples must yield `Sample` objects:

```python
@dataclass
class Sample:
    image_path: Path          # Absolute path to JPEG frame on disk
    label:      int           # Scene class index (0–5)
    metadata:   dict[str, Any]
```

The `label` is the integer index into `SCENE_CLASSES`:

```python
SCENE_CLASSES = [
    "city street",       # label 0
    "highway",           # label 1
    "parking lot",       # label 2
    "residential road",  # label 3
    "tunnel",            # label 4
    "village road",      # label 5
]
```

---

## 3. How Clip-Level Scene Labels Become Frame-Level CNN Labels

The dataset stores scene labels **per clip**, not per frame.

```
scene_attributes.json:
{
  "clip_uuid_A": {"scene": "highway", "weather": "clear", "timeofday": "daytime"},
  "clip_uuid_B": {"scene": "tunnel",  ...}
}
```

Every frame extracted from clip `clip_uuid_A` receives `label = 1` (the index of `"highway"` in `SCENE_CLASSES`). The adapter performs this propagation in `_build_sample()`:

```python
scene_str  = self._scene_attrs[clip_id]["scene"]   # "highway"
label      = self._class_index[scene_str]           # 1
```

This is the only correct mapping. Do not derive the label from the frame filename, bounding box category, or any other field.

---

## 4. How Clip-Based Splitting Prevents Data Leakage

The dataset has **no official train/val/test split**. All 645,714 frames are under a single `train` prefix in the tar shards.

Naively assigning frames to splits frame-by-frame would put consecutive frames from the same clip into different splits, causing temporal leakage (the model sees nearly identical frames in both train and val).

The correct strategy is **clip-level splitting**:

```python
import hashlib

def _assign_splits(clip_uuids):
    result = {}
    for uuid in clip_uuids:
        h = int(hashlib.md5(uuid.encode()).hexdigest(), 16) % 100
        if h < 80:
            result[uuid] = "train"   # ~80% of clips
        elif h < 90:
            result[uuid] = "val"     # ~10% of clips
        else:
            result[uuid] = "test"    # ~10% of clips
    return result
```

**Properties of this approach:**
- All frames from one clip land in exactly one split
- The assignment is deterministic — same UUID always maps to same split
- No randomness — reproducible across machines without a seed
- Expected split sizes: ~6,747 train clips / ~843 val / ~844 test

---

## 5. How Metadata Should Be Attached to Samples

Each `Sample.metadata` dict should carry the following keys:

```python
metadata = {
    # Identity
    "clip_id":    str,           # clip UUID
    "frame_idx":  int,           # 0-based frame index within clip

    # Scene attributes (clip-level — same for all frames in clip)
    "scene":      str,           # e.g. "highway"      (= the CNN label string)
    "weather":    str,           # e.g. "clear"
    "timeofday":  str,           # e.g. "daytime"

    # Detections (frame-level — filtered to min_confidence)
    "detections": list[dict],    # each dict: {category, box2d, confidence, track_id, occluded, truncated}

    # GPS (clip-level, matched to frame by index — None if clip has no GPS)
    "lat":        float | None,
    "lon":        float | None,
    "speed_kmh":  float | None,
    "course_deg": float | None,
}
```

**Rules:**
- `metadata["detections"]` must only contain detections with `confidence >= min_confidence`
- GPS fields must be `None` for the ~44% of clips that have no GPS track
- Model code must never require any metadata key — `sample.metadata` is informational only

---

## 6. How Local vs HuggingFace Data Access Should Be Separated

Three private methods handle data access. They share the same output contract but differ in implementation:

```
_iter_frames_local(clip_ids)  → yields (image_path, clip_id, frame_idx)
_iter_frames_tar(clip_ids)    → yields (image_path, clip_id, frame_idx)
_iter_frames_hf_hub(clip_ids) → yields (image_path, clip_id, frame_idx)
```

`_iter_frames_for_split(split)` is the single dispatch point:

```python
def _iter_frames_for_split(self, split):
    clip_ids = [c for c, s in self._split_map.items() if s == split]

    if self._source == "local":
        frame_iter = self._iter_frames_local(clip_ids)
    elif self._source == "tar":
        frame_iter = self._iter_frames_tar(clip_ids)
    elif self._source == "hf_hub":
        frame_iter = self._iter_frames_hf_hub(clip_ids)

    for image_path, clip_id, frame_idx in frame_iter:
        detections = ...  # load from per-frame JSON or detection.json
        yield self._build_sample(image_path, clip_id, frame_idx, detections)
```

This design means you can implement `"local"` first, verify it works end-to-end with the CNN, and add `"tar"` and `"hf_hub"` later without touching any other method.

### Recommended implementation order

1. **`"local"` first** — simplest, no tar parsing, no network
2. **`"tar"` second** — avoids full extraction, good for disk-constrained environments
3. **`"hf_hub"` last** — useful for training on cloud without local storage

---

## 7. Which Parts You Should Implement

The following methods have `raise NotImplementedError("TODO: ...")` and are yours to implement:

| Method | Difficulty | Notes |
|--------|-----------|-------|
| `_assign_splits()` | ⭐ Easy | ~10 lines. Use hashlib.md5 + modulo. |
| `_load_scene_attributes()` (inside `load_annotations`) | ⭐ Easy | json.load() + validate scene values |
| `_load_gps_tracks()` (inside `load_annotations`) | ⭐ Easy | json.load(), handle missing file gracefully |
| `get_class_names()` | ✅ Done | Already returns SCENE_CLASSES |
| `_build_sample()` | ⭐⭐ Medium | Assemble Sample from path + dicts |
| `get_metadata()` | ⭐⭐ Medium | Count clips per split, return DatasetMetadata |
| `_iter_frames_local()` | ⭐⭐ Medium | Walk `<root>/frames/<clip>/<frame:04d>.jpg` |
| `_iter_frames_for_split()` | ⭐⭐ Medium | Dispatch to correct iter method |
| `get_training_samples()` | ⭐ Easy | Call `_iter_frames_for_split("train")` |
| `get_validation_samples()` | ⭐ Easy | Call `_iter_frames_for_split("val")` |
| `get_test_samples()` | ⭐ Easy | Call `_iter_frames_for_split("test")` |
| `_iter_frames_tar()` | ⭐⭐⭐ Hard | Scan tar shards for matching clip_ids |
| `_iter_frames_hf_hub()` | ⭐⭐⭐ Hard | Stream shards from HF Hub |

**Start with the starred-easy items first** — they unlock the rest.

---

## 8. Architecture Constraints

| Constraint | Reason |
|------------|--------|
| `DatasetAdapter` base class must NOT be modified | Existing 55 tests verify the interface |
| `DatasetRegistry` must NOT be modified | `register("indian_road", IndianRoadAdapter)` is called in `indian_road.py` itself |
| `SceneCNN` and training code must NOT be modified | The CNN only sees `Sample.label` (int) and `Sample.image_path` — no dataset-specific code |
| Segmentation masks must NOT be used as labels | Pixel-to-class mapping is unconfirmed |
| `detection.json` confidence is NOT the scene label | It's a per-box quality score for object detection |

---

## 9. Files Involved

| File | Role |
|------|------|
| [`ml/datasets/adapters/indian_road.py`](../../ml/datasets/adapters/indian_road.py) | **The adapter — implement here** |
| [`ml/datasets/adapters/__init__.py`](../../ml/datasets/adapters/__init__.py) | Imports IndianRoadAdapter to auto-register it |
| [`ml/datasets/base.py`](../../ml/datasets/base.py) | `DatasetAdapter`, `Sample`, `DatasetMetadata` — do not modify |
| [`ml/datasets/config.py`](../../ml/datasets/config.py) | `DatasetConfig` — do not modify |
| [`ml/datasets/registry.py`](../../ml/datasets/registry.py) | `DatasetRegistry` — do not modify |
| [`backend/tests/test_indian_road_adapter.py`](../../backend/tests/test_indian_road_adapter.py) | Skeleton tests — extend as you implement |
| [`docs/datasets/indian-road-inspection.md`](indian-road-inspection.md) | Dataset structure reference |
| [`docs/datasets/indian-road-label-analysis.md`](indian-road-label-analysis.md) | Class counts and label analysis |

---

## 10. Test Status

```
78 passed in 15.89s
  ├─ test_cnn_and_datasets.py      36/36  ✅ (unchanged)
  ├─ test_health.py                 3/3   ✅ (unchanged)
  ├─ test_trips.py                  9/9   ✅ (unchanged)
  ├─ test_video_processing.py       7/7   ✅ (unchanged)
  └─ test_indian_road_adapter.py   23/23  ✅ (new skeleton tests)
```
