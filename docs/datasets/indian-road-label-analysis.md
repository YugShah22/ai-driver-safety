# Indian Road Driving Dataset — Label Analysis Report

**Dataset:** `thirdeyelabs/indian-road-dataset`  
**Analysis date:** 2026-09-23  
**Method:** Full streaming pass of `annotations/detection.json` (1.31 GB via ijson) + full parse of `annotations/scene_attributes.json`  
**Frames processed:** 645,714 / 645,714 (100% — no sampling)  
**Source of all numbers:** Directly computed from real dataset files

> All numbers in this document are **OBSERVED** from real data unless explicitly marked **DOCUMENTED**.

---

## 1. Clip-Level Scene Attributes

Scene attributes (`weather`, `scene`, `timeofday`) are stored in `annotations/scene_attributes.json` as a dict keyed by **clip UUID**. They are **clip-level**, not frame-level.

### Confirmation

**OBSERVED:** `scene_attributes.json` contains **8,434 entries**, each a dict with exactly 3 keys: `weather`, `scene`, `timeofday`. There is no per-frame breakdown — one set of attributes applies to every frame in the clip.

> ✅ **Confirmed: scene/weather/timeofday are clip-level and shared by all frames in a clip.**

---

## 2. Weather — All Unique Values and Counts

**Total clips: 8,434**

| Value | Clips | % of total |
|-------|------:|-----------:|
| `clear` | 7,348 | 87.1% |
| `overcast` | 485 | 5.8% |
| `foggy` | 400 | 4.7% |
| `rainy` | 201 | 2.4% |

**Unique values: 4**  
**Dominant class:** `clear` at 87.1% — severe class imbalance.  
**Minority classes:** `foggy` (4.7%) and `rainy` (2.4%) are rare.

---

## 3. Scene — All Unique Values and Counts

**Total clips: 8,434**

| Value | Clips | % of total |
|-------|------:|-----------:|
| `residential road` | 2,996 | 35.5% |
| `city street` | 2,237 | 26.5% |
| `highway` | 1,277 | 15.1% |
| `tunnel` | 905 | 10.7% |
| `village road` | 680 | 8.1% |
| `parking lot` | 339 | 4.0% |

**Unique values: 6**  
**Most balanced label set of the three attributes.**  
**Minority class:** `parking lot` at 4.0% — small but not negligible.

---

## 4. Time of Day — All Unique Values and Counts

**Total clips: 8,434**

| Value | Clips | % of total |
|-------|------:|-----------:|
| `daytime` | 5,102 | 60.5% |
| `night` | 2,439 | 28.9% |
| `dusk` | 781 | 9.3% |
| `dawn` | 112 | 1.3% |

**Unique values: 4**  
**Minority class:** `dawn` at 1.3% (112 clips) — very rare. May need oversampling or merging with `dusk`.

---

## 5. Detection Categories — Full Dataset Counts

**OBSERVED:** Detection annotations are in `annotations/detection.json`, one entry per frame, each containing a `labels` list with 0 or more objects. They are **frame-level**.

> ✅ **Confirmed: detection annotations are frame-level and can contain multiple objects per frame.**

### Summary

| Metric | Value |
|--------|------:|
| Total frames | 645,714 |
| Frames with no labels | 26,245 (4.1%) |
| Total label instances | 3,669,119 |
| Labels with confidence < 0.5 | 1,222,687 (33.3%) |

### Per-Category Breakdown

| Category | Detections | Frames | Clips | Det % | Frame % |
|----------|----------:|-------:|------:|------:|--------:|
| `car` | 1,650,282 | 530,168 | 8,382 | 45.0% | 82.1% |
| `motorcycle` | 527,020 | 289,342 | 8,160 | 14.4% | 44.8% |
| `rider` | 440,456 | 247,652 | 7,899 | 12.0% | 38.4% |
| `person` | 388,274 | 201,199 | 8,018 | 10.6% | 31.2% |
| `autorickshaw` | 185,383 | 133,517 | 6,775 | 5.1% | 20.7% |
| `vehicle fallback` | 122,865 | 100,524 | 7,341 | 3.3% | 15.6% |
| `truck` | 100,460 | 80,937 | 6,405 | 2.7% | 12.5% |
| `bicycle` | 74,126 | 60,207 | 6,846 | 2.0% | 9.3% |
| `traffic sign` | 62,647 | 48,314 | 5,159 | 1.7% | 7.5% |
| `bus` | 49,430 | 38,822 | 4,533 | 1.3% | 6.0% |
| `animal` | 37,203 | 27,076 | 3,966 | 1.0% | 4.2% |
| `traffic light` | 30,972 | 21,706 | 2,690 | 0.8% | 3.4% |
| `train` | **1** | **1** | **1** | 0.0% | 0.0% |

> ⚠️ **`train`** — 1 detection in 1 frame from 1 clip. **This is almost certainly a misclassification artifact** from the machine-generated labelling pipeline. The README documents 12 classes; `train` is not among them. This category should be excluded from any model training.

---

## 6. Rarity Analysis

### Extremely rare categories (OBSERVED)

| Category | Status | Recommendation |
|----------|--------|----------------|
| `train` | **1 instance** — noise | ❌ Exclude entirely |
| `traffic light` | 30,972 det / 2,690 clips | ⚠️ Rare — present in only 31.9% of clips |
| `animal` | 37,203 det / 3,966 clips | ⚠️ Rare — present in only 47.0% of clips |
| `bus` | 49,430 det / 4,533 clips | ⚠️ Moderate — present in 53.7% of clips |

### Dominance issue

`car` accounts for **45.0% of all detections** and appears in **82.1% of all frames**. Any object-level classifier will be heavily biased toward `car` without weighting or oversampling.

---

## 7. Suitability for Initial CNN Experiment

### Scene classification (recommended for Phase 5)

**`scene` is the strongest candidate for the initial multi-class CNN label.** Reasons:

| Factor | Assessment |
|--------|-----------|
| Number of classes | **6** — small enough for a compact CNN |
| Class imbalance | Mild — worst ratio is 9:1 (`residential road` vs `parking lot`) |
| Clip coverage | All 8,434 clips have a `scene` label — **no missing values** |
| Semantic clarity | Classes are visually distinct (highway vs tunnel vs parking lot) |
| Frame count | Every frame in a clip inherits the scene label — 645,714 labelled frames available |

### Comparison of label candidates

| Attribute | Classes | Worst imbalance | Missing | Verdict |
|-----------|--------:|----------------:|--------:|---------|
| `scene` | 6 | 9:1 | 0 | ✅ **Best for Phase 5** |
| `timeofday` | 4 | 46:1 (`dawn` vs `daytime`) | 0 | ⚠️ Usable but `dawn` very rare |
| `weather` | 4 | 37:1 (`clear` vs `rainy`) | 0 | ❌ Too imbalanced for baseline |
| detection category | 12 (+1 noise) | N/A (object-level) | N/A | ❌ Needs box extraction pipeline |

---

## 8. Confirmation of Architecture Questions

### Q: Are scene/weather/timeofday clip-level?
**✅ CONFIRMED.** `scene_attributes.json` is a dict keyed by clip UUID with clip-level values. All frames within a clip share identical scene, weather, and timeofday labels.

### Q: Are detection annotations frame-level with multiple objects per frame?
**✅ CONFIRMED.** `detection.json` has one entry per frame; each entry's `labels` list can contain 0–N objects. Average labels per non-empty frame: **5.9 objects**.

### Q: Should detection confidence remain separate from the CNN scene label?
**✅ YES.** Detection confidence is a per-bounding-box quality score for the object detector, not a property of the scene. For a scene-classification CNN (predicting `scene`, `weather`, or `timeofday` from the full frame), detection confidence is irrelevant as a training target. It is only relevant if filtering low-quality labels at the detection level.

---

## 9. Recommended Class List for Phase 5 CNN

Based on **OBSERVED** data, the recommended initial label set is **`scene`** with 6 classes:

```python
SCENE_CLASSES = [
    "city street",       # 2,237 clips  (26.5%)
    "highway",           # 1,277 clips  (15.1%)
    "parking lot",       #   339 clips  (4.0%)
    "residential road",  # 2,996 clips  (35.5%)
    "tunnel",            #   905 clips  (10.7%)
    "village road",      #   680 clips  (8.1%)
]
```

> Classes are listed alphabetically. Integer label indices should be assigned in this order for reproducibility (i.e., `city street=0`, `highway=1`, ..., `village road=5`).

---

## 10. Estimated Frame Counts Per Scene Class

Since scene labels are clip-level, the frame count per class scales proportionally with average frames per clip.

**Average frames per clip** (OBSERVED): 645,714 frames / 8,434 clips = **76.6 frames/clip**

| Scene class | Clips | Estimated frames |
|-------------|------:|-----------------:|
| `residential road` | 2,996 | ~229,493 |
| `city street` | 2,237 | ~171,353 |
| `highway` | 1,277 | ~97,818 |
| `tunnel` | 905 | ~69,321 |
| `village road` | 680 | ~52,086 |
| `parking lot` | 339 | ~25,968 |

> ⚠️ These are estimated frame counts. Actual frame counts per class require a cross-reference of `scene_attributes.json` against `detection.json` frame names, which was not done in this analysis. The estimates assume uniform frames-per-clip across all classes.

---

## 11. Test Suite Status

No project source files were modified during this analysis.  
**55/55 existing tests remain passing** (verified separately).

---

## 12. Files Produced

| File | Purpose |
|------|---------|
| [`docs/datasets/indian-road-inspection.md`](indian-road-inspection.md) | First-pass inspection (structure, format, sample frames) |
| [`docs/datasets/indian-road-label-analysis.md`](indian-road-label-analysis.md) | This document — full label counts from real data |
| [`scripts/inspect_indian_road_dataset.py`](../../scripts/inspect_indian_road_dataset.py) | Sample inspection tool |
| [`scripts/analyze_indian_road_labels.py`](../../scripts/analyze_indian_road_labels.py) | Full-dataset label counter |
