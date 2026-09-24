"""
IndianRoadAdapter — skeleton for the Indian Road Driving Dataset.

Dataset: thirdeyelabs/indian-road-dataset  (CC BY 4.0)
Source:  https://huggingface.co/datasets/thirdeyelabs/indian-road-dataset

Phase-5 CNN target: clip-level `scene` classification (6 classes).
This adapter translates the dataset's annotation files into Sample objects
that the FrameDataset / Trainer / Evaluator can consume without change.

--------------------------------------------------------------------------
DESIGN SUMMARY
--------------------------------------------------------------------------

Inputs (all under config.root or sourced from HF Hub):
  annotations/scene_attributes.json  — clip-level {weather, scene, timeofday}
  annotations/detection.json         — frame-level BDD100K detection labels
  data/train-NNNNN-of-00646.tar      — WebDataset shards (images + masks)
  gps/gps_tracks.json                — optional per-clip GPS waypoints

Outputs:
  Sample(image_path=<Path>, label=<int 0-5>, metadata={...})
  label = index into SCENE_CLASSES (alphabetical, confirmed from real data)

Splitting:
  No official train/val/test split exists in the dataset.
  The adapter performs a deterministic 80/10/10 split keyed on clip UUID:
    split = hash(clip_uuid) % 100
    train  : hash % 100 <  80
    val    : hash % 100 >= 80 and < 90
    test   : hash % 100 >= 90
  All frames of one clip land in the same split (no temporal leakage).

Data access modes (controlled by config.extra["source"]):
  "local"   — read pre-extracted JPEG frames from config.root/frames/<clip>/<frame>.jpg
  "hf_hub"  — stream tar shards from HuggingFace Hub (requires HF_TOKEN env var)
  "tar"     — read locally cached tar shard files without extracting them

--------------------------------------------------------------------------
WHAT YOU MUST IMPLEMENT (marked TODO below)
--------------------------------------------------------------------------
  1. _load_scene_attributes()   — parse scene_attributes.json
  2. _load_gps_tracks()         — parse gps_tracks.json (optional)
  3. _assign_splits()           — hash clip UUIDs into train/val/test buckets
  4. _iter_frames_for_split()   — yield (image_path, clip_id, frame_idx) tuples
                                  for a given split — this is where local vs HF
                                  access logic differs
  5. _build_sample()            — assemble a Sample from a frame + metadata
  6. get_metadata()             — return DatasetMetadata
  7. get_class_names()          — return SCENE_CLASSES list
  8. get_training_samples()     — call _iter_frames_for_split("train")
  9. get_validation_samples()   — call _iter_frames_for_split("val")
  10. get_test_samples()        — call _iter_frames_for_split("test")
"""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Iterator, Optional
import tarfile
import tempfile
from huggingface_hub import hf_hub_download

from ..base import DatasetAdapter, DatasetMetadata, DatasetSplit, Sample
from ..config import DatasetConfig
from ..registry import DatasetRegistry

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# CLASS LIST — confirmed from full dataset scan (645,714 frames)
# Ordered alphabetically. Integer label = list index.
# --------------------------------------------------------------------------
SCENE_CLASSES: list[str] = [
    "city street",      # label 0  — 2,237 clips (26.5%)
    "highway",          # label 1  — 1,277 clips (15.1%)
    "parking lot",      # label 2  —   339 clips  (4.0%)
    "residential road", # label 3  — 2,996 clips (35.5%)
    "tunnel",           # label 4  —   905 clips (10.7%)
    "village road",     # label 5  —   680 clips  (8.1%)
]

# --------------------------------------------------------------------------
# SPLIT RATIOS — deterministic, hash-based (no official split exists)
# --------------------------------------------------------------------------
SPLIT_TRAIN_THRESHOLD = 80   # hash % 100 < 80  → train
SPLIT_VAL_THRESHOLD   = 90   # hash % 100 < 90  → val   (else test)

# --------------------------------------------------------------------------
# CONFIG EXTRA KEYS — read from DatasetConfig.extra
# --------------------------------------------------------------------------
EXTRA_SOURCE        = "source"        # "local" | "hf_hub" | "tar"
EXTRA_HF_REPO       = "hf_repo"       # HuggingFace dataset repo id
EXTRA_MIN_CONF      = "min_confidence" # float — filter detections below threshold
EXTRA_FRAMES_SUBDIR = "frames_subdir"  # subdir under root for local extracted frames
EXTRA_SHARDS_SUBDIR = "shards_subdir"  # subdir under root for local .tar shards

DEFAULT_SOURCE   = "local"
DEFAULT_HF_REPO  = "thirdeyelabs/indian-road-dataset"
DEFAULT_MIN_CONF = 0.5
DEFAULT_FRAMES_SUBDIR = "frames"
DEFAULT_SHARDS_SUBDIR = "data"


class IndianRoadAdapter(DatasetAdapter):
    """
    DatasetAdapter for the Indian Road Driving Dataset.

    Target task: scene classification (6 classes).

    Scene labels are clip-level. This adapter propagates each clip's
    scene label to all frames belonging to that clip.

    Configuration (via DatasetConfig):
        config.root                    — dataset root directory (for "local"/"tar" mode)
        config.annot_dir               — directory containing annotation JSON files
        config.extra["source"]         — "local" | "hf_hub" | "tar"  (default: "local")
        config.extra["hf_repo"]        — HuggingFace repo id           (default: see above)
        config.extra["min_confidence"] — detection confidence threshold (default: 0.5)
        config.extra["frames_subdir"]  — subdir for extracted frames    (default: "frames")
        config.extra["shards_subdir"]  — subdir for local tar shards   (default: "data")

    Usage::

        config = DatasetConfig(
            name="indian-road",
            root=Path("/data/indian-road"),
            annot_dir="annotations",
            adapter="indian_road",
            extra={"source": "local"},
        )
        adapter = IndianRoadAdapter(config)
        adapter.load_annotations()

        for sample in adapter.get_training_samples():
            # sample.label     -> int 0-5 (scene class index)
            # sample.metadata  -> dict with clip_id, frame_idx, detections, gps, …
            ...
    """

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

        # ── Internal state populated by load_annotations() ──────────────────
        #
        # _scene_attrs:   { clip_uuid: {"weather": str, "scene": str, "timeofday": str} }
        # _gps_tracks:    { clip_uuid: [ {ts, lat, lon, speed_kmh, course_deg}, … ] }
        # _split_map:     { clip_uuid: "train" | "val" | "test" }
        # _class_index:   { scene_string: int }  — reverse lookup of SCENE_CLASSES
        # _annotations_loaded: bool guard

        self._scene_attrs:        dict[str, dict[str, str]] = {}
        self._gps_tracks:         dict[str, list[dict]]     = {}
        self._split_map:          dict[str, str]             = {}
        self._class_index:        dict[str, int]             = {
            name: idx for idx, name in enumerate(SCENE_CLASSES)
        }
        self._annotations_loaded: bool = False

        # ── Config helpers ───────────────────────────────────────────────────
        extra = config.extra or {}
        self._source:         str   = extra.get(EXTRA_SOURCE,        DEFAULT_SOURCE)
        self._hf_repo:        str   = extra.get(EXTRA_HF_REPO,       DEFAULT_HF_REPO)
        self._min_confidence: float = float(extra.get(EXTRA_MIN_CONF, DEFAULT_MIN_CONF))
        self._frames_subdir:  str   = extra.get(EXTRA_FRAMES_SUBDIR, DEFAULT_FRAMES_SUBDIR)
        self._shards_subdir:  str   = extra.get(EXTRA_SHARDS_SUBDIR, DEFAULT_SHARDS_SUBDIR)

    def load_annotations(self) -> None:

        if self._annotations_loaded:
            raise RuntimeError(
                "load_annotations() has already been called. "
                "Create a new adapter instance if you need to reload."
            )


        scene_path = self.config.annot_path / "scene_attributes.json"

        if not scene_path.exists():
            raise FileNotFoundError(
                f"scene_attributes.json not found: {scene_path}"
            )

        with open(scene_path, encoding="utf-8") as f:
            self._scene_attrs = json.load(f)

        for clip_id, attrs in self._scene_attrs.items():
            scene = attrs.get("scene")

            if scene not in SCENE_CLASSES:
                logger.warning(
                    "Unrecognised scene '%s' for clip %s",
                    scene,
                    clip_id,
                )


        gps_path = self.config.annot_path / "../gps/gps_tracks.json"

        if gps_path.exists():
            with open(gps_path, encoding="utf-8") as f:
                self._gps_tracks = json.load(f)
        else:
            logger.warning(
                "GPS tracks file not found — "
                "GPS metadata will be None for all samples"
            )

        self._split_map = self._assign_splits(
            list(self._scene_attrs.keys())
        )

        self._annotations_loaded = True


    def get_class_names(self) -> list[str]:
        """Return the 6 scene class names in label-index order."""
        return list(SCENE_CLASSES)

    def get_metadata(self) -> DatasetMetadata:

        self._ensure_loaded()

        split_counts = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        for clip_uuid, split in self._split_map.items():
            split_counts[split] += 1
       
        return DatasetMetadata(
            name=self.config.name or "Indian Road Driving Dataset",
            version="2026",
            num_classes=len(SCENE_CLASSES),
            class_names=SCENE_CLASSES,
            total_samples=sum(split_counts.values()),
            split_counts=split_counts,
            description=(
                "Clip-level scene classification. "
                "6 classes. Delhi NCR dashcam footage."
            ),  
        )

    def get_training_samples(self) -> Iterator[Sample]:
        self._ensure_loaded()
        return self._iter_frames_for_split(DatasetSplit.TRAIN)

    def get_validation_samples(self) -> Iterator[Sample]:
        self._ensure_loaded()
        return self._iter_frames_for_split(DatasetSplit.VAL)

    def get_test_samples(self) -> Iterator[Sample]:
        self._ensure_loaded()
        return self._iter_frames_for_split(DatasetSplit.TEST)

    def _ensure_loaded(self) -> None:
        if not self._annotations_loaded:
            raise RuntimeError(
                "load_annotations() must be called before accessing samples or metadata. "
                "Example: adapter.load_annotations()"
            )

    def _assign_splits(self, clip_uuids: list[str]) -> dict[str, str]:
        split_map = {}
        for clip_uuid in clip_uuids:
            
            hash_value = hashlib.md5(clip_uuid.encode()).hexdigest()
            bucket = int(hash_value,16) % 100

            if bucket < 80:
                split = "train"
            elif bucket < 90:
                split = "val"
            else:
                split = "test"

            split_map[clip_uuid] = split
        
        return split_map

    def _iter_frames_for_split(self, split: str) -> Iterator[Sample]:
        clip_ids = []

        for clip_id, clip_split in self._split_map.items():
            if clip_split == split:
                clip_ids.append(clip_id)

        if self._source == "local":
            frame_iterator = self._iter_frames_local(clip_ids)
        elif self._source == "tar":
            frame_iterator = self._iter_frames_tar(clip_ids)
        elif self._source == "hf_hub":
            frame_iterator = self._iter_frames_hf_hub(clip_ids)
        else:
            raise ValueError(f"Unsupported dataset source: {self._source}")

        for image_path, clip_id, frame_idx in frame_iterator:
            sample = self._build_sample(
                image_path=image_path,
                clip_id=clip_id,
                frame_idx=frame_idx,
            )
            yield sample

    def _build_sample(
        self,
        image_path: Path,
        clip_id: str,
        frame_idx: int,
        detections: Optional[list[dict]] = None,
    ) -> Sample:
    
        scene_info = self._scene_attrs[clip_id]

        scene = scene_info["scene"]
        weather = scene_info["weather"]
        timeofday = scene_info["timeofday"]

        label = self._class_index[scene]

        gps_points = self._gps_tracks.get(clip_id, [])

        lat = None
        lon = None
        speed_kmh = None

        if gps_points:
            gps = gps_points[0]
            lat = gps.get("lat")
            lon = gps.get("lon")
            speed_kmh = gps.get("speed_kmh")

        metadata = {
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "scene": scene,
            "weather": weather,
            "timeofday": timeofday,
            "detections": detections or [],
            "lat": lat,
            "lon": lon,
            "speed_kmh": speed_kmh,
        }

        return Sample(
            image_path=image_path,
            label=label,
            metadata=metadata,
        )

    def _iter_frames_local(self, clip_ids: list[str]) -> Iterator[tuple[Path, str, int]]:
        for clip_id in clip_ids:
            clip_dir = (
                self.config.data_root
                / self._frames_subdir
                / clip_id
            )

            if not clip_dir.exists():
                continue

            frame_files = sorted(clip_dir.glob("*.jpg"))

            for image_path in frame_files:
                frame_idx = int(image_path.stem)
                yield image_path, clip_id, frame_idx

    def _iter_frames_tar(self, clip_ids: list[str]) -> Iterator[tuple[Path, str, int]]:
        requested_clips = set(clip_ids)
        shards_dir = self.config.data_root / self._shards_subdir
        shard_files = sorted(shards_dir.glob("*.tar"))

        for shard_path in shard_files:
            with tarfile.open(shard_path, "r") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".jpg"):
                        continue

                    name_parts = Path(member.name).stem.split("_")

                    if len(name_parts) != 2:
                        continue

                    clip_id = name_parts[0]

                    if clip_id not in requested_clips:
                        continue

                    try:
                        frame_idx = int(name_parts[1])
                    except ValueError:
                        continue

                    file_obj = tar.extractfile(member)

                    if file_obj is None:
                        continue

                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=".jpg",
                        delete=False,
                    )

                    with file_obj:
                        temp_file.write(file_obj.read())

                    temp_file.close()

                    yield Path(temp_file.name), clip_id, frame_idx

    def _iter_frames_hf_hub(self, clip_ids: list[str]) -> Iterator[tuple[Path, str, int]]:
        requested_clips = set(clip_ids)

        for shard_idx in range(646):
            shard_name = f"train-{shard_idx:05d}-of-00646.tar"

            shard_path = hf_hub_download(
                repo_id=self._hf_repo,
                filename=f"data/{shard_name}",
                repo_type="dataset",
            )

            with tarfile.open(shard_path, "r|") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".jpg"):
                        continue

                    name_parts = Path(member.name).stem.split("_")

                    if len(name_parts) != 2:
                        continue

                    clip_id = name_parts[0]

                    if clip_id not in requested_clips:
                        continue

                    try:
                        frame_idx = int(name_parts[1])
                    except ValueError:
                        continue

                    file_obj = tar.extractfile(member)

                    if file_obj is None:
                        continue

                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=".jpg",
                        delete=False,
                    )

                    with file_obj:
                        temp_file.write(file_obj.read())

                    temp_file.close()

                    yield Path(temp_file.name), clip_id, frame_idx

DatasetRegistry.register("indian_road", IndianRoadAdapter)
