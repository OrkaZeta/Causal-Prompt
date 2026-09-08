"""Build the complete Exp-1 ActivityNet causal-prompt JSONL dataset.

The builder performs event-window selection, resolves the source video, decodes
the first frame of the selected five-second crop, captions that frame with
Qwen3-VL-8B-Instruct, and writes the complete training record directly to one
JSONL file. No separate initial-view dataset is used.
"""

import torch

import argparse
import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Sequence

from transformers import AutoModelForMultimodalLM, AutoProcessor

import av
from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm

VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CLIP_DURATION = 5.0
MAX_NEW_TOKENS = 128
MAX_IMAGE_SIDE = 672
CAPTION_RETRIES = 2
INITIAL_VIEW_PROMPT = (
    "Describe only what is visibly present in the image. Do not infer future "
    "actions, intentions, hidden objects, or events. Write one grammatical "
    "sentence of 25 to 50 words covering the scene/environment, visible "
    "subjects, visible objects, camera viewpoint, and visual style/lighting. "
    "Return only that sentence, without headings or labels, and end it with a "
    "period."
)
LOGGER = logging.getLogger(__name__)


@dataclass
class CausalPromptSample:
    sample_id: str
    split: str
    video_id: str
    video: str
    crop_start: float  # def 0
    init_decs: str  # from f_init caption
    events_idx: List[int]
    events_timestamps: List[List[float]]
    events_decs: List[str]
    raw_duration: float


class QwenInitialViewCaptioner:
    """Caption a crop's initial frame with Qwen3-VL-8B-Instruct."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"

    def caption(self, images: Sequence[Any]) -> List[str]:
        """Generate raw descriptions for already decoded and resized images."""
        messages = [[{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": INITIAL_VIEW_PROMPT},
        ]}] for image in images]
        prompts = [
            self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            for conversation in messages
        ]
        inputs = self.processor(
            text=prompts,
            images=list(images),
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        prompt_width = inputs["input_ids"].shape[1]
        generated = generated[:, prompt_width:]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @staticmethod
    def load_image(video_path: Path, start_sec: float) -> Any:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if stream.time_base is None:
                raise ValueError(f"Video stream has no time base: {video_path}")
            time_base = float(stream.time_base)
            stream_start = stream.start_time or 0
            target_pts = stream_start + math.ceil(start_sec / time_base)
            seek_pts = stream_start + max(0, math.floor((start_sec - 1) / time_base))
            container.seek(seek_pts, stream=stream, backward=True)

            image = None
            for frame in container.decode(stream):
                if frame.pts is not None and frame.pts >= target_pts:
                    image = frame.to_image().convert("RGB")
                    break

        if image is None:
            raise ValueError(f"Cannot decode frame at {start_sec:.2f}s: {video_path}")

        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

        return image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="friedrichor/ActivityNet_Captions")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-name", default=VLM_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def load_activity_net(hf_dataset: DatasetDict) -> list[CausalPromptSample]:
    samples = []

    for source, split in {"train": "train", "val1": "test"}.items():
        for raw in hf_dataset[source]:
            events = sorted(
                zip(raw["timestamps"], raw["sentences"]),
                key=lambda event: float(event[0][0]),
            )
            samples.append(CausalPromptSample(
                sample_id="",
                split=split,
                video_id=str(raw["video_id"]),
                video=str(raw["video"]),
                crop_start=0.0,
                init_decs="",
                events_idx=list(range(len(events))),
                events_timestamps=[list(map(float, timestamp)) for timestamp, _ in events],
                events_decs=[str(description) for _, description in events],
                raw_duration=float(raw["duration"]),
            ))

    return samples


def process_event_windows(sample: CausalPromptSample) -> list[CausalPromptSample]:
    timestamps = sample.events_timestamps
    valid_windows = []
    for i in range(len(timestamps)):
        base_start = (
            timestamps[i][0]
            if i == 0
            else min(timestamps[i - 1][1], timestamps[i][0])
        )
        for j in range(i + 1, len(timestamps)):
            crop_start = (
                max(0.0, timestamps[j][1] - CLIP_DURATION)
                if i == 0
                else base_start
            )
            crop_end = crop_start + CLIP_DURATION
            events_fit = all(
                timestamps[k][0] < crop_end and timestamps[k][1] > crop_start
                for k in range(i, j + 1)
            )
            if timestamps[j][1] - base_start <= CLIP_DURATION and events_fit:
                valid_windows.append((i, j, crop_start))

    maximal_windows = [
        (i, j, crop_start)
        for i, j, crop_start in valid_windows
        if not any(
            other_i <= i and j <= other_j and (other_i, other_j) != (i, j)
            for other_i, other_j, _ in valid_windows
        )
    ]

    samples = []
    for i, j, crop_start in maximal_windows:
        samples.append(replace(
            sample,
            sample_id=f"{sample.video_id}_E{i + 1}_E{j + 1}",
            crop_start=float(round(crop_start, 2)),
            events_idx=list(range(i, j + 1)),
            events_timestamps=[
                [
                    float(math.floor(max(0.0, timestamps[k][0] - crop_start) * 100 + 1e-9) / 100),
                    float(math.ceil(min(CLIP_DURATION, timestamps[k][1] - crop_start) * 100 - 1e-9) / 100),
                ]
                for k in range(i, j + 1)
            ],
            events_decs=sample.events_decs[i:j + 1],
        ))
    return samples


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER.info(
        f"Starting causal dataset build: splits=train/test "
        f"video_root={args.video_root} output_dir={args.output_dir} "
        f"batch_size={args.batch_size} model={args.model_name} device={args.device}"
    )

    """1. Load HF friedrichor/ActivityNet_Captions JSONL"""
    invalid = 0
    dataset = load_dataset(args.dataset)
    ds_samples = load_activity_net(dataset)

    LOGGER.info(f"Loaded {len(ds_samples)} ActivityNet samples")

    """2. Dataset filter"""
    ds_sample_filtered = [
        window
        for sample in ds_samples
        if (args.video_root / sample.video).is_file()
        for window in process_event_windows(sample)
    ]

    LOGGER.info(f"Dataset filter: {len(ds_sample_filtered)}/{len(ds_samples)} samples kept")

    """3. Caption initial frame for samples"""
    # use processed corp_start vs raw_duration to locate the target frame from GT videos, then caption it
    captioner = QwenInitialViewCaptioner(args.model_name, args.device)
    captioned_samples = []

    for batch_start in tqdm(
            range(0, len(ds_sample_filtered), args.batch_size), desc="Caption initial frames", unit="batch"
    ):
        batch_samples = ds_sample_filtered[batch_start:batch_start + args.batch_size]
        valid_samples, images = [], []

        # Stage 3.1: Decode initial frames
        for sample in batch_samples:
            try:
                video_path = args.video_root / sample.video
                image = captioner.load_image(video_path, sample.crop_start)

                valid_samples.append(sample)
                images.append(image)

            except Exception as e:
                invalid += 1
                LOGGER.warning(f"Skip {sample.video_id}: initial-frame decode failed: {e}")

        if not images:
            continue

        # Stage 3.2: Caption the decoded frames as one batch
        try:
            captions = captioner.caption(images)

            for sample, caption in zip(valid_samples, captions):
                caption = caption.strip()

                if not caption:
                    invalid += 1
                    LOGGER.warning(f"Skip {sample.video_id}: empty initial-frame caption")
                    continue

                sample.init_decs = caption
                captioned_samples.append(sample)

        except Exception as e:
            invalid += len(valid_samples)
            LOGGER.warning(f"Skip caption batch {batch_start // args.batch_size}: {e}")
    LOGGER.info(f"Initial-frame captioning: {len(captioned_samples)}/{len(ds_sample_filtered)} samples complete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for split in ("train", "test"):
        output = args.output_dir / f"activitynet_causal_5s_{split}.jsonl"
        split_samples = [sample for sample in captioned_samples if sample.split == split]
        with output.open("w", encoding="utf-8") as handle:
            for sample in split_samples:
                handle.write(json.dumps(sample.__dict__, ensure_ascii=False) + "\n")
        written += len(split_samples)
        LOGGER.info(f"Wrote {len(split_samples)} {split} records to {output}")

    LOGGER.info(f"Wrote {written} complete causal-prompt records, skipped {invalid} invalid samples")


if __name__ == "__main__":
    main()
