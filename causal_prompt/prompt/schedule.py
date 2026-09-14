import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

SCHEDULES = (
    "full",
    "full_tagged",
    "causal_next",
    "causal",
    "current_future",
    "current_next",
    "current",
)
CAUSAL_PROMPT_FIELDS = {
    "sample_id", "split", "crop_start", "init_decs", "events_idx",
    "events_timestamps", "events_decs",
}


@dataclass(frozen=True)
class TimedEvent:
    event_id: int
    description: str
    start_sec: float
    end_sec: float
    start_pixel: int
    end_pixel: int

    def overlaps(self, start_pixel: int, end_pixel: int) -> bool:
        return self.start_pixel < end_pixel and self.end_pixel > start_pixel


@dataclass
class ScheduledPrompt:
    prompt_id: str
    prompt: str
    source: dict[str, Any]
    mode: str
    fps: float
    duration_sec: float
    pixel_frame_count: int
    latent_frame_count: int
    chunk_size: int
    events: tuple[TimedEvent, ...]
    pixel_prompts: list[str]
    frame_prompts: list[str]
    chunk_prompts: list[str]


def _global_prompt(record: dict[str, Any]) -> str:
    prompt_id = record.get("sample_id", "<unknown>")
    initial_view = record.get("init_decs")
    if isinstance(initial_view, str) and initial_view.strip():
        return initial_view.strip()
    raise ValueError(f"Record {prompt_id!r} has no non-empty 'init_decs' string.")


def _timed_events(
        record: dict[str, Any], fps: float, duration_sec: float, pixel_frame_count: int
) -> tuple[TimedEvent, ...]:
    prompt_id = record.get("sample_id", "<unknown>")
    indices = record.get("events_idx")
    timestamps = record.get("events_timestamps")
    descriptions = record.get("events_decs")
    if not all(isinstance(values, list) for values in (indices, timestamps, descriptions)):
        raise ValueError(f"Record {prompt_id!r} has no events.")
    if not indices or not len(indices) == len(timestamps) == len(descriptions):
        raise ValueError(f"Record {prompt_id!r} has inconsistent event arrays.")

    events = []
    previous_start = -1.0
    for event_index, (event_id, timestamp, description) in enumerate(
        zip(indices, timestamps, descriptions)
    ):
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"Record {prompt_id!r} event {event_index} has no description."
            )
        if not isinstance(timestamp, list) or len(timestamp) != 2:
            raise ValueError(f"Record {prompt_id!r} event {event_index} needs [start, end].")
        start_sec, end_sec = map(float, timestamp)
        if not 0 <= start_sec < end_sec <= duration_sec:
            raise ValueError(
                f"Record {prompt_id!r} event {event_index} has invalid time range "
                f"[{start_sec}, {end_sec}] for duration {duration_sec}."
            )
        if start_sec < previous_start:
            raise ValueError(f"Record {prompt_id!r} events are not time ordered.")
        previous_start = start_sec

        start_pixel = min(pixel_frame_count - 1, max(0, round(start_sec * fps)))
        if math.isclose(end_sec, duration_sec):
            end_pixel = pixel_frame_count
        else:
            end_pixel = min(pixel_frame_count, max(start_pixel + 1, round(end_sec * fps)))
        events.append(
            TimedEvent(
                event_id=int(event_id),
                description=description.strip(),
                start_sec=start_sec,
                end_sec=end_sec,
                start_pixel=start_pixel,
                end_pixel=end_pixel,
            )
        )
    return tuple(events)


def _selected_event_indices(
        events: tuple[TimedEvent, ...], mode: str, start_pixel: int, end_pixel: int
) -> tuple[int, ...]:
    active = tuple(
        index for index, event in enumerate(events) if event.overlaps(start_pixel, end_pixel)
    )
    past_or_active = tuple(
        index for index, event in enumerate(events) if event.start_pixel < end_pixel
    )
    active_or_future = tuple(
        index for index, event in enumerate(events) if event.end_pixel > start_pixel
    )
    future = tuple(
        index for index, event in enumerate(events) if event.start_pixel >= end_pixel
    )

    if mode == "full":
        return tuple(range(len(events)))
    if mode == "full_tagged":
        return tuple(range(len(events)))
    if mode == "causal":
        return past_or_active
    if mode == "causal_next":
        return tuple(dict.fromkeys((*past_or_active, *future[:1])))
    if mode == "current_future":
        return active_or_future
    if mode == "current_next":
        return tuple(dict.fromkeys((*active, *future[:1])))
    if mode == "current":
        return active
    raise ValueError(f"Unknown schedule mode: {mode}")


def _tagged_description(
        event_index: int,
        events: tuple[TimedEvent, ...],
        start_pixel: int,
        end_pixel: int,
) -> str:
    event = events[event_index]
    if event.overlaps(start_pixel, end_pixel):
        tag = "current"
    elif event.end_pixel <= start_pixel:
        tag = "past"
    else:
        future_indices = [
            index
            for index, candidate in enumerate(events)
            if candidate.start_pixel >= end_pixel
        ]
        tag = "next" if future_indices and event_index == future_indices[0] else "future"
    return f"[{tag}] {event.description}"


def _prompt_for_interval(
        global_prompt: str,
        events: tuple[TimedEvent, ...],
        mode: str,
        start_pixel: int,
        end_pixel: int,
        is_first: bool,
) -> str:
    # The first latent represents the observed initial view in these two modes;
    # revealing an event that happens to overlap t=0 would violate the Exp-1
    # G-only contract for F_0.
    if is_first and mode in {"causal", "current"}:
        return global_prompt
    parts = [global_prompt] if is_first else []
    for event_index in _selected_event_indices(events, mode, start_pixel, end_pixel):
        if mode == "full_tagged":
            parts.append(
                _tagged_description(event_index, events, start_pixel, end_pixel)
            )
        else:
            parts.append(events[event_index].description)
    return " ".join(parts)


def _latent_pixel_spans(
        pixel_frame_count: int, temporal_downsample: int
) -> tuple[tuple[int, int], ...]:
    latent_frame_count = math.ceil((pixel_frame_count - 1) / temporal_downsample) + 1
    spans = [(0, 1)]
    for latent_index in range(1, latent_frame_count):
        start = 1 + (latent_index - 1) * temporal_downsample
        end = min(pixel_frame_count, 1 + latent_index * temporal_downsample)
        spans.append((start, end))
    return tuple(spans)


def schedule_record(
        record: dict[str, Any],
        mode: str = "full",
        fps: float = 16.0,
        temporal_downsample: int = 4,
        chunk_size: int = 1,
) -> ScheduledPrompt:
    missing = CAUSAL_PROMPT_FIELDS - record.keys()
    if missing:
        raise ValueError(f"Record is missing fields: {sorted(missing)}")
    prompt_id = record.get("sample_id")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError("Record has no non-empty string 'sample_id'.")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", prompt_id) is None:
        raise ValueError(f"Record id is not filename-safe: {prompt_id!r}.")
    if mode not in SCHEDULES:
        raise ValueError(f"Unknown schedule mode: {mode}")
    if fps <= 0 or temporal_downsample <= 0 or chunk_size <= 0:
        raise ValueError("fps, temporal_downsample, and chunk_size must be positive.")

    duration_sec = 5.0
    pixel_frame_count = round(duration_sec * fps) + 1
    global_prompt = _global_prompt(record)
    events = _timed_events(record, fps, duration_sec, pixel_frame_count)
    # ``prompt`` is the unchanging full-video fallback. Dynamic schedules,
    # especially full_tagged, are represented by frame_prompts/chunk_prompts.
    full_video_prompt = " ".join(
        [global_prompt, *(event.description for event in events)]
    )

    pixel_prompts = [
        _prompt_for_interval(
            global_prompt, events, mode, pixel_index, pixel_index + 1, pixel_index == 0
        )
        for pixel_index in range(pixel_frame_count)
    ]
    latent_spans = _latent_pixel_spans(pixel_frame_count, temporal_downsample)
    frame_prompts = [
        _prompt_for_interval(
            global_prompt, events, mode, start, end, latent_index == 0
        )
        for latent_index, (start, end) in enumerate(latent_spans)
    ]

    chunk_prompts = []
    for chunk_start in range(0, len(latent_spans), chunk_size):
        chunk_end = min(len(latent_spans), chunk_start + chunk_size)
        start_pixel = latent_spans[chunk_start][0]
        end_pixel = latent_spans[chunk_end - 1][1]
        chunk_prompts.append(
            _prompt_for_interval(
                global_prompt,
                events,
                mode,
                start_pixel,
                end_pixel,
                chunk_start == 0,
            )
        )

    return ScheduledPrompt(
        prompt_id=prompt_id,
        prompt=full_video_prompt,
        source=record,
        mode=mode,
        fps=fps,
        duration_sec=duration_sec,
        pixel_frame_count=pixel_frame_count,
        latent_frame_count=len(latent_spans),
        chunk_size=chunk_size,
        events=events,
        pixel_prompts=pixel_prompts,
        frame_prompts=frame_prompts,
        chunk_prompts=chunk_prompts,
    )


class JSONLPromptDataset(IterableDataset[ScheduledPrompt]):
    """Stream and schedule JSONL rows on demand without retaining the dataset."""

    def __init__(
            self,
            path: str | Path,
            mode: str = "full",
            fps: float = 16.0,
            temporal_downsample: int = 4,
            chunk_size: int = 1,
            split: str | None = None,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.mode = mode
        self.fps = fps
        self.temporal_downsample = temporal_downsample
        self.chunk_size = chunk_size
        self.split = split
        if self.mode not in SCHEDULES:
            raise ValueError(f"Unknown schedule mode: {self.mode}")
        if not self.path.is_file():
            raise FileNotFoundError(f"Prompt JSONL does not exist: {self.path}")

    def __iter__(self) -> Iterator[ScheduledPrompt]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        with self.path.open(encoding="utf-8") as handle:
            record_index = 0
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                assigned = record_index % worker_count == worker_id
                record_index += 1
                if not assigned:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {self.path}:{line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected a JSON object at {self.path}:{line_number}."
                    )
                if self.split is not None and record.get("split") != self.split:
                    continue
                try:
                    yield schedule_record(
                        record,
                        mode=self.mode,
                        fps=self.fps,
                        temporal_downsample=self.temporal_downsample,
                        chunk_size=self.chunk_size,
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Failed to schedule {self.path}:{line_number}: {error}"
                    ) from error


def load_scheduled_prompts(
        path: str | Path,
        mode: str = "full",
        fps: float = 16.0,
        temporal_downsample: int = 4,
    chunk_size: int = 1,
    split: str | None = None,
) -> JSONLPromptDataset:
    return JSONLPromptDataset(
        path,
        mode=mode,
        fps=fps,
        temporal_downsample=temporal_downsample,
        chunk_size=chunk_size,
        split=split,
    )


class CausalPromptTrainingDataset(Dataset[dict[str, Any]]):
    """Random-access view of one split, scheduled once per latent block."""

    def __init__(self, path: str | Path, mode: str, *, split: str = "train",
                 fps: float = 16.0, temporal_downsample: int = 4,
                 chunk_size: int = 1, expected_latent_frames: int | None = 21,
                 sample_id: str | None = None) -> None:
        self.path, self.mode, self.split = Path(path), mode, split
        self.fps, self.temporal_downsample = fps, temporal_downsample
        self.chunk_size, self.expected_latent_frames = chunk_size, expected_latent_frames
        if self.path.suffix.lower() != ".jsonl" or not self.path.is_file():
            raise FileNotFoundError(f"Causal-prompt JSONL does not exist: {self.path}")
        self._records = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("split") == split and (
                    sample_id is None or record.get("sample_id") == sample_id
                ):
                    missing = CAUSAL_PROMPT_FIELDS - record.keys()
                    if missing:
                        raise ValueError(f"Missing fields at {self.path}:{line_number}: {sorted(missing)}")
                    self._records.append(record)
        if not self._records:
            raise ValueError(
                f"No split={split!r} records for sample_id={sample_id!r} in {self.path}"
            )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = schedule_record(self._records[index], self.mode, self.fps,
                               self.temporal_downsample, self.chunk_size)
        if self.expected_latent_frames is not None and item.latent_frame_count != self.expected_latent_frames:
            raise ValueError(f"Sample {item.prompt_id!r} has {item.latent_frame_count} latent frames; expected {self.expected_latent_frames}.")
        return {"idx": index, "prompt_ids": item.prompt_id, "prompts": item.prompt,
                "block_prompts": item.chunk_prompts}


def collate_causal_prompt_batch(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty causal-prompt batch.")
    block_count = len(samples[0]["block_prompts"])
    if not block_count or any(len(sample["block_prompts"]) != block_count for sample in samples):
        raise ValueError("Every sample must contain the same non-zero block count.")
    return {"idx": torch.tensor([sample["idx"] for sample in samples]),
            "prompt_ids": [sample["prompt_ids"] for sample in samples],
            "prompts": [sample["prompts"] for sample in samples],
            "block_prompts": [list(sample["block_prompts"]) for sample in samples]}


def encode_block_prompt_batch(text_encoder: Callable[..., dict[str, torch.Tensor]],
                              sample_block_prompts: Sequence[Sequence[str]]) -> list[dict[str, torch.Tensor]]:
    if not sample_block_prompts:
        raise ValueError("Scheduled prompt batch is empty.")
    block_count = len(sample_block_prompts[0])
    if not block_count or any(len(prompts) != block_count for prompts in sample_block_prompts):
        raise ValueError("Scheduled prompt samples have inconsistent block counts.")
    unique = list(dict.fromkeys(prompt for prompts in sample_block_prompts for prompt in prompts))
    encoded = text_encoder(text_prompts=unique)
    if not encoded or any(value.shape[0] != len(unique) for value in encoded.values()):
        raise ValueError("Text encoder output does not match the unique prompt batch.")
    lookup, cache, result = {prompt: i for i, prompt in enumerate(unique)}, {}, []
    for block_index in range(block_count):
        indices = tuple(lookup[prompts[block_index]] for prompts in sample_block_prompts)
        if indices not in cache:
            tensor_indices = torch.tensor(indices, device=next(iter(encoded.values())).device)
            cache[indices] = {key: value.index_select(0, tensor_indices) for key, value in encoded.items()}
        result.append(cache[indices])
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render scheduled prompts from JSONL.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--mode", choices=SCHEDULES, default="full")
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--temporal_downsample", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = load_scheduled_prompts(
        args.jsonl,
        mode=args.mode,
        fps=args.fps,
        temporal_downsample=args.temporal_downsample,
        chunk_size=args.chunk_size,
    )
    for item in dataset:
        print(
            json.dumps(
                {
                    "id": item.prompt_id,
                    "mode": item.mode,
                    "prompt": item.prompt,
                    "frame_prompts": item.frame_prompts,
                    "chunk_prompts": item.chunk_prompts,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
