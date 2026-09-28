#!/usr/bin/env python3
"""Persistent AVTR renderer used by the Colab Gradio front end.

The process owns the Pixi/TensorRT environment and communicates with the
lightweight conversational server using newline-delimited JSON on stdin/stdout.
Human-readable model logs may also appear on stdout; protocol messages are
distinguished by ``PROTOCOL_PREFIX``.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import torch

import generate_offline as offline
from avtr1_renderer.avatar_loader import Avatar
from avtr1_renderer.pipeline import Pipeline
from avtr1_renderer.types import Chunk, RenderOptions


PROTOCOL_PREFIX = "AVTR_WORKER_JSON:"
MAX_CACHED_AVATARS = 4


def _emit(message: dict[str, Any]) -> None:
    print(PROTOCOL_PREFIX + json.dumps(message, separators=(",", ":")), flush=True)


def _render(
    pipeline: Pipeline,
    avatar: Avatar,
    *,
    speech_path: Path,
    output_path: Path,
    cfg_self_audio: float,
    noise_trunc_z: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    speech_raw = offline._load_mono_16k(speech_path)
    speech, listen = offline._align_tracks(speech_raw, None, None)
    window = offline._chunk_window(pipeline)
    step = offline._chunk_step(pipeline)
    speech_chunks = offline._slice_chunks(speech, window, step)
    listen_chunks = offline._slice_chunks(listen, window, step)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_h, out_w = avatar.source.shape[-2:]
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        size=(out_w, out_h),
        fps=offline.FPS,
        codec="libx264",
        pix_fmt_in="yuv420p",
        pix_fmt_out="yuv420p",
        quality=8,
        macro_block_size=1,
        audio_path=str(speech_path),
        audio_codec="aac",
    )
    writer.send(None)

    options = RenderOptions(
        pixel_format="yuv_i420",
        bg_id="plain_white",
        cfg_self_audio=cfg_self_audio,
        noise_trunc_z=noise_trunc_z,
        stream_frames=True,
    )
    state = None
    produced = 0
    try:
        for speech_chunk, listen_chunk in zip(speech_chunks, listen_chunks):
            chunk = Chunk(audio_speech=speech_chunk, audio_listen=listen_chunk)
            state, frames = pipeline.process_chunk(avatar, chunk, state, options)
            for frame in frames:
                writer.send(frame.data.tobytes())
                produced += 1
    finally:
        writer.close()

    if produced == 0:
        raise RuntimeError("AVTR produced no video frames")
    return {
        "output": str(output_path),
        "frames": produced,
        "seconds": produced / offline.FPS,
        "elapsed": time.perf_counter() - started,
    }


def main() -> None:
    print("Persistent renderer: loading AVTR pipeline once...", flush=True)
    pipeline, _ = Pipeline.from_artifacts(avatar_ids=None)
    avatars: OrderedDict[str, Avatar] = OrderedDict()
    print("Persistent renderer: ready for jobs.", flush=True)
    _emit({"event": "ready"})

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        request_id: str | None = None
        try:
            request = json.loads(raw_line)
            request_id = str(request["id"])
            if request.get("command") == "shutdown":
                _emit({"id": request_id, "ok": True})
                return
            if request.get("command") != "render":
                raise ValueError(f"Unknown command: {request.get('command')!r}")

            avatar_id = str(request["avatar_id"])
            portrait_path = Path(request["portrait"]).resolve()
            speech_path = Path(request["speech"]).resolve()
            output_path = Path(request["output"]).resolve()
            if not speech_path.is_file():
                raise FileNotFoundError(f"Speech audio not found: {speech_path}")

            avatar = avatars.pop(avatar_id, None)
            if avatar is None:
                print(f"Persistent renderer: registering avatar {avatar_id!r}...", flush=True)
                avatar = pipeline.load_avatar(portrait_path, avatar_id=avatar_id)
            avatars[avatar_id] = avatar
            while len(avatars) > MAX_CACHED_AVATARS:
                evicted_id, evicted_avatar = avatars.popitem(last=False)
                del evicted_avatar
                torch.cuda.empty_cache()
                print(f"Persistent renderer: evicted avatar {evicted_id!r}.", flush=True)

            result = _render(
                pipeline,
                avatar,
                speech_path=speech_path,
                output_path=output_path,
                cfg_self_audio=float(request["cfg_self_audio"]),
                noise_trunc_z=float(request["noise_trunc_z"]),
            )
            _emit({"id": request_id, "ok": True, **result})
        except Exception as exc:
            traceback.print_exc()
            _emit(
                {
                    "id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


if __name__ == "__main__":
    main()
