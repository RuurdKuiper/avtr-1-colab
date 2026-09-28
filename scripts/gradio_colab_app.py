#!/usr/bin/env python3
"""Gradio front end for a turn-by-turn, voice-cloned AVTR-1 avatar.

This process runs in the lightweight conversational-model environment. AVTR-1
renders in its own Pixi environment so its TensorRT/PyTorch pins stay isolated.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import gradio as gr
import torch
import torchaudio as ta
from PIL import Image, ImageOps


SYSTEM_PROMPT = (
    "You are a friendly interactive avatar. Answer in one or two short, natural "
    "spoken sentences. Use plain text without markdown, lists, stage directions, "
    "or emoji."
)
TTS_REPO_ID = "ResembleAI/chatterbox"
TTS_ALLOW_PATTERNS = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]
PORTRAIT_SIZE = (1280, 720)
MIN_ENROLLMENT_SECONDS = 8.0
LANGUAGES = {
    "English": {
        "code": "en",
        "instruction": "Reply only in natural spoken English.",
    },
    "Nederlands": {
        "code": "nl",
        "instruction": "Antwoord uitsluitend in natuurlijk gesproken Nederlands.",
    },
}


class RendererWorkerDied(RuntimeError):
    """Raised when the persistent renderer exits before replying."""


class AvatarServer:
    def __init__(
        self,
        *,
        repo: Path,
        pixi: Path,
        storage: Path,
        work_dir: Path,
        llm_model: str,
        stt_model: str,
    ) -> None:
        self.repo = repo.resolve()
        self.pixi = pixi.resolve()
        self.storage = storage.resolve()
        self.work_dir = work_dir.resolve()
        self.llm_model_name = llm_model
        self.stt_model_name = stt_model
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_lock = threading.Lock()
        self._stt = None
        self._tokenizer = None
        self._llm = None
        self._tts = None
        self._tts_reference: str | None = None
        self._renderer_process: subprocess.Popen[str] | None = None
        self._renderer_logs: deque[str] = deque(maxlen=80)
        self._model_load_timings: dict[str, float] = {}
        atexit.register(self.close)

    def close(self) -> None:
        """Release the renderer subprocess when the Gradio server exits."""
        process = self._renderer_process
        self._renderer_process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _reference_frames_dir(self) -> Path:
        matches = sorted(self.storage.glob("*/avatars_artifacts/reference_frames"))
        if not matches:
            raise gr.Error(
                "AVTR portrait artifacts are missing. Run the notebook setup/download cell first."
            )
        return matches[0]

    def enroll(
        self,
        portrait_path: str | None,
        voice_path: str | None,
    ) -> tuple[dict[str, Any], str, str | None, str | None]:
        if not portrait_path:
            raise gr.Error("Take or upload a portrait first.")
        if not voice_path:
            raise gr.Error("Record or upload a clean voice-reference clip first.")

        session_id = uuid.uuid4().hex
        session_dir = self.work_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        avatar_id = f"colab_{session_id}"
        portrait_out = session_dir / "portrait.png"
        registered_portrait = self._reference_frames_dir() / f"{avatar_id}.png"
        voice_out = session_dir / "voice_reference.wav"

        try:
            with Image.open(portrait_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                # Match the renderer's 16:9 canvas without changing facial
                # proportions. The saved-preview component shows this exact
                # crop, so the user can retake/reframe before starting.
                image = ImageOps.fit(
                    image,
                    PORTRAIT_SIZE,
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
                image.save(portrait_out, format="PNG")
            shutil.copy2(portrait_out, registered_portrait)
            self._convert_audio(Path(voice_path), voice_out, sample_rate=24_000)
        except Exception:
            shutil.rmtree(session_dir, ignore_errors=True)
            registered_portrait.unlink(missing_ok=True)
            raise

        state = {
            "session_id": session_id,
            "session_dir": str(session_dir),
            "avatar_id": avatar_id,
            "portrait": str(portrait_out),
            "registered_portrait": str(registered_portrait),
            "voice_reference": str(voice_out),
            "history": [],
            "turn": 0,
        }
        return (
            state,
            "Ready. Record a question in the Conversation section.",
            str(portrait_out),
            str(voice_out),
        )

    def kiosk_enroll(
        self,
        portrait_path: str | None,
        voice_path: str | None,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[dict[str, Any], str, str | None, Any, Any, str]:
        """Finish the mirror phase and reveal the enrolled avatar."""
        started = time.perf_counter()
        if voice_path:
            duration = self._audio_duration(Path(voice_path))
            if duration < MIN_ENROLLMENT_SECONDS:
                raise gr.Error(
                    f"Keep talking a little longer: {duration:.1f}s recorded, "
                    f"but at least {MIN_ENROLLMENT_SECONDS:.0f}s is needed."
                )
        progress(0.05, desc="Saving your portrait and voice")
        state, _, portrait, _ = self.enroll(portrait_path, voice_path)
        saved_at = time.perf_counter()
        with self._gpu_lock:
            progress(0.20, desc="Preparing the local models")
            cold_loads = self._load_models(progress)
            progress(0.65, desc="Learning the voice reference")
            voice_time = self._prepare_voice_reference(state["voice_reference"])
            progress(0.80, desc="Preparing the avatar")
            avatar_started = time.perf_counter()
            self._prepare_avatar(
                avatar_id=state["avatar_id"],
                portrait=Path(state["registered_portrait"]),
            )
            avatar_time = time.perf_counter() - avatar_started
        total = time.perf_counter() - started
        enrollment_timings = {
            "Save portrait and audio": saved_at - started,
            **{
                f"Cold {name.removeprefix('load_')} load": value
                for name, value in cold_loads.items()
            },
            "Voice conditioning": voice_time,
            "Avatar registration": avatar_time,
            "Enrollment total": total,
        }
        profile = "### Enrollment profile\n\n| Stage | Time |\n|---|---:|\n" + "\n".join(
            f"| {name} | {value:.2f}s |" for name, value in enrollment_timings.items()
        )
        progress(1.0, desc="Avatar ready")
        return (
            state,
            f"Voice captured. Your avatar is ready ({duration:.1f}s reference).",
            portrait,
            gr.update(visible=False),
            gr.update(visible=True),
            profile,
        )

    def reset(
        self, state: dict[str, Any] | None
    ) -> tuple[dict[str, Any], list, str, str, None, None]:
        if state:
            state = dict(state)
            state["history"] = []
            state["turn"] = 0
        return state or {}, [], "", "", None, None

    def restart_kiosk(self, state: dict[str, Any] | None) -> tuple[Any, ...]:
        """Delete one visitor's temporary media and return to the mirror."""
        if state:
            avatar_id = state.get("avatar_id")
            if (
                avatar_id
                and self._renderer_process is not None
                and self._renderer_process.poll() is None
            ):
                with self._gpu_lock:
                    try:
                        self._renderer_request(
                            {
                                "id": uuid.uuid4().hex,
                                "command": "evict_avatar",
                                "avatar_id": avatar_id,
                            }
                        )
                    except RendererWorkerDied:
                        self.close()
            registered = state.get("registered_portrait")
            if registered:
                Path(registered).unlink(missing_ok=True)
            session_dir = state.get("session_dir")
            if session_dir:
                shutil.rmtree(session_dir, ignore_errors=True)
            if self._tts_reference == state.get("voice_reference"):
                self._tts_reference = None
        return (
            {},
            gr.update(visible=True),
            gr.update(visible=False),
            "### Look into the camera\nRecord at least 8 seconds of natural speech.",
            None,
            None,
            None,
            [],
            "",
            "",
            "",
        )

    def _convert_audio(self, source: Path, destination: Path, *, sample_rate: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is not installed in the Colab runtime")
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                str(destination),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Could not decode the audio: {result.stderr[-1200:]}")

    def _audio_duration(self, source: Path) -> float:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise RuntimeError("ffprobe is not installed in the Colab runtime")
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Could not inspect the audio: {result.stderr[-1200:]}")
        return float(result.stdout.strip())

    def _load_models(self, progress: gr.Progress) -> dict[str, float]:
        loaded_now: dict[str, float] = {}
        if self._stt is None:
            started = time.perf_counter()
            progress(0.05, desc="Loading speech recognition")
            from faster_whisper import WhisperModel

            self._stt = WhisperModel(
                self.stt_model_name, device="cuda", compute_type="float16"
            )
            elapsed = time.perf_counter() - started
            loaded_now["load_stt"] = elapsed
            self._model_load_timings["load_stt"] = elapsed
            print(f"Loaded {self.stt_model_name} in {elapsed:.1f}s", flush=True)
        if self._tokenizer is None or self._llm is None:
            started = time.perf_counter()
            progress(0.12, desc="Loading conversational model")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self._llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            elapsed = time.perf_counter() - started
            loaded_now["load_llm"] = elapsed
            self._model_load_timings["load_llm"] = elapsed
            print(f"Loaded {self.llm_model_name} in {elapsed:.1f}s", flush=True)
        if self._tts is None:
            started = time.perf_counter()
            progress(0.20, desc="Loading voice-cloning model")
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(
                repo_id=TTS_REPO_ID,
                allow_patterns=TTS_ALLOW_PATTERNS,
                token=os.getenv("HF_TOKEN") or None,
            )
            # chatterbox-tts 0.1.7 loads the multilingual V2 checkpoint by
            # default. Its from_local() predates the newer t3_model selector.
            self._tts = ChatterboxMultilingualTTS.from_local(
                local_path, device="cuda"
            )
            elapsed = time.perf_counter() - started
            loaded_now["load_tts"] = elapsed
            self._model_load_timings["load_tts"] = elapsed
            print(f"Loaded Chatterbox Multilingual in {elapsed:.1f}s", flush=True)
        return loaded_now

    def _transcribe(self, audio_path: str, language_code: str) -> str:
        segments, _ = self._stt.transcribe(
            audio_path, beam_size=5, vad_filter=True, language=language_code
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _respond(
        self,
        question: str,
        history: list[dict[str, str]],
        language_instruction: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT} {language_instruction}",
            },
            *history[-8:],
            {"role": "user", "content": question},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._llm.device)
        with torch.inference_mode():
            generated = self._llm.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.08,
            )
        reply_ids = generated[0][inputs.input_ids.shape[-1] :]
        reply = self._tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
        del inputs, generated, reply_ids
        return reply

    def _synthesize(
        self,
        reply: str,
        reference: str,
        output: Path,
        language_code: str,
    ) -> dict[str, float]:
        timings = {"voice_conditioning": 0.0}
        reference = str(Path(reference).resolve())
        with torch.inference_mode():
            if self._tts_reference != reference:
                timings["voice_conditioning"] = self._prepare_voice_reference(reference)
            started = time.perf_counter()
            # Omitting audio_prompt_path makes Chatterbox reuse self.conds
            # instead of re-reading and re-encoding the same reference clip.
            wav = self._tts.generate(reply, language_id=language_code)
            timings["tts"] = time.perf_counter() - started
            print(f"Synthesized reply in {timings['tts']:.1f}s", flush=True)
        ta.save(str(output), wav.cpu(), self._tts.sr)
        del wav
        return timings

    def _prepare_voice_reference(self, reference: str) -> float:
        reference = str(Path(reference).resolve())
        if self._tts_reference == reference:
            return 0.0
        started = time.perf_counter()
        print("Encoding the voice reference (once for this enrollment)...", flush=True)
        with torch.inference_mode():
            self._tts.prepare_conditionals(reference, exaggeration=0.5)
        self._tts_reference = reference
        elapsed = time.perf_counter() - started
        print(
            f"Voice reference encoded in {elapsed:.1f}s; subsequent turns will reuse it.",
            flush=True,
        )
        return elapsed

    def _prepare_avatar(self, *, avatar_id: str, portrait: Path) -> None:
        response = self._renderer_request(
            {
                "id": uuid.uuid4().hex,
                "command": "load_avatar",
                "avatar_id": avatar_id,
                "portrait": str(portrait.resolve()),
            }
        )
        if not response.get("ok"):
            raise RuntimeError(
                f"AVTR avatar preparation failed: {response.get('error', 'unknown worker error')}"
            )

    def warmup(self) -> None:
        """Load all reusable models before admitting the first visitor."""
        started = time.perf_counter()
        print("Preloading conversational models before opening the kiosk...", flush=True)
        with self._gpu_lock:
            self._load_models(lambda *args, **kwargs: None)
            self._ensure_renderer_worker()
        print(f"Kiosk model warmup finished in {time.perf_counter() - started:.1f}s.", flush=True)

    def _render(
        self,
        *,
        avatar_id: str,
        portrait: Path,
        speech: Path,
        output: Path,
        cfg_self_audio: float,
        noise_trunc_z: float,
    ) -> dict[str, float]:
        request_started = time.perf_counter()
        request = {
            "id": uuid.uuid4().hex,
            "command": "render",
            "avatar_id": avatar_id,
            "portrait": str(portrait.resolve()),
            "speech": str(speech.resolve()),
            "output": str(output.resolve()),
            "cfg_self_audio": float(cfg_self_audio),
            "noise_trunc_z": float(noise_trunc_z),
        }

        for attempt in range(2):
            try:
                response = self._renderer_request(request)
                if not response.get("ok"):
                    raise RuntimeError(
                        f"AVTR rendering failed: {response.get('error', 'unknown worker error')}"
                    )
                print(
                    f"Persistent renderer produced {response['frames']} frames in "
                    f"{response['elapsed']:.1f}s "
                    f"({response['elapsed'] / response['seconds']:.2f}x real time).",
                    flush=True,
                )
                return {
                    "renderer_total": time.perf_counter() - request_started,
                    "renderer_inference": float(response["elapsed"]),
                }
            except RendererWorkerDied:
                self.close()
                if attempt == 0:
                    print("Renderer worker stopped unexpectedly; restarting once...", flush=True)
                    continue
                raise

    def _renderer_request(self, request: dict[str, Any]) -> dict[str, Any]:
        process = self._ensure_renderer_worker()
        if process.stdin is None:
            raise RendererWorkerDied("Renderer worker stdin is unavailable")
        try:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RendererWorkerDied("Could not send a job to the renderer worker") from exc

        while True:
            message = self._read_renderer_message(process)
            if message.get("id") == request["id"]:
                return message

    @staticmethod
    def _timing_report(timings: dict[str, float]) -> str:
        labels = {
            "load_stt": "Cold load · speech recognition",
            "load_llm": "Cold load · language model",
            "load_tts": "Cold load · voice model",
            "stt": "Speech recognition",
            "llm": "Language model",
            "voice_conditioning": "Voice enrollment (first turn only)",
            "tts": "Voice synthesis",
            "renderer_total": "Avatar render (including startup/avatar load)",
            "renderer_inference": "Avatar render core",
            "total": "Total turn",
        }
        rows = [
            f"| {labels.get(name, name)} | {seconds:.2f}s |"
            for name, seconds in timings.items()
        ]
        peak = max(
            ((name, seconds) for name, seconds in timings.items() if name != "total"),
            key=lambda item: item[1],
            default=("total", timings.get("total", 0.0)),
        )
        gpu = ""
        if torch.cuda.is_available():
            allocated = torch.cuda.max_memory_allocated() / (1024**3)
            reserved = torch.cuda.max_memory_reserved() / (1024**3)
            gpu = (
                f"\n\nPeak GPU memory this process: **{allocated:.1f} GB allocated / "
                f"{reserved:.1f} GB reserved**."
            )
            nvidia_smi = shutil.which("nvidia-smi")
            if nvidia_smi:
                result = subprocess.run(
                    [
                        nvidia_smi,
                        "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    used, total = result.stdout.splitlines()[0].split(",")
                    gpu += (
                        " Total GPU use across both processes: "
                        f"**{used.strip()} / {total.strip()} MiB**."
                    )
        return (
            f"Slowest measured stage: **{labels.get(peak[0], peak[0])} ({peak[1]:.2f}s)**\n\n"
            "| Stage | Time |\n|---|---:|\n"
            + "\n".join(rows)
            + gpu
        )

    def _ensure_renderer_worker(self) -> subprocess.Popen[str]:
        process = self._renderer_process
        if process is not None and process.poll() is None:
            return process

        env = os.environ.copy()
        env["AVTR1_LOCAL_STORAGE"] = str(self.storage)
        env["PYTHONUNBUFFERED"] = "1"
        self._renderer_logs.clear()
        print("Starting persistent AVTR renderer (one-time model load)...", flush=True)
        process = subprocess.Popen(
            [
                str(self.pixi),
                "run",
                "python",
                "scripts/gradio_renderer_worker.py",
            ],
            cwd=self.repo,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._renderer_process = process
        while True:
            message = self._read_renderer_message(process)
            if message.get("event") == "ready":
                return process

    def _read_renderer_message(
        self, process: subprocess.Popen[str]
    ) -> dict[str, Any]:
        if process.stdout is None:
            raise RendererWorkerDied("Renderer worker stdout is unavailable")
        while True:
            line = process.stdout.readline()
            if line == "":
                returncode = process.poll()
                details = "\n".join(self._renderer_logs)
                raise RendererWorkerDied(
                    f"Renderer worker exited with status {returncode}.\n{details}"
                )
            line = line.rstrip()
            if line.startswith("AVTR_WORKER_JSON:"):
                try:
                    return json.loads(line.removeprefix("AVTR_WORKER_JSON:"))
                except json.JSONDecodeError as exc:
                    raise RendererWorkerDied(
                        f"Renderer worker sent invalid protocol data: {line}"
                    ) from exc
            self._renderer_logs.append(line)
            print(f"[renderer] {line}", flush=True)

    def turn(
        self,
        question_audio: str | None,
        state: dict[str, Any] | None,
        language: str,
        cfg_self_audio: float,
        noise_trunc_z: float,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[str, str, str, str, list[dict[str, str]], dict[str, Any], str]:
        if not state or not state.get("voice_reference"):
            raise gr.Error("Enroll a portrait and voice before starting a conversation.")
        if not question_audio:
            raise gr.Error("Record a question first.")
        language_config = LANGUAGES.get(language)
        if language_config is None:
            raise gr.Error(f"Unsupported language: {language}")
        language_code = language_config["code"]

        # AVTR/TensorRT and the conversational models all share one GPU.
        with self._gpu_lock:
            try:
                turn_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                timings = self._load_models(progress)
                progress(0.28, desc="Transcribing your question")
                started = time.perf_counter()
                question = self._transcribe(question_audio, language_code)
                timings["stt"] = time.perf_counter() - started
                if not question:
                    raise gr.Error("No speech was detected. Please record the question again.")

                progress(0.42, desc="Writing a response")
                history = list(state.get("history", []))[-8:]
                started = time.perf_counter()
                reply = self._respond(
                    question, history, language_config["instruction"]
                )
                timings["llm"] = time.perf_counter() - started
                if not reply:
                    raise RuntimeError("The language model returned an empty reply")

                turn_number = int(state.get("turn", 0)) + 1
                session_dir = Path(state["session_dir"])
                reply_audio = session_dir / f"reply_{turn_number:03d}.wav"
                reply_video = session_dir / f"reply_{turn_number:03d}.mp4"

                progress(0.58, desc="Cloning the voice")
                timings.update(
                    self._synthesize(
                        reply,
                        state["voice_reference"],
                        reply_audio,
                        language_code,
                    )
                )
                gc.collect()
                torch.cuda.empty_cache()

                progress(0.72, desc="Rendering the avatar video")
                timings.update(
                    self._render(
                        avatar_id=state["avatar_id"],
                        portrait=Path(state["registered_portrait"]),
                        speech=reply_audio,
                        output=reply_video,
                        cfg_self_audio=cfg_self_audio,
                        noise_trunc_z=noise_trunc_z,
                    )
                )

                history.extend(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": reply},
                    ]
                )
                state = dict(state)
                state["history"] = history[-8:]
                state["turn"] = turn_number
                timings["total"] = time.perf_counter() - turn_started
                with (session_dir / "timings.jsonl").open("a", encoding="utf-8") as metrics:
                    metrics.write(
                        json.dumps(
                            {
                                "turn": turn_number,
                                "language": language_code,
                                "question_chars": len(question),
                                "reply_chars": len(reply),
                                "timings_seconds": timings,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                profile = self._timing_report(timings)
                print(profile, flush=True)
                progress(1.0, desc="Done")
                return (
                    question,
                    reply,
                    str(reply_audio),
                    str(reply_video),
                    history,
                    state,
                    profile,
                )
            except gr.Error:
                raise
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

    def kiosk_turn(
        self,
        question_audio: str | None,
        state: dict[str, Any] | None,
        language: str,
        cfg_self_audio: float,
        noise_trunc_z: float,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[Any, ...]:
        """Conversation turn plus the kiosk's still-to-video transition."""
        result = self.turn(
            question_audio,
            state,
            language,
            cfg_self_audio,
            noise_trunc_z,
            progress,
        )
        video_path = result[3]
        return (
            *result[:3],
            gr.update(value=video_path, visible=True),
            *result[4:],
            gr.update(visible=False),
        )


def build_ui(server: AvatarServer) -> gr.Blocks:
    webcam_options = (
        {
            "webcam_options": gr.WebcamOptions(
                mirror=True,
                constraints={
                    "facingMode": "user",
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                    "aspectRatio": {"ideal": 16 / 9},
                },
            )
        }
        if hasattr(gr, "WebcamOptions")
        else {"mirror_webcam": True}
    )
    with gr.Blocks(
        title="Local AVTR-1 conversational avatar",
        css="""
        .kiosk-stage { max-width: 1100px; margin: 0 auto; }
        .kiosk-stage video, .kiosk-stage img { max-height: 70vh; object-fit: contain; }
        """,
    ) as demo:
        state = gr.State({})
        gr.Markdown(
            "# Conversational AVTR-1 avatar\n"
            "Everything is inferred in this Colab runtime. The Kiosk tab is the installation "
            "flow; the Manual controls tab is useful for testing. Only use a face and voice "
            "with the person's explicit consent."
        )

        with gr.Tab("Kiosk experience"):
            kiosk_status = gr.Markdown(
                "### Look into the camera\n"
                "Start the camera, then record at least 8 seconds of natural speech. "
                "Your microphone recording is not played back. Stopping the recording "
                "automatically creates the avatar."
            )
            with gr.Group(visible=True, elem_classes="kiosk-stage") as mirror_stage:
                kiosk_camera = gr.Image(
                    label="Live mirror",
                    sources=["webcam"],
                    type="filepath",
                    streaming=True,
                    **webcam_options,
                )
                latest_kiosk_frame = gr.Image(type="filepath", visible=False)
                kiosk_voice = gr.Audio(
                    label="Record 8–15 seconds, then press stop",
                    sources=["microphone"],
                    type="filepath",
                    format="wav",
                )
            with gr.Group(visible=False, elem_classes="kiosk-stage") as avatar_stage:
                kiosk_still = gr.Image(
                    label="Your avatar",
                    interactive=False,
                )
                kiosk_video = gr.Video(
                    label="Your avatar",
                    autoplay=True,
                    visible=False,
                )
                kiosk_question = gr.Audio(
                    label="Ask the avatar a question, then press stop",
                    sources=["microphone"],
                    type="filepath",
                    format="wav",
                )
                kiosk_language = gr.Radio(
                    choices=list(LANGUAGES),
                    value="English",
                    label="Conversation language / Gesprekstaal",
                )
                kiosk_transcript = gr.Textbox(label="Visitor", interactive=False)
                kiosk_response = gr.Textbox(label="Avatar", interactive=False)
                kiosk_chat = gr.Chatbot(label="Conversation")
                kiosk_response_audio = gr.Audio(visible=False)
                with gr.Accordion("Performance profile", open=False):
                    kiosk_profile = gr.Markdown()
                with gr.Accordion("Motion tuning", open=False):
                    kiosk_cfg = gr.Slider(
                        1.5, 3.5, value=2.7, step=0.1, label="Speech guidance"
                    )
                    kiosk_noise = gr.Slider(
                        0.8, 1.7, value=1.5, step=0.1, label="Motion range"
                    )
                kiosk_restart = gr.Button("Finish and welcome the next visitor")

        with gr.Tab("Manual controls"):
            with gr.Row():
                with gr.Column():
                    portrait = gr.Image(
                        label="Portrait",
                        sources=["webcam", "upload"],
                        type="filepath",
                    )
                    voice_reference = gr.Audio(
                        label="Clean 8–15 second voice sample",
                        sources=["microphone", "upload"],
                        type="filepath",
                        format="wav",
                    )
                    enroll_button = gr.Button("Save portrait and voice", variant="primary")
                    enrollment_status = gr.Textbox(label="Status", interactive=False)
                    saved_portrait = gr.Image(label="Saved portrait", interactive=False)
                    saved_voice = gr.Audio(label="Saved voice reference", interactive=False)
                with gr.Column():
                    language = gr.Radio(
                        choices=list(LANGUAGES),
                        value="English",
                        label="Conversation language / Gesprekstaal",
                    )
                    question_audio = gr.Audio(
                        label="Record a question",
                        sources=["microphone", "upload"],
                        type="filepath",
                        format="wav",
                    )
                    ask_button = gr.Button("Ask avatar", variant="primary")
                    with gr.Accordion("Motion tuning", open=False):
                        cfg_self_audio = gr.Slider(
                            1.5, 3.5, value=2.7, step=0.1, label="Speech guidance"
                        )
                        noise_trunc_z = gr.Slider(
                            0.8, 1.7, value=1.5, step=0.1, label="Motion range"
                        )
            with gr.Row():
                transcript = gr.Textbox(label="You", interactive=False)
                response = gr.Textbox(label="Avatar", interactive=False)
            with gr.Row():
                response_audio = gr.Audio(label="Cloned response", interactive=False)
                response_video = gr.Video(
                    label="Rendered avatar", interactive=False, autoplay=True
                )
            chat = gr.Chatbot(label="Conversation")
            with gr.Accordion("Performance profile", open=False):
                profile = gr.Markdown()
            reset_button = gr.Button("Clear conversation history")

        # The browser owns the live camera view. Colab samples it at a low rate
        # and retains only the latest frame for enrollment.
        kiosk_camera.stream(
            lambda frame: frame,
            inputs=kiosk_camera,
            outputs=latest_kiosk_frame,
            stream_every=0.5,
            time_limit=120,
            concurrency_limit=4,
            api_visibility="private",
        )
        kiosk_voice.stop_recording(
            server.kiosk_enroll,
            inputs=[latest_kiosk_frame, kiosk_voice],
            outputs=[
                state,
                kiosk_status,
                kiosk_still,
                mirror_stage,
                avatar_stage,
                kiosk_profile,
            ],
            concurrency_limit=1,
            api_visibility="private",
        )
        kiosk_question.stop_recording(
            server.kiosk_turn,
            inputs=[kiosk_question, state, kiosk_language, kiosk_cfg, kiosk_noise],
            outputs=[
                kiosk_transcript,
                kiosk_response,
                kiosk_response_audio,
                kiosk_video,
                kiosk_chat,
                state,
                kiosk_profile,
                kiosk_still,
            ],
            concurrency_limit=1,
            api_visibility="private",
        )
        kiosk_restart.click(
            server.restart_kiosk,
            inputs=[state],
            outputs=[
                state,
                mirror_stage,
                avatar_stage,
                kiosk_status,
                kiosk_voice,
                kiosk_still,
                kiosk_video,
                kiosk_chat,
                kiosk_transcript,
                kiosk_response,
                kiosk_profile,
            ],
            api_visibility="private",
        )

        enroll_button.click(
            server.enroll,
            inputs=[portrait, voice_reference],
            outputs=[state, enrollment_status, saved_portrait, saved_voice],
            api_visibility="private",
        )
        ask_button.click(
            server.turn,
            inputs=[question_audio, state, language, cfg_self_audio, noise_trunc_z],
            outputs=[transcript, response, response_audio, response_video, chat, state, profile],
            concurrency_limit=1,
            api_visibility="private",
        )
        reset_button.click(
            server.reset,
            inputs=[state],
            outputs=[state, chat, transcript, response, response_audio, response_video],
            api_visibility="private",
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--pixi", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("/content/avtr_gradio_sessions")
    )
    parser.add_argument("--llm-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--stt-model", default="small")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Open the UI before loading models (makes the first visitor wait).",
    )
    parser.add_argument("--no-share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Server preflight: Python {sys.version.split()[0]} | "
        f"Gradio {gr.__version__} | PyTorch {torch.__version__}",
        flush=True,
    )
    print(f"Repository: {args.repo}\nStorage: {args.storage}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This app requires a Colab GPU runtime")
    print(f"CUDA ready: {torch.cuda.get_device_name(0)}", flush=True)
    server = AvatarServer(
        repo=args.repo,
        pixi=args.pixi,
        storage=args.storage,
        work_dir=args.work_dir,
        llm_model=args.llm_model,
        stt_model=args.stt_model,
    )
    if not args.no_preload:
        server.warmup()
    demo = build_ui(server)
    print("Gradio interface constructed successfully; opening share tunnel...", flush=True)
    username = os.environ.get("GRADIO_USERNAME", "avatar")
    password = os.environ.get("GRADIO_PASSWORD")
    auth = (username, password) if password else None
    demo.queue(max_size=8, default_concurrency_limit=1, api_open=False).launch(
        server_name="0.0.0.0",
        share=not args.no_share,
        auth=auth,
        auth_message="Private Colab avatar prototype",
        allowed_paths=[str(args.work_dir.resolve())],
        show_error=True,
    )


if __name__ == "__main__":
    main()
