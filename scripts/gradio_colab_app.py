#!/usr/bin/env python3
"""Gradio front end for a turn-by-turn, voice-cloned AVTR-1 avatar.

This process runs in the lightweight conversational-model environment. AVTR-1
renders in its own Pixi environment so its TensorRT/PyTorch pins stay isolated.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
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
TTS_REPO_ID = "ResembleAI/chatterbox-turbo"
TTS_ALLOW_PATTERNS = ["ve.safetensors", "t3_turbo_v1.safetensors", "s3gen_meanflow.safetensors", "conds.pt", "*.json", "*.txt", "*.model"]


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

    def reset(self, state: dict[str, Any] | None) -> tuple[dict[str, Any], list, str, str, None, None]:
        if state:
            state = dict(state)
            state["history"] = []
            state["turn"] = 0
        return state or {}, [], "", "", None, None

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

    def _load_models(self, progress: gr.Progress) -> None:
        if self._stt is None:
            started = time.perf_counter()
            progress(0.05, desc="Loading speech recognition")
            from faster_whisper import WhisperModel

            self._stt = WhisperModel(
                self.stt_model_name, device="cuda", compute_type="float16"
            )
            print(f"Loaded {self.stt_model_name} in {time.perf_counter() - started:.1f}s", flush=True)
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
            print(f"Loaded {self.llm_model_name} in {time.perf_counter() - started:.1f}s", flush=True)
        if self._tts is None:
            started = time.perf_counter()
            progress(0.20, desc="Loading voice-cloning model")
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(repo_id=TTS_REPO_ID, allow_patterns=TTS_ALLOW_PATTERNS, token=os.getenv("HF_TOKEN") or None)
            self._tts = ChatterboxTurboTTS.from_local(local_path, device="cuda")
            print(f"Loaded Chatterbox Turbo in {time.perf_counter() - started:.1f}s", flush=True)

    def _transcribe(self, audio_path: str) -> str:
        segments, _ = self._stt.transcribe(
            audio_path, beam_size=5, vad_filter=True
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _respond(self, question: str, history: list[dict[str, str]]) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history[-8:],
            {"role": "user", "content": question},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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

    def _synthesize(self, reply: str, reference: str, output: Path) -> None:
        with torch.inference_mode():
            wav = self._tts.generate(reply, audio_prompt_path=reference)
        ta.save(str(output), wav.cpu(), self._tts.sr)
        del wav

    def _render(
        self,
        *,
        avatar_id: str,
        speech: Path,
        output: Path,
        cfg_self_audio: float,
        noise_trunc_z: float,
    ) -> None:
        wrapper = f"""
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('avtr1_generate_offline', Path('scripts/generate_offline.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
default_render_options = module.RenderOptions
def tuned_render_options(**kwargs):
    kwargs.update(cfg_self_audio={float(cfg_self_audio)}, noise_trunc_z={float(noise_trunc_z)})
    return default_render_options(**kwargs)
module.RenderOptions = tuned_render_options
module.main()
"""
        env = os.environ.copy()
        env["AVTR1_LOCAL_STORAGE"] = str(self.storage)
        result = subprocess.run(
            [
                str(self.pixi),
                "run",
                "python",
                "-c",
                wrapper,
                "--avatar",
                avatar_id,
                "--speech",
                str(speech),
                "--bg",
                "plain_white",
                "--out",
                str(output),
            ],
            cwd=self.repo,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            details = (result.stderr or result.stdout)[-3000:]
            raise RuntimeError(f"AVTR rendering failed:\n{details}")

    def turn(
        self,
        question_audio: str | None,
        state: dict[str, Any] | None,
        cfg_self_audio: float,
        noise_trunc_z: float,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[str, str, str, str, list[dict[str, str]], dict[str, Any]]:
        if not state or not state.get("voice_reference"):
            raise gr.Error("Enroll a portrait and voice before starting a conversation.")
        if not question_audio:
            raise gr.Error("Record a question first.")

        # AVTR/TensorRT and the conversational models all share one GPU.
        with self._gpu_lock:
            try:
                self._load_models(progress)
                progress(0.28, desc="Transcribing your question")
                question = self._transcribe(question_audio)
                if not question:
                    raise gr.Error("No speech was detected. Please record the question again.")

                progress(0.42, desc="Writing a response")
                history = list(state.get("history", []))[-8:]
                reply = self._respond(question, history)
                if not reply:
                    raise RuntimeError("The language model returned an empty reply")

                turn_number = int(state.get("turn", 0)) + 1
                session_dir = Path(state["session_dir"])
                reply_audio = session_dir / f"reply_{turn_number:03d}.wav"
                reply_video = session_dir / f"reply_{turn_number:03d}.mp4"

                progress(0.58, desc="Cloning the voice")
                self._synthesize(reply, state["voice_reference"], reply_audio)
                gc.collect()
                torch.cuda.empty_cache()

                progress(0.72, desc="Rendering the avatar video")
                self._render(
                    avatar_id=state["avatar_id"],
                    speech=reply_audio,
                    output=reply_video,
                    cfg_self_audio=cfg_self_audio,
                    noise_trunc_z=noise_trunc_z,
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
                progress(1.0, desc="Done")
                return (
                    question,
                    reply,
                    str(reply_audio),
                    str(reply_video),
                    history,
                    state,
                )
            except gr.Error:
                raise
            except Exception as exc:
                raise gr.Error(str(exc)) from exc


def build_ui(server: AvatarServer) -> gr.Blocks:
    with gr.Blocks(title="Local AVTR-1 conversational avatar") as demo:
        state = gr.State({})
        gr.Markdown(
            "# Conversational AVTR-1 avatar\n"
            "Everything is inferred in this Colab runtime. First enroll a portrait and "
            "voice sample, then use push-to-talk turns. Only use a face and voice with "
            "the person's explicit consent."
        )

        with gr.Tab("1. Enroll portrait and voice"):
            with gr.Row():
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

        with gr.Tab("2. Conversation"):
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
                response_video = gr.Video(label="Rendered avatar", interactive=False)
            chat = gr.Chatbot(label="Conversation")
            reset_button = gr.Button("Clear conversation history")

        enroll_button.click(
            server.enroll,
            inputs=[portrait, voice_reference],
            outputs=[state, enrollment_status, saved_portrait, saved_voice],
            api_visibility="private",
        )
        ask_button.click(
            server.turn,
            inputs=[question_audio, state, cfg_self_audio, noise_trunc_z],
            outputs=[transcript, response, response_audio, response_video, chat, state],
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
    parser.add_argument("--llm-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--stt-model", default="small.en")
    parser.add_argument("--no-share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Server preflight: Python {sys.version.split()[0]} | Gradio {gr.__version__} | PyTorch {torch.__version__}", flush=True)
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
