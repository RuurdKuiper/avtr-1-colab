# AVTR-1 proof of concept on Google Colab

The current AVTR-1 runtime is built for Linux and NVIDIA CUDA. It does not run
natively on an Apple Silicon Mac without a substantial renderer/runtime port.
For the simplest proof of concept, use the single-cell Gradio notebook:

[`colab/avtr1_gradio_poc.ipynb`](colab/avtr1_gradio_poc.ipynb)

It launches a password-protected temporary `gradio.live` interface for webcam
portrait capture, microphone voice enrollment, and push-to-talk conversational
turns. If you prefer to inspect and run every stage separately, use:

[`colab/avtr1_offline_poc.ipynb`](colab/avtr1_offline_poc.ipynb)

## Before opening the notebook

1. Sign in to Hugging Face and accept the access conditions on
   <https://huggingface.co/avaturn-live/avtr-1>.
2. Create a Hugging Face token with read access.
3. In Colab, select a GPU runtime. An L4 or A100 is preferred. A T4 is supported
   by TensorRT but is older than the upstream project's Ampere recommendation,
   so engine building and rendering may be slower or run out of memory.
4. Add the token to Colab secrets as `HF_TOKEN`, or enter it in the notebook's
   hidden password prompt.

The notebook clones a fresh upstream checkout into `/content`, installs the
locked pixi environments, downloads the gated weights, and builds the two
mandatory AVTR-1 TensorRT motion engines for the assigned GPU. Renderer and
HuBERT models use their CUDA ONNX Runtime fallbacks by default; the notebook
offers a switch for a slower full TensorRT build when repeated-render speed is
more important than setup time. It then lets you capture a portrait with the
laptop camera and record speech through the browser before rendering and
previewing the resulting MP4. File upload remains available as a fallback when
camera access is unavailable.

An optional final section turns the offline renderer into a fully local,
turn-by-turn conversational prototype. It records a question in the browser,
transcribes it with faster-whisper, generates a short reply with Qwen3-1.7B,
synthesizes that reply in the recorded voice with Chatterbox Turbo, and feeds
the result back through AVTR-1. These models run in the Colab runtime and do
not call hosted inference APIs. The first conversational turn downloads the
additional model weights and is therefore substantially slower than later
turns.

The Gradio notebook wraps the same stages in one persistent server and keeps
the speech, language, and voice models warm between turns. Its share URL is a
tunnel to the current Colab process, not a deployment: it stops when the cell
or runtime stops. Browser media passes through the Gradio share tunnel to the
Colab runtime, so use the generated password and do not treat the URL as a
private production endpoint.

For best results, use a centered, near-frontal portrait with visible shoulders,
even lighting, and no face occlusion. A 3–10 second clip is enough for an
animation-only test; for voice cloning, use a clean 8–15 second reference so
the TTS model has more speaker information.
The notebook asks for browser camera and microphone permission and provides its
own capture and stop buttons. Only use a person's image and voice with their
explicit consent.

The render cell exposes conservative motion controls. Its `lively` defaults use
stronger speech guidance (`CFG_SELF_AUDIO = 2.7`) and a wider motion-noise range
(`NOISE_TRUNC_Z = 1.5`) than upstream, without requiring an engine rebuild.
Return them to `2.0` and `1.2` respectively if the face distorts or the pose
jumps. The public renderer animates face/head motion only; its simplified
pasteback stage does not implement the reference pipeline's body/shoulder-motion
simulation.

## Important limitations

- Colab does not guarantee a particular GPU. TensorRT engines are tied to the
  build environment and GPU architecture, so rebuild them when the runtime or
  GPU changes.
- The first run is the slow one because dependencies, weights, and engines all
  need to be prepared.
- This notebook demonstrates offline generation and optional turn-by-turn
  conversation, not the low-latency WebRTC live demo.
- Colab runtimes are temporary. Download the resulting MP4 before disconnecting.
- The model, renderer, streamer, and InsightFace dependency have different
  license conditions. Review the repository license files before use, and only
  use portraits/audio for which you have appropriate consent.

## Why native macOS is not a quick switch

The upstream project currently pins `linux-64`, PyTorch CUDA wheels,
`onnxruntime-gpu`, TensorRT, and CUDA NVCC. Its runtime passes CUDA tensor
pointers directly between PyTorch, ONNX Runtime, TensorRT, and a Linux
`libgrid_sample_3d_plugin.so` plugin. Docker Desktop on an Apple Silicon Mac
does not provide an NVIDIA CUDA GPU, so putting the existing code in a Linux
container does not solve the backend mismatch.

A real Mac port would need a new MPS/Core ML or CPU inference backend, portable
implementations of the custom rendering operations, device-neutral tensor
handling throughout the pipeline, new dependency/platform locks, and extensive
output validation. That is a separate engineering project rather than a slower
configuration of the current runtime.
