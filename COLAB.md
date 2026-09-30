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

The notebook clones a fresh repository checkout into `/content`, installs the
locked pixi environments, downloads the gated weights, and builds all seven
required TensorRT engines for the assigned GPU: two AVTR-1 motion engines, four
renderer engines, and HuBERT. The renderer's warp graph contains a custom
`GridSample3D` operation, and the HuBERT ONNX export has a dynamic output shape,
so neither currently has a viable ONNX Runtime fallback. It then lets you
capture a portrait with the
laptop camera and record speech through the browser before rendering and
previewing the resulting MP4. File upload remains available as a fallback when
camera access is unavailable.

This fork's notebook clones `RuurdKuiper/avtr-1-colab` by default so the kiosk
and local conversational server are present. The optional `AVTR_REPO_URL`
Colab secret overrides that URL; remove an old secret that still points to
`avaturn-live/avtr-1`, or update it to the fork URL.

The Gradio notebook mounts Google Drive by default and persists the seven
TensorRT engines plus the AVTR normalizer under
`MyDrive/avtr1-engine-cache/`. Active engines are restored into fast local
`/content` storage rather than loaded directly from Drive. Cache directories
are keyed by GPU name, compute capability, exact TensorRT version, and the
engine-builder source fingerprint; incomplete caches are ignored. The first
compatible run still builds and uploads the engines, while later fresh Colab
runtimes restore them. Set the optional Colab secret `AVTR_DRIVE_CACHE` to
`false` to disable Drive mounting and use ephemeral storage only. A different
GPU or TensorRT version intentionally creates a separate cache.

An optional final section turns the offline renderer into a fully local,
turn-by-turn conversational prototype. It records a question in the browser,
transcribes it with multilingual faster-whisper, generates a short reply with
Qwen3-4B-Instruct-2507, synthesizes that reply in the recorded voice with Chatterbox
Multilingual, and feeds the result back through AVTR-1. These models run in
the Colab runtime and do not call hosted inference APIs. The interface switches
between English and Dutch without reloading Qwen or AVTR. On A100, the notebook
uses the BF16 model: the similarly named fine-grained FP8 checkpoint needs an
H100-class GPU for native FP8 kernels. Set the optional `LLM_MODEL` Colab secret
to override the checkpoint.

The Gradio notebook wraps the same stages in one persistent server and keeps
the speech, language, voice, and AVTR renderer models warm between turns. Its
Kiosk tab first shows a silent browser-camera mirror. Stopping an 8–15 second
voice recording captures the latest camera frame, conditions the voice, registers
the avatar, and replaces the mirror with the persistent avatar conversation view.
The reusable models are preloaded before the share URL opens. Each enrollment and
conversation turn displays a timing table for speech recognition, LLM generation,
voice conditioning/synthesis, and avatar rendering, plus peak GPU memory for the
conversational process. AVTR runs in a persistent isolated Pixi worker, so new
enrollments do not rebuild its pipeline. Its share URL is a tunnel
to the current Colab process, not a deployment: it stops when the cell or
runtime stops. Browser media passes through the Gradio share tunnel to the
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
- The kiosk is a continuous visual experience, but conversation remains
  push-to-talk and turn-based. It waits for a complete TTS waveform and rendered
  MP4; it is not yet the low-latency WebRTC/live-streaming backend.
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
