import modal
import torch
from fastapi.responses import Response


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .run_commands(["apt update", "apt upgrade -y"])
    .apt_install(["build-essential", "git"])
    .pip_install("fastapi[standard]")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("generator.py", remote_path="/root/generator.py")
    .add_local_file("models.py", remote_path="/root/models.py")
    .add_local_file("watermarking.py", remote_path="/root/watermarking.py")
)


modal_volume = modal.Volume.from_name("csm-tts-data", create_if_missing=True)
app = modal.App(name="csm-tts-modal", image=image, volumes={"/root/data": modal_volume})


@app.function(gpu="H100", secrets=[modal.Secret.from_name("huggingface-secret")])
def tts_streaming(text: str):
    import sys
    import io
    import wave
    import numpy as np

    sys.path.append("/root")
    from generator import load_csm_1b

    generator = load_csm_1b(device="cuda" if torch.cuda.is_available() else "cpu")
    sample_rate = generator.sample_rate

    # Generate all audio frames
    frame_iter = generator.generate_stream(text, speaker=0, context=[])
    frames = [frame.cpu().numpy() for frame in frame_iter]
    audio = np.concatenate(frames)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    buf.seek(0)
    return buf.read()


@app.function()
@modal.fastapi_endpoint(docs=True)
def tts_stream(text: str):
    wav_bytes = tts_streaming.remote(text)
    return Response(wav_bytes, media_type="audio/wav")
