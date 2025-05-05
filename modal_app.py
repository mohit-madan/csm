import modal
import io
import torch


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


@app.cls(
    gpu="H100",
    volumes={"/root/data": modal_volume},
    timeout=600,
    min_containers=1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class TTS:
    @modal.enter()
    def load(self):
        import sys

        sys.path.append("/root/data")
        from generator import load_csm_1b

        self.generator = load_csm_1b(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sample_rate = self.generator.sample_rate

    @modal.method()
    def tts(self, text: str):
        import soundfile as sf

        audio = self.generator.generate(text, speaker=0, context=[])
        buf = io.BytesIO()
        sf.write(buf, audio.cpu().numpy(), self.sample_rate, format="wav")
        buf.seek(0)
        return buf.read()


@app.function()
@modal.fastapi_endpoint(docs=True)
def tts(text: str):
    from fastapi.responses import Response

    audio_bytes = TTS().tts.remote(text)
    return Response(audio_bytes, media_type="audio/wav")
