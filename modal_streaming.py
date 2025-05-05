import modal
import io
import torch
import asyncio
from typing import AsyncIterator, List
from fastapi.responses import StreamingResponse


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
        self.segments = []  # To store context for multi-turn conversations

    @modal.method()
    def tts_true_streaming(
        self, text: str, speaker_id: int = 0, reset_context: bool = False
    ):
        """Generate audio frames incrementally as they are being produced by the model"""
        import soundfile as sf
        import numpy as np
        from io import BytesIO

        # Reset context if requested
        if reset_context:
            self.segments = []

        # Get frame iterator from the generator
        frame_iter = self.generator.generate_stream(
            text, speaker=speaker_id, context=self.segments
        )

        # For storing complete audio for context
        collected_frames = []

        # Initialize WAV header
        wav_header = BytesIO()
        # Write a placeholder WAV header that will be overwritten with each frame
        # We use a 10-second placeholder (adjust based on expected output length)
        placeholder_length = 10 * self.sample_rate
        sf.write(
            wav_header, np.zeros(placeholder_length), self.sample_rate, format="wav"
        )
        wav_header_bytes = wav_header.getvalue()[:44]  # Standard WAV header is 44 bytes

        # First yield the header
        yield wav_header_bytes

        # Process each frame
        for frame in frame_iter:
            # Store frame for context
            collected_frames.append(frame)

            # Convert frame to WAV data (excluding header)
            frame_np = frame.cpu().numpy()
            buf = BytesIO()
            sf.write(buf, frame_np, self.sample_rate, format="wav")
            buf.seek(0)

            # Skip the header (first 44 bytes) and yield only the audio data
            buf.read(44)  # Skip header
            frame_data = buf.read()
            yield frame_data

        # Store the generated audio in context for future requests
        if collected_frames:
            from generator import Segment

            full_audio = torch.cat(collected_frames, dim=0)
            if full_audio.ndim == 2 and full_audio.shape[1] == 2:
                full_audio = full_audio.mean(dim=1, keepdim=True)
            elif full_audio.ndim == 2 and full_audio.shape[1] == 1:
                full_audio = full_audio.squeeze(1)
            self.segments.append(
                Segment(text=text, speaker=speaker_id, audio=full_audio)
            )
            self.generator.update_ctx_tokens(self.segments)


@app.function()
@modal.fastapi_endpoint(docs=True)
async def tts_stream(text: str, speaker_id: int = 0, reset_context: bool = False):
    """Endpoint for true streaming TTS where audio is generated incrementally"""

    async def stream_generator() -> AsyncIterator[bytes]:
        for chunk in TTS().tts_true_streaming.remote_gen(
            text, speaker_id, reset_context
        ):
            yield chunk
            # Small delay to avoid overwhelming the client
            await asyncio.sleep(0.01)

    return StreamingResponse(
        stream_generator(),
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename=tts_output.wav"},
    )
