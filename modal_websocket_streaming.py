import modal
import torch
from huggingface_hub import hf_hub_download
from generator import Segment
import torchaudio
import numpy as np
import time


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
app = modal.App(
    name="csm-tts-websocket-modal", image=image, volumes={"/root/data": modal_volume}
)

image_with_source = image.add_local_python_source("generator", "models", "watermarking")


@app.cls(
    gpu="A100",
    volumes={"/root/data": modal_volume},
    timeout=600,
    min_containers=1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    image=image_with_source,
)
class TTS:
    @modal.enter()
    def load(self):
        import sys

        start_time = time.time()

        sys.path.append("/root/data")
        from generator import load_csm_1b

        self.generator = load_csm_1b(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sample_rate = self.generator.sample_rate

        # Download prompt files
        prompt_filepath_conversational_a = hf_hub_download(
            repo_id="sesame/csm-1b", filename="prompts/conversational_a.wav"
        )
        prompt_filepath_conversational_b = hf_hub_download(
            repo_id="sesame/csm-1b", filename="prompts/conversational_b.wav"
        )

        SPEAKER_PROMPTS = {
            "conversational_a": {
                "text": (
                    "like revising for an exam I'd have to try and like keep up the momentum because I'd "
                    "start really early I'd be like okay I'm gonna start revising now and then like "
                    "you're revising for ages and then I just like start losing steam I didn't do that "
                    "for the exam we had recently to be fair that was a more of a last minute scenario "
                    "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
                    "sort of start the day with this not like a panic but like a"
                ),
                "audio": prompt_filepath_conversational_a,
            },
            "conversational_b": {
                "text": (
                    "like a super Mario level. Like it's very like high detail. And like, once you get "
                    "into the park, it just like, everything looks like a computer game and they have all "
                    "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
                    "will have like a question block. And if you like, you know, punch it, a coin will "
                    "come out. So like everyone, when they come into the park, they get like this little "
                    "bracelet and then you can go punching question blocks around."
                ),
                "audio": prompt_filepath_conversational_b,
            },
        }

        self.segments = [
            Segment(
                text=SPEAKER_PROMPTS["conversational_a"]["text"],
                speaker=0,
                audio=self.load_prompt_audio(
                    SPEAKER_PROMPTS["conversational_a"]["audio"], self.sample_rate
                ),
            )
        ]

        self.generator.update_ctx_tokens(self.segments)
        print(
            f"[Timing] Model/context loaded in {time.time() - start_time:.2f} seconds"
        )

    @modal.method()
    def tts_true_streaming(
        self, text: str, speaker_id: int = 0, reset_context: bool = False
    ):
        """Generate audio frames incrementally as they are being produced by the model"""
        import soundfile as sf
        from io import BytesIO

        overall_start = time.time()
        print(f"[TTS] Start tts_true_streaming at {overall_start:.4f}")

        # Reset context if requested
        if reset_context:
            self.segments = []

        # Pseudocode: trim context if too long
        max_context_segments = 5  # or whatever fits under the limit
        if len(self.segments) > max_context_segments:
            self.segments = self.segments[-max_context_segments:]

        # Get frame iterator from the generator
        pre_gen_stream = time.time()
        print(f"[TTS] Before generator.generate_stream at {pre_gen_stream:.4f}")
        frame_iter = self.generator.generate_stream(
            text, speaker=speaker_id, context=self.segments
        )
        post_gen_stream = time.time()
        print(
            f"[TTS] After generator.generate_stream at {post_gen_stream:.4f} (delta: {post_gen_stream - pre_gen_stream:.4f}s)"
        )

        # For storing complete audio for context
        collected_frames = []

        # Initialize WAV header
        wav_header = BytesIO()
        # Write a placeholder WAV header that will be overwritten with each frame
        # We use a 10-second placeholder (adjust based on expected output length)
        placeholder_length = 5 * self.sample_rate
        sf.write(
            wav_header, np.zeros(placeholder_length), self.sample_rate, format="wav"
        )
        wav_header_bytes = wav_header.getvalue()[:44]  # Standard WAV header is 44 bytes

        # First yield the header
        pre_header_yield = time.time()
        print(
            f"[TTS] Yielding WAV header at {pre_header_yield:.4f} (delta from start: {pre_header_yield - overall_start:.4f}s)"
        )
        yield wav_header_bytes

        # --- Buffering logic ---
        frame_size = 512  # Example: model outputs 512 samples per frame
        buffer_frames = 3  # 5 frames = 2560 samples (~106ms at 24kHz)
        buffer = []
        samples_in_buffer = 0

        frame_count = 0
        frame_times = []
        first_chunk_yielded = False

        for frame in frame_iter:
            frame_start = time.time()

            collected_frames.append(frame)
            frame_np = frame.cpu().numpy().astype("float32")
            buffer.append(frame_np)
            samples_in_buffer += frame_np.shape[0]

            if samples_in_buffer >= frame_size * buffer_frames:
                chunk = np.concatenate(buffer, axis=0)
                if not first_chunk_yielded:
                    print(
                        f"[TTS] Yielding first audio chunk at {time.time():.4f} (delta from start: {time.time() - overall_start:.4f}s)"
                    )
                    first_chunk_yielded = True
                yield chunk.tobytes()
                buffer = []
                samples_in_buffer = 0

            frame_end = time.time()
            frame_times.append(frame_end - frame_start)
            frame_count += 1

        # Flush remaining
        if buffer:
            chunk = np.concatenate(buffer, axis=0)
            if not first_chunk_yielded:
                print(
                    f"[TTS] Yielding first audio chunk (from flush) at {time.time():.4f} (delta from start: {time.time() - overall_start:.4f}s)"
                )
                first_chunk_yielded = True
            yield chunk.tobytes()

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

        overall_end = time.time()
        print(
            f"[Timing] TTS request for '{text[:30]}...' took {overall_end - overall_start:.2f} seconds"
        )
        if frame_times:
            print(
                f"[Timing] Average frame generation time: {np.mean(frame_times):.4f} seconds over {frame_count} frames"
            )

        # After all audio chunks are sent
        yield b""  # Zero-length chunk as end-of-audio signal

    def load_prompt_audio(self, audio_path, target_sample_rate):
        start = time.time()
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = audio_tensor.squeeze(0)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
        )
        print(
            f"[Timing] Loaded and resampled prompt audio in {time.time() - start:.2f} seconds"
        )
        return audio_tensor


@app.function()
@modal.asgi_app()
def websocket_tts_app():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    import json

    app = FastAPI()

    @app.websocket("/ws")
    async def websocket_handler(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                # Receive JSON message with TTS parameters
                data = await websocket.receive_text()
                params = json.loads(data)
                text = params.get("text", "")
                speaker_id = params.get("speaker_id", 0)
                reset_context = params.get("reset_context", False)

                # Stream audio chunks as binary messages
                first_chunk_sent = False
                for chunk in TTS().tts_true_streaming.remote_gen(
                    text, speaker_id, reset_context
                ):
                    if not first_chunk_sent:
                        print(
                            f"[WebSocket] Sending first chunk to client at {time.time():.4f}"
                        )
                        first_chunk_sent = True
                    await websocket.send_bytes(chunk)
                # After streaming all chunks
                await websocket.send_bytes(b"")  # End-of-audio signal
        except WebSocketDisconnect:
            pass  # Client disconnected

    return app
