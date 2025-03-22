import numpy as np
import soundfile as sf
from generator import load_csm_1b

generator = load_csm_1b()
audio_segments = []

msg = "Hello, world! Testing to stream the MIMI decoder."

for frame in generator.generate_stream(msg, speaker=0, context=[]):
    audio_segments.append(frame.detach().cpu().numpy())

full_audio = np.concatenate(audio_segments)
sf.write("output.wav", full_audio, generator.sample_rate)