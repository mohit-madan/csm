import sounddevice as sd
import numpy as np
import queue
from generator import load_csm_1b
import torch
from time import time

def print_time(start_time, msg):
    duration = (time() - start_time)
    if duration >=1:
        print(f"{msg} {duration:.02f} s")
    else:
        print(f"{msg} {duration*1000:.02f} ms")

class AudioStream:
    def __init__(self, samplerate):
        self.q = queue.Queue(maxsize=30)
        self.is_prebuffering = True
        self.prebuffer_size = 6
        self.frames_in_queue = 0
        self.stream = sd.OutputStream(
            samplerate=samplerate,
            blocksize=1920,
            channels=1,
            callback=self.callback
        )
    def callback(self, outdata, frames, time, status):
        if self.q.qsize() != self.frames_in_queue:
            self.frames_in_queue = self.q.qsize()
            print(f"{self.frames_in_queue} frames in queue.")
            if self.start_time:
                print_time(self.start_time, "Time to first frame ready for playback:")
                self.start_time = None

        if self.is_prebuffering:
            if self.q.qsize() < self.prebuffer_size:
                outdata[:] = 0
                return
            else:
                self.is_prebuffering = False
        try:
            new_frame = self.q.get_nowait().unsqueeze(1)

            if new_frame.shape[0] < outdata.shape[0]:
                padding = torch.zeros(outdata.shape[0] - new_frame.shape[0], 1)
                new_frame = torch.vstack((new_frame, padding))
            outdata[:] = new_frame.detach().cpu().numpy()
        except queue.Empty:
            outdata[:] = 0
            self.is_prebuffering = True

    def start(self):
        self.stream.start()
        return self.q




start_time = time()
generator = load_csm_1b()
print_time(start_time, "load_csm_1b() time:")

msg1 = "Hello, world! Testing to stream the MIMI decoder."
msg2 = """CSM (Conversational Speech Model) is a speech generation model from Sesame that generates RVQ audio codes from text and audio inputs. The model architecture employs a Llama backbone and a smaller audio decoder that produces Mimi audio codes.
A fine-tuned variant of CSM powers the interactive voice demo shown in our blog post.
A hosted Hugging Face space is also available for testing audio generation."""

stream_handler = AudioStream(generator.sample_rate)

q = stream_handler.start()

for msg in [msg1, msg2]:
    print("Press Enter key to start streaming TTS...")
    input()
    stream_handler.start_time = time()
    frame_iter = generator.generate_stream(msg, speaker=0, context=[])
    print_time(start_time, "generate_stream() time:")

    for frame in frame_iter:
        q.put(frame)