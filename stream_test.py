import sounddevice as sd
import numpy as np
import queue
from generator import load_csm_1b
import torch
from time import time

FRAME_SIZE = 1920
BUFFER_SIZE = 4

def print_time(start_time, msg):
    duration = (time() - start_time)
    if duration >=1:
        print(f"{msg} {duration:.02f} s")
    else:
        print(f"{msg} {duration*1000:.02f} ms")

class AudioStream:
    def __init__(self, samplerate):
        self.q = queue.Queue(maxsize=40)
        self.is_prebuffering = True
        self.prebuffer_size = BUFFER_SIZE
        self.buffer = []
        self.frames_in_queue = 0
        self.stream = sd.OutputStream(
            samplerate=samplerate,
            blocksize=FRAME_SIZE,
            channels=1,
            callback=self.callback
        )

    @torch.inference_mode()
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

            outdata[:] = new_frame.numpy()
        except queue.Empty:
            outdata[:] = 0
            self.is_prebuffering = True

    def start(self):
        self.stream.start()
        return self.q

start_time = time()
generator = load_csm_1b()
print_time(start_time, "load_csm_1b() time:")

stream_handler = AudioStream(generator.sample_rate)

q = stream_handler.start()

text = "Hello, world! Testing to stream the MIMI decoder. Please type in some text and CSM will generate the audio."

while [ 1 ]:
    stream_handler.start_time = time()
    frame_iter = generator.generate_stream(text, speaker=0, context=[])
    print_time(stream_handler.start_time, "generate_stream() time:")

    for frame in frame_iter:
        cpu_frame = frame.detach().cpu()
        q.put(cpu_frame)

    print("Type text to and hit Enter key to start streaming TTS...")
    text = input()