import sounddevice as sd
import numpy as np
import queue
from generator import load_csm_1b, Segment
import torch
from time import time

FRAME_SIZE = 1920
BUFFER_SIZE = 2

def print_time(start_time, msg):
    duration = (time() - start_time)
    if duration >=1:
        print(f"{msg} {duration:.02f} s")
    else:
        print(f"{msg} {duration*1000:.02f} ms")

class AudioStream:
    def __init__(self):
        self.q = queue.Queue(maxsize=40)
        self.generator = load_csm_1b()
        self.is_prebuffering = True
        self.prebuffer_size = BUFFER_SIZE
        self.generated_segments = []
        self.audio = []
        self.frames_in_queue = 0
        self.stream = sd.OutputStream(
            samplerate=self.generator.sample_rate,
            blocksize=FRAME_SIZE,
            channels=1,
            callback=self.callback
        )
        self.stream.start()

    def speak(self, text):
        self.start_time = time()
        self.text = text
        print(f"generated_segments shape is: {len(self.generated_segments)}")
        frame_iter = self.generator.generate_stream(
            self.text,
            speaker=0,
            context=self.generated_segments)

        for frame in frame_iter:
            self.audio.append(frame)
            cpu_frame = frame.detach().cpu()
            self.q.put(cpu_frame)      

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
            if torch.all(new_frame == 0): # EOS
                print("EOS detected in stream_test!")
                full_audio = torch.stack(self.audio).view(-1)
                self.generated_segments.append(Segment(text=self.text, speaker=0, audio=full_audio))
                self.generator.update_ctx_tokens(self.generated_segments)
                self.audio = []
            else:
                outdata[:] = new_frame.numpy()
        except queue.Empty:
            outdata[:] = 0
            self.is_prebuffering = True

def main():
    stream_handler = AudioStream()

    text = "Hello, world! Testing to stream the MIMI decoder. Please type in some text and CSM will generate the audio."

    while [ 1 ]:
        for t in text.replace("\n", "").split('.'):
            print("Sentence: ", t)
            stream_handler.speak(t)
        print("Type text to and hit Enter key to start streaming TTS... (empty string to quit)")
        print()
        text = input()
        if not text:
            break

if __name__ == "__main__":
    main() 