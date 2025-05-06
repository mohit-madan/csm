import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd
import io
import wave
import time
import logging
from queue import Queue
from threading import Event, Thread

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WS_URL = "wss://sei-ai--csm-tts-websocket-modal-websocket-tts-app.modal.run/ws"

class AudioPlayer:
    def __init__(self):
        self.queue = Queue()
        self.stop_event = Event()
        self.stream = None
        self.sample_rate = None
        self.channels = None
        self.playback_thread = None
        self.is_playing = False
        self.end_of_audio = False
        
    def start(self, sample_rate, channels):
        self.sample_rate = sample_rate
        self.channels = channels
        self.stop_event.clear()
        self.is_playing = True
        self.end_of_audio = False
        
        # Start playback in a separate thread
        self.playback_thread = Thread(target=self._play_audio)
        self.playback_thread.daemon = True
        self.playback_thread.start()
        
    def add_chunk(self, chunk):
        if chunk and len(chunk) > 0:
            self.queue.put(chunk)
            
    def stop(self):
        self.stop_event.set()
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
        self.is_playing = False
        
    def signal_end_of_audio(self):
        self.end_of_audio = True
        
    def _play_audio(self):
        """Playback thread that consumes audio chunks from the queue"""
        logger.info(f"Starting audio playback thread with sample rate: {self.sample_rate}Hz, channels: {self.channels}")
        
        # Calculate blocksize - make it smaller than chunk size for better responsiveness
        # Typical chunk size is 7680 bytes, which is 1920 frames for 16-bit mono audio
        # We'll use a smaller blocksize for smoother playback
        blocksize = 1024  # Smaller block size for more frequent callbacks
        
        # Callback to feed audio data to the sound device
        def audio_callback(outdata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
                
            if self.stop_event.is_set():
                raise sd.CallbackStop
                
            # Calculate how many bytes we need
            bytes_per_frame = 4  # float32
            bytes_needed = frames * self.channels * bytes_per_frame
            
            # Try to get enough data from the queue
            audio_data = bytearray()
            try:
                # Get first chunk with timeout to avoid hanging if queue is empty
                chunk = self.queue.get(timeout=0.1)
                audio_data.extend(chunk)
                self.queue.task_done()
                
                # Get more chunks if needed
                while len(audio_data) < bytes_needed and not self.queue.empty():
                    chunk = self.queue.get_nowait()
                    audio_data.extend(chunk)
                    self.queue.task_done()
            except Exception as e:
                # Queue empty or timeout
                if len(audio_data) == 0:
                    if self.end_of_audio and self.queue.empty():
                        logger.info("End of audio and queue empty, stopping playback")
                        raise sd.CallbackStop
                    else:
                        # No data available, output silence
                        outdata.fill(0)
                        return
            
            # Process available data
            if len(audio_data) >= bytes_needed:
                # We have enough data
                data_arr = np.frombuffer(audio_data[:bytes_needed], dtype=np.float32)
                outdata[:] = data_arr.reshape(-1, self.channels)
                
                # Put any remaining data back in the queue for next callback
                if len(audio_data) > bytes_needed:
                    self.queue.put(bytes(audio_data[bytes_needed:]))
            else:
                # Partial data - use what we have and pad with zeros
                data_arr = np.frombuffer(audio_data, dtype=np.float32)
                frames_available = len(data_arr) // self.channels
                outdata[:frames_available] = data_arr.reshape(-1, self.channels)
                outdata[frames_available:].fill(0)
                logger.debug(f"Partial data: {len(audio_data)} bytes, needed {bytes_needed}")
        
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                blocksize=blocksize,
                dtype='float32'
            ) as self.stream:
                # Wait until the stop event is set
                while not self.stop_event.is_set():
                    # Check if we should stop because we've reached the end of audio
                    if self.end_of_audio and self.queue.empty():
                        logger.info("End of audio reached and queue is empty")
                        # Give time for final audio to play
                        time.sleep(0.5)
                        break
                    time.sleep(0.1)  # Check periodically
        except Exception as e:
            logger.error(f"Error in audio playback: {e}")
        finally:
            logger.info("Audio playback thread ended")
            self.is_playing = False

async def play_streamed_audio(websocket):
    # Create audio player
    player = AudioPlayer()
    
    try:
        # Wait for header
        header = await websocket.recv()
        
        # Skip any zero-length chunks from previous requests
        while len(header) == 0:
            header = await websocket.recv()

        if len(header) != 44:  # Standard WAV header size
            logger.error(f"Invalid WAV header received! Got length: {len(header)}")
            return

        # Parse WAV header
        with io.BytesIO(header) as f:
            with wave.open(f, "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                
        logger.info(f"Audio format: {sample_rate}Hz, {channels} channels, {sampwidth}-byte samples")
        
        # Pre-buffer audio data
        prebuffer_chunks = 5
        prebuffered = 0
        
        logger.info(f"Pre-buffering {prebuffer_chunks} chunks...")
        chunks = []
        while prebuffered < prebuffer_chunks:
            chunk = await websocket.recv()
            if len(chunk) == 0:
                logger.info("End of audio received during pre-buffering")
                break
            chunks.append(chunk)
            prebuffered += 1
            logger.info(f"Pre-buffered chunk {prebuffered}/{prebuffer_chunks} ({len(chunk)} bytes)")
        
        # Start the audio player
        player.start(sample_rate, channels)
        
        # Add pre-buffered chunks to the player
        for chunk in chunks:
            player.add_chunk(chunk)
        
        # Continue receiving chunks
        chunks_received = prebuffered
        while True:
            chunk = await websocket.recv()
            chunks_received += 1
            
            if len(chunk) == 0:
                logger.info(f"End of audio signal received. Total chunks: {chunks_received}")
                player.signal_end_of_audio()
                break
                
            player.add_chunk(chunk)
            
            # Log progress periodically
            if chunks_received % 10 == 0:
                logger.info(f"Received {chunks_received} chunks")
        
        # Wait for the audio to finish playing
        while player.is_playing:
            await asyncio.sleep(0.1)
            
    except websockets.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Error in audio streaming: {e}")
    finally:
        # Stop the player if it's still running
        player.stop()

async def tts_client():
    logger.info("Connecting to TTS server...")
    try:
        async with websockets.connect(WS_URL) as websocket:
            logger.info("Connected! Ready for text input.")
            while True:
                text = input("Enter text (or 'quit' to exit): ")
                if text.lower() == 'quit':
                    break
                
                logger.info(f"Requesting TTS for: \"{text}\"")
                await websocket.send(json.dumps({"text": text}))
                await play_streamed_audio(websocket)
    except websockets.ConnectionError:
        logger.error("Failed to connect to the TTS server. Please check the server URL and your internet connection.")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    print("TTS Streaming Client")
    print("--------------------")
    asyncio.run(tts_client())