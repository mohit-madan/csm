import websocket
import json
import io
import sounddevice as sd
import soundfile as sf
import numpy as np
import asyncio
import websockets
import time

# List to store all received messages
received_messages = []
stream = None
first_chunk = True

# Set these to match your TTS server's output!
SAMPLE_RATE = 24000  # or 24000, etc.
CHANNELS = 1  # or 2 for stereo
DTYPE = "float32"  # or 'int16', etc.


# def on_message(ws, message):
#     global stream, first_chunk
#     if isinstance(message, bytes):
#         if first_chunk:
#             # First chunk: parse WAV header and get format/data
#             with io.BytesIO(message) as wav_io:
#                 data, samplerate = sf.read(wav_io, dtype="float32")
#                 channels = data.shape[1] if data.ndim > 1 else 1
#                 # Start the stream
#                 stream = sd.OutputStream(
#                     samplerate=samplerate, channels=channels, dtype="float32"
#                 )
#                 stream.start()
#                 stream.write(data)
#             first_chunk = False
#         else:
#             # Subsequent chunks: raw PCM, play directly
#             # Assume float32, 1 channel, 24000 Hz (from previous chunk)
#             audio_array = np.frombuffer(message, dtype="float32")
#             stream.write(audio_array)
#     else:
#         print("Received (text):", message)
#         received_messages.append(message)


# def on_error(ws, error):
#     print("Error:", error)


# def on_close(ws, close_status_code, close_msg):
#     print("Closed")
#     if stream:
#         stream.stop()
#         stream.close()
#     # Print the full response at the end
#     full_response = "".join(received_messages)
#     print("Full response:", full_response)


# def on_open(ws):
#     # Send the text message as JSON after the connection is established
#     payload = {"text": "Hi, I am a streaming response from sesame. How are you?"}
#     ws.send(json.dumps(payload))
#     print("Message sent:", payload)


# # WebSocket URL without query parameters
# url = "wss://sei-ai--csm-tts-modal-websocket-tts-app-dev.modal.run/ws"

# # Create the WebSocket app and connect
# ws = websocket.WebSocketApp(
#     url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open
# )

# # Run the WebSocket client
# ws.run_forever()


async def test_ws():
    uri = "wss://sei-ai--csm-tts-modal-websocket-tts-app-dev.modal.run/ws"
    audio_chunks = []
    header = None

    script_start_time = time.time()
    print(f"[{0.000:.3f}s] Script started.")

    try:
        connect_start_time = time.time()
        async with websockets.connect(uri) as websocket:
            connect_end_time = time.time()
            print(
                f"[{connect_end_time - script_start_time:.3f}s] WebSocket connection setup complete "
                f"(setup time: {connect_end_time - connect_start_time:.3f}s)"
            )

            await websocket.send(
                '{"text": "Hi, I am a streaming response from sesame. How are you?"}'
            )
            send_time = time.time()
            print(
                f"[{send_time - script_start_time:.3f}s] Message sent to server "
                f"(send after connect: {send_time - connect_end_time:.3f}s)"
            )

            first_token_time = None
            while True:
                try:
                    chunk = await websocket.recv()
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed by server.")
                    break
                if not chunk:
                    break
                if header is None:
                    header = chunk
                    first_token_time = time.time()
                    print(
                        f"[{first_token_time - script_start_time:.3f}s] First token received "
                        f"(time from send: {first_token_time - send_time:.3f}s)"
                    )
                    continue
                audio_chunks.append(np.frombuffer(chunk, dtype=np.float32))
            end_of_stream_time = time.time()
            print(
                f"[{end_of_stream_time - script_start_time:.3f}s] End of stream "
                f"(time from first token: {end_of_stream_time - first_token_time:.3f}s)"
            )
    except KeyboardInterrupt:
        print("Interrupted! Saving audio received so far...")
    finally:
        if audio_chunks:
            audio = np.concatenate(audio_chunks)
            print(f"Saving {audio.shape[0]} samples to output.wav")
            sf.write("output.wav", audio, 24000, subtype="FLOAT")
            print("Saved to output.wav")


try:
    asyncio.run(test_ws())
except KeyboardInterrupt:
    # This will only trigger if you interrupt before the async function starts
    print("Interrupted before streaming started.")
