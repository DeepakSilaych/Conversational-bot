import asyncio
import websockets
import json
from typing import Callable, Optional


class AssemblyAITranscriber:
    def __init__(self, api_key: str, language: str = "hi"):
        print(f"[ASSEMBLY_STT] Initializing with language: {language}")
        self.api_key = api_key
        self.language = language
        self.ws = None
        self._running = False
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self._session_id = None
        
        # Audio buffer for accumulating chunks
        self._audio_buffer = bytearray()
        self._min_chunk_size = 1600  # 50ms at 16kHz = 800 samples * 2 bytes = 1600 bytes
        
        # AssemblyAI v3 Universal-Streaming endpoint
        self.ws_url = "wss://streaming.assemblyai.com/v3/ws"
        
    async def start(self, on_transcript: Callable[[str, bool], asyncio.Future]):
        print("[ASSEMBLY_STT] Starting Universal-Streaming transcription...")
        self._main_loop = asyncio.get_running_loop()
        self._running = True
        
        try:
            # Build URL with parameters
            url = f"{self.ws_url}?sample_rate=16000&encoding=pcm_s16le"
            
            # Authentication in header
            headers = {
                "Authorization": self.api_key
            }
            
            self.ws = await websockets.connect(url, extra_headers=headers)
            print("[ASSEMBLY_STT] Connected to v3 API")
            
            # Listen for messages
            async def receive_loop():
                try:
                    while self._running and self.ws:
                        message = await self.ws.recv()
                        
                        # v3 sends text messages as JSON
                        if isinstance(message, str):
                            data = json.loads(message)
                            msg_type = data.get("type")
                            
                            if msg_type == "Begin":
                                self._session_id = data.get("id")
                                print(f"[ASSEMBLY_STT] Session started: {self._session_id}")
                                
                            elif msg_type == "Turn":
                                transcript = data.get("transcript", "").strip()
                                if transcript and self._main_loop:
                                    is_final = data.get("is_final", False)
                                    
                                    print(f"[ASSEMBLY_STT] {'Final' if is_final else 'Partial'}: '{transcript}'")
                                    
                                    asyncio.run_coroutine_threadsafe(
                                        on_transcript(transcript, is_final),
                                        self._main_loop
                                    )
                                
                            elif msg_type == "Termination":
                                duration = data.get("audio_duration_seconds", 0)
                                print(f"[ASSEMBLY_STT] Session terminated after {duration}s")
                                self._running = False
                                
                        # Handle binary messages if any
                        else:
                            print(f"[ASSEMBLY_STT] Received binary message: {len(message)} bytes")
                            
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"[ASSEMBLY_STT] Connection closed: {e}")
                    self._running = False
                except Exception as e:
                    print(f"[ASSEMBLY_STT] Error in receive loop: {e}")
                    self._running = False
            
            # Start receive loop
            receive_task = asyncio.create_task(receive_loop())
            
            # Keep connection alive
            while self._running:
                await asyncio.sleep(0.1)
                
            await receive_task
            
        except Exception as e:
            print(f"[ASSEMBLY_STT] Connection error: {e}")
            self._running = False
        finally:
            print("[ASSEMBLY_STT] Cleaning up...")
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass

    def feed_audio(self, data: bytes):
        """Feed audio data to AssemblyAI with buffering"""
        if self._running and self.ws and self._main_loop:
            # Add to bufferfinish(self):
            self._audio_buffer.extend(data)
            
            # Send chunks of at least 50ms (1600 bytes)
            while len(self._audio_buffer) >= self._min_chunk_size:
                chunk = bytes(self._audio_buffer[:self._min_chunk_size])
                self._audio_buffer = self._audio_buffer[self._min_chunk_size:]
                
                asyncio.run_coroutine_threadsafe(
                    self._send_audio(chunk),
                    self._main_loop
                )
    
    async def _send_audio(self, audio_data: bytes):
        """Send raw audio data to AssemblyAI WebSocket"""
        try:
            if self.ws and self._running:
                # Send raw bytes directly
                await self.ws.send(audio_data)
        except websockets.exceptions.ConnectionClosedError:
            print(f"[ASSEMBLY_STT] Connection closed, cannot send audio")
            self._running = False
        except Exception as e:
            print(f"[ASSEMBLY_STT] Error sending audio: {e}")
    
    def finish(self):
        """Stop transcription and close connection"""
        print("[ASSEMBLY_STT] Finishing...")
        
        # Flush remaining audio buffer
        if len(self._audio_buffer) > 0 and self._running and self.ws and self._main_loop:
            remaining = bytes(self._audio_buffer)
            self._audio_buffer.clear()
            asyncio.run_coroutine_threadsafe(
                self._send_audio(remaining),
                self._main_loop
            )
        
        self._running = False
        if self.ws and self._main_loop:
            # Send terminate message as JSON
            terminate_msg = json.dumps({"type": "Terminate"})
            asyncio.run_coroutine_threadsafe(
                self._send_and_close(terminate_msg),
                self._main_loop
            )
    
    async def _send_and_close(self, message: str):
        """Send final message and close connection"""
        try:
            if self.ws:
                await self.ws.send(message)
                await asyncio.sleep(0.5)  # Give time for termination
                await self.ws.close()
        except Exception as e:
            print(f"[ASSEMBLY_STT] Error during close: {e}")