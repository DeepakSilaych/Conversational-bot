import os
import asyncio
import json
import time
import threading
from typing import Callable, Optional
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents


class DeepgramTranscriber:
    """
    Streams raw PCM audio to Deepgram for real-time transcription.
    """

    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        interim: bool = True,
        language: str = "en-IN",
    ):
        self.sample_rate = sample_rate
        self.interim = interim
        self.language = language

        # Deepgram client and placeholder for WebSocket connection
        self.dg_client = DeepgramClient(api_key)
        self.dg_conn = None

        # Will store asyncio loop reference once `start()` is called
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        # Mic/state flags
        self._running = False

        self._utterance_in_progress = False
        self._utterance_start_ts = None


    def _handle_transcript(self, evt, result, on_transcript: Callable[[str, bool], asyncio.Future]):
        """
        Internal callback invoked by Deepgram on each transcript event.

        - Extracts the top alternative transcript.
        - If non-empty, schedules `on_transcript(text, is_final)` on the asyncio loop.
        - On final segments, saves buffered PCM to a WAV file.
        """
        try:
            alt = result.channel.alternatives[0]
            text = alt.transcript.strip()
            is_final = getattr(result, "is_final", False)
            # words = getattr(alt, "words", [])
        except Exception:
            # Malformed payload; skip
            return

        if not text:
            # No words recognized yet
            return

        # print(result.channel.alternatives[0]) 
        # print(result)

        # If this is an interim (is_final=False) and we haven’t stamped a start yet:
        now = time.perf_counter()
        if not is_final and not self._utterance_in_progress:
           # First time we see non‐empty interim: mark start
            print(f"[{time.perf_counter():.3f}] [STT] first interim: “{text}”")
            self._utterance_in_progress = True
            self._utterance_start_ts = time.perf_counter()

        # When this is final, record end time and compute latency
        if is_final and self._utterance_in_progress:
            print(f"[{time.perf_counter():.3f}] [STT] final transcript: “{text}”")
            utterance_end_ts = time.perf_counter()
            ut_latency = utterance_end_ts - self._utterance_start_ts
            print(f"[STT] Utterance transcription latency: {ut_latency:.3f} sec")
            # Reset for next utterance
            self._utterance_in_progress = False
            self._utterance_start_ts = None

        # Forward transcript to user callback in the asyncio loop
        if self._main_loop and on_transcript:
            try:
                asyncio.run_coroutine_threadsafe(on_transcript(text, is_final), self._main_loop)
            except Exception:
                pass

    async def start(self, on_transcript: Callable[[str, bool], asyncio.Future]):
        """
        Connects to Deepgram and begins streaming.

        If use_mic=False, you must call `feed_audio()` to send PCM bytes.
        Otherwise, internal mic capture (in `_send_audio()`) will send audio.

        Args:
            on_transcript: async callback receiving (text: str, is_final: bool).
        """
        if self._running:
            raise RuntimeError("DeepgramTranscriber is already running.")
        self._running = True

        # Create a new Deepgram WebSocket connection
        try:
            self.dg_conn = self.dg_client.listen.websocket.v("1")
        except Exception as e:
            raise RuntimeError(f"[STT] Failed to initialize Deepgram client: {e}")

        self.dg_conn.on(
            LiveTranscriptionEvents.Transcript,
            lambda evt, result: self._handle_transcript(evt, result, on_transcript),
        )

        # Store asyncio loop so we can schedule callbacks
        self._main_loop = asyncio.get_running_loop()

        # Build LiveOptions
        opts = LiveOptions(
            punctuate=True,
            smart_format=True,
            model="nova-3",
            encoding="linear16",
            sample_rate=self.sample_rate,
            channels=1,
            interim_results=self.interim,
            vad_events=True,
            endpointing=200,
            language=self.language,
        )
        opts.language_detection=True
        try:
            success = await asyncio.to_thread(self.dg_conn.start, opts)
        except Exception as e:
            raise RuntimeError(f"[STT] Deepgram start failed: {e}")

        if not success:
            raise RuntimeError("[STT] Deepgram connection failed to start.")

        # Spawn keep-alive pings to prevent WebSocket timeout
        def _keepalive():
            while self._running:
                time.sleep(5)
                try:
                    if self.dg_conn:
                        self.dg_conn.send(json.dumps({"type": "KeepAlive"}))
                except Exception:
                    break

        threading.Thread(target=_keepalive, daemon=True).start()
        await asyncio.Future()

    def feed_audio(self, data: bytes):
        """
        Send raw PCM16 bytes (16 kHz mono) directly to Deepgram.
        
        Args:
            data: raw PCM chunks, in int16 little-endian bytes.
        """

        if self.dg_conn:
            try:
                self.dg_conn.send(data)
            except Exception:
                pass

    def finish(self):
        """
        Gracefully closes the Deepgram WebSocket connection.
        """
        self._running = False
        if self.dg_conn:
            try:
                self.dg_conn.finish()
            except Exception:
                pass