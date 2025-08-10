import asyncio
import time
from typing import Callable, Optional
from google.cloud import speech
from google.oauth2 import service_account


class GoogleTranscriber:
    def __init__(self, credentials_path: str, language: str = "en-IN"):
        self.language = language
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=self.language,
                enable_automatic_punctuation=True,
            ),
            interim_results=True,
            single_utterance=False
        )

        self.client = speech.SpeechClient(credentials=service_account.Credentials.from_service_account_file(credentials_path))
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

    def _generator(self):
        while self._running:
            data = self.audio_queue.get_nowait()
            if data is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=data)

    async def start(self, on_transcript: Callable[[str, bool], asyncio.Future]):
        self._main_loop = asyncio.get_running_loop()
        self._running = True

        def request_gen():
            while self._running:
                try:
                    data = self.audio_queue.get_nowait()
                    if data is None:
                        break
                    yield speech.StreamingRecognizeRequest(audio_content=data)
                except asyncio.QueueEmpty:
                    time.sleep(0.01)

        responses = self.client.streaming_recognize(self.streaming_config, requests=request_gen())
        for response in responses:
            for result in response.results:
                text = result.alternatives[0].transcript.strip()
                if text and self._main_loop:
                    asyncio.run_coroutine_threadsafe(
                        on_transcript(text, result.is_final), self._main_loop
                    )
        self._running = False

    def feed_audio(self, data: bytes):
        if self._running:
            self.audio_queue.put_nowait(data)

    def finish(self):
        self._running = False
        self.audio_queue.put_nowait(None)
