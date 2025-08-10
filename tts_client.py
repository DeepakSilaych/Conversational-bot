import asyncio
from typing import Optional, AsyncGenerator
from fastapi import WebSocket
import time
import os

class TTSClient:
    def __init__(
        self, 
        provider: str = "elevenlabs",
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        azure_key: Optional[str] = None,
        azure_region: Optional[str] = None,
        optimize_latency: int = 2
    ):
        self.provider = provider.lower()
        
        if self.provider == "elevenlabs":
            from elevenlabs.client import AsyncElevenLabs
            self.client = AsyncElevenLabs(api_key=api_key)
            self.voice_id = voice_id
            self.optimize_latency = optimize_latency
            
        elif self.provider == "azure":
            import azure.cognitiveservices.speech as speechsdk
            
            if not azure_key:
                azure_key = os.getenv("AZURE_SPEECH_KEY")
            if not azure_region:
                azure_region = os.getenv("AZURE_REGION", "centralindia")

            self.azure_config = speechsdk.SpeechConfig(
                subscription=azure_key,
                region=azure_region
            )
            # Azure voices for Indian languages
            # For code-mixed Hindi+English, Swara is the best
            self.azure_voice = "hi-IN-SwaraNeural" # male voice, excellent for code-mixing
            # OR use "hi-IN-SwaraNeural" for female voice
            # "hi-IN-AaravNeural", "hi-IN-KunalNeural", "hi-IN-RehaanNeural, "hi-IN-ArjunNeural"

    async def speak_sentence(
        self,
        sentence: str,
        ws: WebSocket,
        chosen_lang_el: str,
        last_user_end_ts: Optional[float]
    ) -> AsyncGenerator[bytes, None]:
        """Original method signature - returns async generator"""
        
        if self.provider == "elevenlabs":
            # ElevenLabs implementation
            backoff = 1.0
            max_backoff = 5.0

            while True:
                try:
                    audio_iter = self.client.text_to_speech.stream(
                        text=sentence,
                        voice_id=self.voice_id,
                        model_id="eleven_flash_v2_5",
                        optimize_streaming_latency=self.optimize_latency,
                        language_code=chosen_lang_el,
                        output_format="pcm_44100",
                            voice_settings={
                                "stability": 0.85,  # High for consistent Minraj voice
                                "similarity_boost": 0.90,  # Very high to maintain voice characteristics
                                "use_speaker_boost": True  # Enhanced clarity
                             }
                    )
                    break
                except Exception as e:
                    print(f"[TTS] Transient error: {e}, retrying in {backoff} seconds.")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)

            #print(f"[TTS] Streaming started for: \"{sentence[:50]}...\"")
            return audio_iter
            
        else:
            # Azure implementation
            return self._azure_stream(sentence, chosen_lang_el)
    
    async def _azure_stream(self, sentence: str, chosen_lang_el: str) -> AsyncGenerator[bytes, None]:
        """Azure TTS with async synthesis"""
        import azure.cognitiveservices.speech as speechsdk
        
        voice_name = self.azure_voice
        print(f"[AZURE_TTS] Using voice: {voice_name}")
        
        # Configure output format
        self.azure_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw44100Hz16BitMonoPcm
        )
        
        # SSML
        ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='hi-IN'>
            <voice name='{voice_name}'>
                <prosody rate='1.0' pitch='0%'>
                    {sentence}
                </prosody>
            </voice>
        </speak>"""
        
        # Create synthesizer without audio config to get audio data directly
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.azure_config,
            audio_config=None
        )
        
        # Perform synthesis in a thread to avoid blocking
        result = await asyncio.to_thread(
            synthesizer.speak_ssml_async(ssml).get
        )
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Get the audio data
            audio_data = result.audio_data
            
            if audio_data and len(audio_data) > 0:
                # Yield audio in chunks
                chunk_size = 8192
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i+chunk_size]
                    if chunk:
                        yield chunk
            else:
                print("[AZURE_TTS] No audio data received")
        else:
            print(f"[AZURE_TTS] Synthesis failed: {result.reason}")
            if result.cancellation_details:
                print(f"[AZURE_TTS] Error: {result.cancellation_details.error_details}")
                print(f"[AZURE_TTS] Did you update the speech resource key and region?")