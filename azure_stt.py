import asyncio
import time
import json
import azure.cognitiveservices.speech as speechsdk
from typing import Callable, Optional


class AzureTranscriber:
    def __init__(self, key: str, region: str, language: str = "en-IN"):
        print(f"[AZURE_STT] Initializing with language: {language}, region: {region}")
        self.language = language
        self.speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        self.speech_config.speech_recognition_language = language
        
        # CRITICAL: Enable detailed output format to get confidence scores
        self.speech_config.output_format = speechsdk.OutputFormat.Detailed
        
        # Enable word-level timestamps for word-level confidence
        self.speech_config.request_word_level_timestamps()
        
        # Enable detailed logging for debugging
        self.speech_config.set_property(
            speechsdk.PropertyId.Speech_LogFilename, 
            "azure_speech.log"
        )
        
        # Configure audio format to match your input
        # Your client sends 16kHz mono PCM
        self.audio_stream = speechsdk.audio.PushAudioInputStream(
            stream_format=speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000,
                bits_per_sample=16,
                channels=1
            )
        )
        self.audio_config = speechsdk.audio.AudioConfig(stream=self.audio_stream)
        self.recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, 
            audio_config=self.audio_config
        )

        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._connection_id = None  # Add connection ID for validation

    async def start(self, on_transcript: Callable[[str, bool], asyncio.Future], connection_id: str = None):
        print("[AZURE_STT] Starting recognition...")
        self._main_loop = asyncio.get_running_loop()
        self._running = True
        self._connection_id = connection_id if connection_id else "unknown"  # Store connection ID

        def recognized_handler(evt):
            """Handle final recognition results with simple confidence filtering"""
            # CHECK IF STILL RUNNING
            if not self._running:
                print(f"[AZURE_STT] Ignoring recognized event - not running")
                return
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = evt.result.text
                
                # Filter out empty transcripts
                if not text or not text.strip():
                    #print("[AZURE_STT] Ignoring empty transcript")
                    return
                
                confidence = 1.0  # Default if JSON parsing fails
                
                # Access JSON directly from result
                if hasattr(evt.result, 'json') and evt.result.json:
                    try:
                        result_data = json.loads(evt.result.json)
                        #print(f"[AZURE_STT DEBUG] JSON structure: {json.dumps(result_data, indent=2)[:500]}...")
                        
                        # Check if we have NBest results
                        if 'NBest' in result_data and len(result_data['NBest']) > 0:
                            best_result = result_data['NBest'][0]
                            confidence = best_result.get('Confidence', 1.0)
                            
                            #print(f"[AZURE_STT] Text: '{text}' | Confidence: {confidence}")
                            
                            # Simple approach: different thresholds based on length
                            cleaned = text.strip().rstrip('.').lower()
                            word_count = len(cleaned.split())
                            
                            # Use lower threshold for short utterances
                            if word_count <= 2:
                                threshold = 0.65
                            else:
                                threshold = 0.6
                            
                            # Apply confidence filter
                            if confidence < threshold:
                               # print(f"[AZURE_STT] Ignoring low confidence ({confidence:.2f} < {threshold}): '{text}'")
                                return
                            
                            # Still filter single letters (except 'I' and 'a')
                            if len(cleaned) == 1 and cleaned.isalpha() and cleaned not in ['i', 'a']:
                                #print(f"[AZURE_STT] Ignoring single letter: '{text}'")
                                return
                                
                    except Exception as e:
                        print(f"[AZURE_STT] Error parsing JSON: {e}")
                else:
                    print("[AZURE_STT] No JSON result available")
                    
                # print(f"[AZURE_STT] Final text accepted: '{text}'")
                if self._running and self._main_loop:  # Double check
                    future = asyncio.run_coroutine_threadsafe(
                        on_transcript(text, True),
                        self._main_loop
                    )

        def recognizing_handler(evt):
            # CHECK IF STILL RUNNING
            if not self._running:
                print(f"[AZURE_STT] Ignoring recognizing event - not running")
                return
            text = evt.result.text.strip()
            if text and self._main_loop and self._running:
               # print(f"[AZURE_STT] Interim text: '{text}'")
                asyncio.run_coroutine_threadsafe(
                    on_transcript(text, False), 
                    self._main_loop
                )

        def canceled_handler(evt):
           # print(f"[AZURE_STT] CANCELED: Reason={evt.reason}")
            if evt.reason == speechsdk.CancellationReason.Error:
                print(f"[AZURE_STT] Error details: {evt.error_details}")
                # Don't stop on error, just log it
            elif evt.reason == speechsdk.CancellationReason.EndOfStream:
                print("[AZURE_STT] End of stream reached")
                self._running = False
                done.set()

        def session_stopped_handler(evt):
            print("[AZURE_STT] Session stopped event")
            self._running = False
            done.set()

        # Connect handlers
        self.recognizer.recognizing.connect(recognizing_handler)
        self.recognizer.recognized.connect(recognized_handler)
        self.recognizer.canceled.connect(canceled_handler)
        self.recognizer.session_stopped.connect(session_stopped_handler)

        done = asyncio.Event()

        try:
            #print("[AZURE_STT] Starting continuous recognition...")
            self.recognizer.start_continuous_recognition()
            # print("[AZURE_STT] Recognition started, waiting...")
            await done.wait()
        except Exception as e:
            #print(f"[AZURE_STT] Error during recognition: {e}")
            self._running = False
        finally:
            #Ensure handlers are disconnected
            self._disconnect_handlers()


    def _disconnect_handlers(self):
        """Disconnect all event handlers"""
        try:
            print("[AZURE_STT] Disconnecting event handlers...")
            self.recognizer.recognizing.disconnect_all()
            self.recognizer.recognized.disconnect_all()
            self.recognizer.canceled.disconnect_all()
            self.recognizer.session_stopped.disconnect_all()
            print("[AZURE_STT] Event handlers disconnected")
        except Exception as e:
            print(f"[AZURE_STT] Error disconnecting handlers: {e}")
            
    def feed_audio(self, data: bytes):
        """Feed audio data to Azure STT"""
        if self._running:
            try:
                # Azure expects the audio data as-is
                self.audio_stream.write(data)
            except Exception as e:
                print(f"[AZURE_STT] Error feeding audio: {e}")
        else:
            print("[AZURE_STT] Not running, ignoring audio")

    def finish(self):
        """Clean shutdown of Azure STT"""
        print(f"[AZURE_STT] Finishing (connection: {self._connection_id})...")
        if self._running:
            self._running = False
            try:
                # Stop recognition first
                self.recognizer.stop_continuous_recognition()
                print("[AZURE_STT] Recognition stopped")
                
                # Disconnect handlers
                self._disconnect_handlers()
                
                # Close audio stream
                self.audio_stream.close()
                print("[AZURE_STT] Audio stream closed")
                
                print("[AZURE_STT] Stopped successfully")
            except Exception as e:
                print(f"[AZURE_STT] Error during shutdown: {e}")