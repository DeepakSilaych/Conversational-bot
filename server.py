import os
import asyncio 
import time
import threading
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket
from fastapi import UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from deepgram_stt import DeepgramTranscriber
from assemblyai_stt import AssemblyAITranscriber
from llm_client import LLMClient
from tts_client import TTSClient
from dotenv import load_dotenv
from typing import Optional, AsyncGenerator, Tuple, Dict, Set
from supabase import create_client
from datetime import datetime, timezone
import json
import subprocess
import re
import atexit
from shared import db_executor


# Loads environment variables from .env file

load_dotenv()
DG_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ASSEMBLY_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
#VOICE_ID = os.getenv("ELEVEN_VOICE_ID")
#VOICE_ID = os.getenv("ELEVEN_VOICE_ID_MANI")
VOICE_ID = os.getenv("ELEVEN_VOICE_ID_MINRAJ")
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
# Add Azure TTS configuration
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")
USE_AZURE_TTS = os.getenv("USE_AZURE_TTS", "true").lower() == "true"

# Debug prints
#print(f"[DEBUG] USE_AZURE_TTS: {USE_AZURE_TTS}")
#print(f"[DEBUG] AZURE_KEY exists: {bool(AZURE_SPEECH_KEY)}")
#print(f"[DEBUG] AZURE_REGION: {AZURE_REGION}")

# Initializes FastAPI app with CORS enabled for any origin

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_event():
    print("[SHUTDOWN] Shutting down database thread pool...")
    db_executor.shutdown(wait=True)
    print("[SHUTDOWN] Database thread pool shut down")

# Serves the main HTML and static assets

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.ico")

# Initialization of Supabase client for storage and database operations

supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Endpoint to accept uploaded audio and store in Supabase

@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    lead_id: str = Form(...)   # received from the client
):
    # 1) Saving Incoming file to a temp file with original extension (.webm generally) in the project folder
    base_dir  = os.path.dirname(__file__)
    ext       = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(base_dir, f"temp_audio{ext}")
    data      = await file.read()
    with open(temp_path, "wb") as f:
        f.write(data)

    # 2) Converting non-wav formats to 48kHz mono WAV for consistency (more user friendly)
    if ext == ".wav":
        wav_path = temp_path
    else:
        wav_path = os.path.join(base_dir, f"temp_audio.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_path,
            "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
            wav_path
        ], check=True)
        os.remove(temp_path)

    print(f"WAV File: {wav_path}")

    # 3) Uploading to Supabase Storage in bucket "conversation_recordings"
    timestamp = int(time.time())
    key = f"{lead_id}_{timestamp}.wav"
    bucket    = supabase_client.storage.from_("conversation-recordings")
    print(f"[Bucket] {bucket}")
    with open(wav_path, "rb") as f:
        res = bucket.upload(key, f, {"content-type": "audio/wav"})
    print(f"[res] {res}")

    # 4) Obtaining public URL and update the leads table
    public_url = bucket.get_public_url(key)
    print(f"Public URL: {public_url}")
    upd = supabase_client.table("leads") \
        .update({"recorded_conv_link": public_url}) \
        .eq("id", lead_id) \
        .execute()

    return {"status": "ok", "url": public_url}


RESPONSE_TIMEOUT = 30 
# WebSocket endpoint to handle real-time STT → LLM → TTS loop

# Mapping from lead_id to set of WebSocket connections
lead_clients: Dict[str, Set[WebSocket]] = {}

# ADD THIS RIGHT AFTER IT:
active_connections: Dict[str, WebSocket] = {}  # connection_id -> WebSocket

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    print(f"[DEBUG] WebSocket connection attempt")
    await ws.accept()
    print(f"[DEBUG] WebSocket accepted")
    lead_id = None  # Initialize early
    stt = None      # Initialize early
    llm = None      # Initialize early
    # 1. Add a unique connection ID at the start of websocket_endpoint
    connection_id = str(uuid.uuid4())
    print(f"[WS] New connection: {connection_id}")
    active_connections[connection_id] = ws
    # First message includes conversation context IDs (required)

    msg = await ws.receive()
    print(f"[DEBUG] First message received: {msg}")
    if msg.get("text"):
        print("RAW PAYLOAD:", repr(msg["text"]))
        data = json.loads(msg["text"])
        if data.get("type") == "conversation_info":
            lead_id = data["lead_id"]
            # Validate lead_id
            if not lead_id or lead_id == "null" or lead_id == "undefined":
                print(f"Invalid lead ID received: {lead_id}")
                await ws.close()
                return
            print("Lead ID recieved from frontend: ", lead_id)
            if not lead_id:
                print("Did not receive Lead ID from frontend, hence aborting...")
                await ws.close()
                return
            if lead_id not in lead_clients:
                lead_clients[lead_id] = set()
                lead_clients[lead_id].add(ws)
            for client in lead_clients[lead_id]:
                print(f"code for another websocket")
                # if client != ws:
                #     await safe_send_json(client,{
                #                 "type": "text",
                #                 "data": {
                #                     "message": "Hi you there",
                #                     "viewType": 1,
                #                     "viewTypeName": "simpleIncomingText",
                #                     "from": "agent",
                #                     "showMessage": True,
                #                     "time": int(time.time() * 1000)
                #                 }
                #             })
            session_id = data["session_id"]
            sender_id = data["sender_id"]

    # Language selection via query param (default "en-IN")

    chosen_language = ws.query_params.get("lang", "en-IN")
    print(f"[Server] Starting STT with language = {chosen_language}")
    chosen_lang_el = "hi" # TTS lang

    print("Lead ID sent to LLM: ", lead_id) # Debug 

    # Instantiate LLM and TTS clients for this session

    llm = LLMClient(lead_id=lead_id, session_id=session_id, sender_id=sender_id, on_plans=lambda plans: asyncio.create_task(safe_send_json(ws, {"type": "full_response", "data": plans}))) 
    # Use Azure or ElevenLabs based on configuration
    if USE_AZURE_TTS:
        print("[DEBUG] Creating Azure TTS client")
        tts_client = TTSClient(
            provider="azure",
            azure_key=AZURE_SPEECH_KEY,
            azure_region=AZURE_REGION
        )
    else:
        print("[DEBUG] Creating ElevenLabs TTS client")
        tts_client = TTSClient(
            provider="elevenlabs",
            api_key=ELEVEN_KEY,
            voice_id=VOICE_ID)

    # Conversation state variables

    last_interim: str = ""  # last interim transcript
    last_interim_ts: float = 0.0  # timestamp of last interim
    spec_launched: bool = False  # prevents duplicate LLM calls
    tts_interrupted: bool = False  # prevents agent speaking old message
    waiting_for_callback_ack: bool = False
    # process_timer = None  # ADD THIS LINE
    prev_sid: str = ""  # last TTS request ID handled
    last_user_end_ts: Optional[float] = None   # when user finished speaking
    last_user_activity_ts: float = time.perf_counter()
    last_agent_activity_ts: float = time.perf_counter()

    # Add these connection health variables HERE
    connection_error_count = 0
    last_successful_send = time.perf_counter()
    
    current_tts_sentence: Optional[str] = None  # sentence currently being spoken
    combined_tts_sentence: Optional[str] = None  # all sentences until interruption
    total_seconds_map: dict[str, int] = {}  # duration (sec) of each TTS sid at 44100 Hz
    tts_queue: asyncio.Queue[str] = asyncio.Queue()  # pending TTS sids
    pending_tts: dict[str, tuple] = {}  # sid -> (audio iterator, handshake ts)
    filler_sid: Optional[str] = None
    long_wait_sid: Optional[str] = None
    # TTS playback tracking for echo prevention
    tts_playing_until: float = 0.0  # timestamp when current TTS will finish
    recent_tts_content: list[str] = []  # track recent TTS sentences for content matching
    

    # TTS worker (Worker to consume TTS PCM chunks and forward via WebSocket)

    async def stop_tts_for_sid(sid: str):
        if sid in pending_tts:
            print(f"[🛑] Cancelling TTS sid={sid}")
            del pending_tts[sid]  # remove from queue if not yet picked up
            await safe_send_json({"type": "cancel_tts", "sid": sid})

    async def safe_send_json(ws: WebSocket, data: dict):
        nonlocal connection_error_count, last_successful_send
        if ws.client_state.name != "CONNECTED":
            print(f"[⛔] Cannot send JSON — WebSocket already closed: {data}")
            return False
        try:
            await ws.send_json(data)
            # IMPORTANT: Update success timestamp for ALL sends, not just full_response
            connection_error_count = 0
            last_successful_send = time.perf_counter()
            if data.get("type") == "full_response":
                print(f"[✅] Sent full_response to frontend: {data.get('data', {}).get('message', '')[:50]}...")
            return True
        except Exception as e:
            print(f"[❌] Failed to send JSON over WebSocket: {e}")
            connection_error_count += 1
            # Trigger LLM state recovery on WebSocket errors
            return False

    async def safe_send_bytes(ws: WebSocket, data: bytes):
        nonlocal connection_error_count, last_successful_send
        if ws.client_state.name != "CONNECTED":
            print(f"[⛔] Cannot send bytes — WebSocket already closed.")
            return False
        try:
            await ws.send_bytes(data)
            connection_error_count = 0  # Reset on success
            last_successful_send = time.perf_counter()
            return True
        except Exception as e:
            print(f"[❌] Failed to send bytes over WebSocket: {e}")
            connection_error_count += 1
            return False

    
    async def check_connection_health():
        nonlocal connection_error_count, last_successful_send
        
        while True:
            await asyncio.sleep(30)  # Check every 5 seconds
            
            # If too many errors or no successful sends for 30s
            if connection_error_count > 3  and (time.perf_counter() - last_successful_send > 15):
                print("[⚠️] Connection unhealthy, triggering recovery...")
                if llm:
                    await llm.restore_state_after_error()
                connection_error_count = 0
                last_successful_send = time.perf_counter()
    
    asyncio.create_task(check_connection_health())  
    
    async def tts_worker():
        nonlocal last_agent_activity_ts, tts_playing_until, recent_tts_content
        while True:
            sid = await tts_queue.get() # get next TTS request ID
            try:
                pair = pending_tts.pop(sid, None)
                if not pair:
                    continue
                
                audio_iter, handshake_ts = pair
                
                # Mark TTS as starting (estimate 5 seconds initially)
                tts_playing_until = time.perf_counter() + 5.0
               #print(f"[📢] TTS started, echo protection active")
                
                total_samples = 0  # counts PCM samples at 44.1 kHz
                enter_ts = time.perf_counter()
                
                first = True
                try:
                    async for pcm_chunk in audio_iter:
                        now = time.perf_counter()
                        if not pcm_chunk:
                            continue
                        total_samples += len(pcm_chunk) // 2
                        last_agent_activity_ts = now
                        if first:
                            print(f"[{now:.3f}] ← first TTS chunk sid={sid}")
                            if last_user_end_ts:
                                e2e = now - last_user_end_ts
                            #   print(f"[TTS WORKER] Agent Speech Time Taken for sid={sid}: {e2e:.3f}s")
                            first = False
                        result = await safe_send_bytes(ws, pcm_chunk)
                        if not result:
                            connection_error_count += 1
                            print(f"[❌] WebSocket TTS send failed: Connection error #{connection_error_count}")
                            if connection_error_count > 3:
                                # Too many failures, break out and let recovery happen
                                break
                        else:
                            connection_error_count = 0  # Reset on success
                            last_successful_send = time.perf_counter()
                except AssertionError as e:
                    print("[❌] Skipping PCM chunk due to WebSocket frame issue:", e)
                except Exception as e:
                    print("[❌] WebSocket TTS send failed:", e)
    
                # record duration of this TTS response in seconds
                total_seconds = total_samples / 44100.0
                total_seconds_map[sid] = total_seconds
                
                # Update with more accurate end time (audio still playing on frontend)
                tts_playing_until = time.perf_counter() + total_seconds + 0.5
               #print(f"[📢] TTS will finish playing at {tts_playing_until:.3f} ({total_seconds:.1f}s of audio)")
                
            finally:
                tts_queue.task_done()

    asyncio.create_task(tts_worker())

 
            
    async def user_nudge_monitor():
        nonlocal last_user_activity_ts, last_agent_activity_ts
        while True:
            await asyncio.sleep(5)
            print(f"[⏱️] Agent last spoke {time.perf_counter() - last_agent_activity_ts:.2f}s ago")
            if waiting_for_callback_ack:
                continue  # don't nudge during fallback mode
            if time.perf_counter() - last_agent_activity_ts > 60 and not waiting_for_callback_ack:
                print("[🔔] Nudging user after 30s silence...")
                nudge_text = "Are you still there? Just checking in."

                sid = str(uuid.uuid4())
                handshake_ts = time.perf_counter()
                gen = await tts_client.speak_sentence(nudge_text, ws, chosen_lang_el, handshake_ts)
                pending_tts[sid] = (gen, handshake_ts, 1)  # priority 1 so it doesn't preempt LLM
                await tts_queue.put((1, sid))
                if ws.client_state.name != "CONNECTED":
                    print("[⛔] Cannot send message — WebSocket already closed.")
                    return
                await safe_send_json(ws,{
                    "type": "tts_start",
                    "sid": sid,
                })

                # Avoid spamming nudges
                last_user_activity_ts = time.perf_counter()  # reset after nudge

    #asyncio.create_task(user_nudge_monitor())

    # Send text to LLM, stream tokens, split into sentences, and enqueue TTS
    async def launch_llm(text: str):
        nonlocal spec_launched, last_interim, current_tts_sentence, combined_tts_sentence, filler_sid, long_wait_sid, tts_interrupted
        t_llm_start = time.perf_counter()
        #print(f"[{t_llm_start:.3f}] → sending to LLM (text='{text[:30]} ...')")
        print(f"[LAUNCH_LLM] Called with text: {text}")
        
        final = ""
        response_buffer = "" # stores and concatenate the incoming tokens from LLM
        combined_tts_sentence = ""
        first_token = False
        last_token_ts = None
        
        response_started = asyncio.Event()

        async def filler_task():
            nonlocal filler_sid
            print(f"FILLER TASK STARTED")
            await asyncio.sleep(10)
            filler_text = "Please wait while I fetch the details."
         
            sid = str(uuid.uuid4())
            handshake_ts = time.perf_counter()
            gen = await tts_client.speak_sentence(filler_text, ws, chosen_lang_el, handshake_ts)
            pending_tts[sid] = (gen, handshake_ts)
            filler_sid = sid
            tts_queue.put_nowait(sid)
            await safe_send_json(ws,{
                "type": "tts_start",
                "sid": sid,
            })
        
        filler = asyncio.create_task(filler_task())
        
        # Fallback after 60 seconds
        long_wait_prompt_sent = False

        async def long_wait_fallback():
            nonlocal long_wait_sid
            print(f"LONG WAIT TASK STARTED")
            await asyncio.sleep(30)
            if not response_started.is_set():
                print("[⚠️] LLM still hasn’t responded — sending fallback message.")
                fallback_msg = "Sorry it’s taking time. Would you like to wait, or I can call you back on your registered number within the next 24 hours? What’s a good time for a call?"

                # TTS it
                sid = str(uuid.uuid4())
                handshake_ts = time.perf_counter()
                gen = await tts_client.speak_sentence(fallback_msg, ws, chosen_lang_el, handshake_ts)
                long_wait_sid = sid
                pending_tts[sid] = (gen, handshake_ts)
                tts_queue.put_nowait(sid)
                await safe_send_json(ws,{
                    "type": "tts_start",
                    "sid": sid,
                })
                # Step 3: Send to frontend as full agent response
                #await safe_send_json(ws,{
                #   "data": {
                #        "message": fallback_msg,
                #        "viewType": 1,
                #       "viewTypeName": "simpleIncomingText",
                #        "from": "agent",
                #        "showMessage": True,
                 #       "time": int(time.time() * 1000)
                  #  }
                #})

                # Enable callback acknowledgment mode
                nonlocal waiting_for_callback_ack
                waiting_for_callback_ack = True

                # Schedule force-cut in 30s if no user response
                async def force_disconnect():
                    print(f"[DEBUG] Force Disconnect called.")
                    await asyncio.sleep(30)
                    if waiting_for_callback_ack:
                        print(f"[⏱️] No user response after fallback. Ending call.")
                        try:
                             # TTS it
                            closing_msg = "No worries. We'll call you back shortly. Bye!"
                            sid = str(uuid.uuid4())
                            handshake_ts = time.perf_counter()
                            gen = await tts_client.speak_sentence(closing_msg, ws, chosen_lang_el, handshake_ts)
                            pending_tts[sid] = (gen, handshake_ts)
                            tts_queue.put_nowait(sid)
                            await safe_send_json(ws,{
                                "type": "tts_start",
                                "sid": sid,
                            })
                        
                           # await safe_send_json(ws,{
                           #     "type": "full_response",
                           #     "data": {
                           #        "message": closing_msg,
                           #         "viewType": 1,
                           #         "viewTypeName": "simpleIncomingText",
                           #         "from": "agent",
                           #         "showMessage": True,
                           #         "time": int(time.time() * 1000)
                           #     }
                           # })
                            async def wait_for_tts_completion(sid: str):
                                while sid in pending_tts:
                                    await asyncio.sleep(0.1)
                                print(f"[✅] TTS sid={sid} playback completed.")
                        
                            # ✅ Wait for all queued TTS to finish playing
                            await wait_for_tts_completion(sid)
            
                            # Clear the state so normal flow can resume
                            #waiting_for_callback_ack = False

                            try:
                                await asyncio.to_thread(stt.finish)  # only if finish is a sync function
                                print("[📴] STT closed after callback confirmation.")
                            except Exception as e:
                                print(f"[❌] Error closing STT: {e}")
                            try:
                                loop = asyncio.get_event_loop()
                                loop.create_task(ws.close())
                                #await ws.close()
                                print("[📴] WebSocket closed after callback confirmation.")
                            except Exception as e:
                                print(f"[❌] Error closing WebSocket: {e}")
                        except Exception as e:
                            print(f"[❌] Error during forced disconnect: {e}")
                asyncio.create_task(force_disconnect())

        fallback_task = asyncio.create_task(long_wait_fallback())

        try:
            async with asyncio.timeout(100):  # 30 second timeout
                token_count = 0
                async for token in llm.stream_response(text):
                   
                    now = time.perf_counter()
                    if token and not first_token:
                       # print(f"[{now:.3f}] ← first LLM token")
                        first_token = True
                        response_started.set()  # ✅ mark that response has begun
                        if not filler.done():
                            filler.cancel()
                               # ✅ Also stop any pending filler audio if it's mid-play
                        if filler_sid:
                            await stop_tts_for_sid(filler_sid)
                            filler_sid = None   
                        if long_wait_sid:
                            print("[🛑] Interrupting long-wait message with LLM response...")
                            await stop_tts_for_sid(long_wait_sid)
    
                            # speak interruption message
                            interrupt_sid = str(uuid.uuid4())
                            interrupt_text = "Just wait, I got the answer."
                            interrupt_handshake_ts = time.perf_counter()
                            gen = await tts_client.speak_sentence(interrupt_text, ws, chosen_lang_el, interrupt_handshake_ts)
                            pending_tts[interrupt_sid] = (gen, interrupt_handshake_ts, 0)
                            await tts_queue.put_nowait((interrupt_sid))
                            await safe_send_json(ws,{"type": "tts_start", "sid": interrupt_sid})
    
                            long_wait_sid = None
                    last_token_ts = now
    
                    try:
                        await safe_send_json(ws, {"type": "token", "token": token})
                         # ADD THIS DEBUG PRINT:
                        if token_count <= 5:  # Print first 5 tokens
                          # print(f"[TOKEN {token_count}] Sent to frontend: '{token}'")
                          token_count += 1
                    except Exception as e:
                        #print(f"[❌] LLM token send failed: {e}")
                        return  # optionally break out of streaming
    
                    response_buffer += token
                    while True:
                        # flush out full sentences via regex
                        m = re.search(r"([\.\!\?|])(\s|$)", response_buffer)
                        if not m:
                            break
                        end = m.end()
                        sent = response_buffer[:end].strip()
                        print(f"[{time.perf_counter():.3f}] [Buffer] flush: '{sent}'")
                        last_user_end_ts = time.perf_counter()
    
                        # generate TTS for this sentence
                        sid = str(uuid.uuid4())
                        handshake_ts = time.perf_counter()
                        gen = await tts_client.speak_sentence(sent, ws, chosen_lang_el, last_user_end_ts)
                        print(f"[gen]: {gen}")
                        current_tts_sentence = sent
                        recent_tts_content.append(sent.lower())
                        if len(recent_tts_content) > 10:  # Keep last 10 sentences
                            recent_tts_content.pop(0)
                        if combined_tts_sentence == "":
                            combined_tts_sentence = current_tts_sentence
                        else:
                            combined_tts_sentence += " " + current_tts_sentence
    
                        pending_tts[sid] = (gen, handshake_ts)
                        tts_queue.put_nowait(sid)
                        await safe_send_json(ws,{
                            "type": "tts_start",
                            "sid": sid,
                        })
                        response_buffer = response_buffer[end:] #Updating response buffer to remove already spoken or (sent to TTS) text
                
                if last_token_ts:
                    print(f"[{last_token_ts:.3f}] ← last LLM token")
                
                # any leftover text after streaming ends
                if response_buffer.strip():
                    final = response_buffer.strip()
                    print(f"[{time.perf_counter():.3f}] [Buffer] final flush: '{final}'")
                    last_user_end_ts = time.perf_counter()
                    sid = str(uuid.uuid4())
                    handshake_ts = time.perf_counter()
                    gen = await tts_client.speak_sentence(final, ws, chosen_lang_el, last_user_end_ts)
                    print(f"[gen]: {gen}")
                    current_tts_sentence = final
                    if combined_tts_sentence == "":
                        combined_tts_sentence = current_tts_sentence
                    else:
                        combined_tts_sentence += " " + current_tts_sentence
                    pending_tts[sid] = (gen, handshake_ts)
                    tts_queue.put_nowait(sid)
                    await safe_send_json(ws,{
                        "type": "tts_start",
                        "sid": sid,
                    })
                else:
                    print("[⚠️] No final message from LLM")

                # ADD THIS NEW CODE HERE:
                # Send the complete message for chat display
                if combined_tts_sentence and combined_tts_sentence.strip():
                   # await safe_send_json(ws, {
                    #   "data": {
                     #       "message": combined_tts_sentence.strip(),
                     #       "text": combined_tts_sentence.strip(),
                     #       "viewType": 1,
                     #       "viewTypeName": "simpleIncomingText",
                     #       "from": "agent",
                     #       "showMessage": True,
                     #       "time": int(time.time() * 1000)
                      #  }
                    #})
                    print(f"[CHAT DISPLAY] Sent complete message: {combined_tts_sentence[:50]}...")
                    
                await safe_send_json(ws,{"type": "response_end"})

        except asyncio.TimeoutError:
            print("[❌] LLM response timeout after 30 seconds")
            await safe_send_json(ws, {"type": "error", "message": "Response timeout"})
            return
        finally:
            t_llm_end = time.perf_counter()
            print(f"[TIMING] Speech took {t_llm_end - t_llm_start:.3f} sec")
            spec_launched = False
            last_interim = ""
            tts_interrupted = False  # ADD THIS LINE
            if not filler.done():
                filler.cancel()
            if filler_sid:
                await stop_tts_for_sid(filler_sid)
                filler_sid = None   
            if not fallback_task.done():
                fallback_task.cancel()  

    # Send text to LLM, stream tokens, split into sentences, and enqueue TTS
    async def launch_llm_text(text: str):
        t_llm_start = time.perf_counter()
        print(f"[LAUNCH_LLM_TEXT] Called with text: {text}")
        final = ""
        response_buffer = "" # stores and concatenate the incoming tokens from LLM
        
        try:
            # Store user message in transcript
            now = datetime.now(timezone.utc).isoformat()
            user_transcript = {"role": "user", "message": text, "timestamp": now}
            if not hasattr(llm, 'transcript'):
                llm.transcript = []
            llm.transcript.append(user_transcript)
            
            token_count = 0
            async for token in llm.stream_response(text):
                token_count += 1
                now = time.perf_counter()
                try:
                    # Send each token to frontend
                    await safe_send_json(ws, {"type": "token", "token": token})
                    response_buffer += token
                    
                    # Debug print to verify tokens are being sent
                    if token_count <= 5:  # Print first 5 tokens for debugging
                        print(f"[TOKEN {token_count}] Sent: '{token}'")
                        
                except Exception as e:
                    print(f"[❌] LLM token send failed: {e}")
                    break  # Continue accumulating even if send fails

            # After all tokens, send the complete message as full_response
            if response_buffer.strip():
                final = response_buffer.strip()
                print(f"[{time.perf_counter():.3f}] [Final message] '{final}'")

                # Uncomment the full_response block:
               # await safe_send_json(ws, {
               #      "type": "full_response", 
               #       "data": {
               #           "message": final,
               #           "text": final,
               #          "viewType": 1,
               #           "viewTypeName": "simpleIncomingText",
               #           "from": "agent",
               #           "showMessage": True,
               #          "time": int(time.time() * 1000)
               #     }
               #   })
                
                # Store agent response in transcript
                agent_transcript = {"role": "agent", "message": final, "timestamp": datetime.now(timezone.utc).isoformat()}
                llm.transcript.append(agent_transcript)
                
            else:
                print("[⚠️] No final message from LLM")
                
            await safe_send_json(ws, {"type": "response_end"})
            
        except Exception as e:
            print(f"[❌] Error in launch_llm_text: {e}")
            import traceback
            traceback.print_exc()
        finally:
            t_llm_end = time.perf_counter()
            print(f"[TIMING] Text response took {t_llm_end - t_llm_start:.3f} sec")
            print(f"[STATS] Total tokens sent: {token_count}")

    # Insert immediate transcript note into Supabase when interrupted mid-sentence

    async def insert_mid_call_note_immediate(text: str):
        now = datetime.now(timezone.utc).isoformat()
        message = text
        transcript = [{"role":"agent","message":message,"timestamp":now}]
        content = "Agent Call Message Interrupt"
        print(f"Transcript Immediate JSON:\n{message}")
        note = { 
            "id": str(uuid.uuid4()),
            "lead_id": lead_id,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "type": "web_call",
            "call_sid": None,
            "transcript_summary": None,
            "transcript": transcript
        }
        res = supabase_client.from_("lead_notes").insert(note).execute()
        print(f"[SUPABASE] Immediate Row inserted in lead_notes with id={note['id']}")

    # Callback invoked for each STT transcript event

    async def on_transcript(text: str, is_final: bool):
        print(f"[TRANSCRIPT] received: {text} | final: {is_final} | conn: {connection_id}")
        # CRITICAL: Validate this is still the active connection
        if ws.client_state.name != "CONNECTED":
            print(f"[IGNORE] Transcript for closed connection {connection_id}: '{text}'")
            return
            
        # Also check if WebSocket is in lead_clients
        if lead_id not in lead_clients or ws not in lead_clients[lead_id]:
            print(f"[IGNORE] Transcript for removed connection {connection_id}: '{text}'")
            return
    
        nonlocal last_interim, last_interim_ts, spec_launched, last_user_end_ts, current_tts_sentence, waiting_for_callback_ack, last_user_activity_ts

        #if agent_speaking:
        #    print(f"[🔇] Ignoring transcript during agent speech: '{text}'")
        #    return
        
        last_user_activity_ts = time.perf_counter()
        # await safe_send_json(ws,{"type": "transcript", "text": text, "final": is_final})  # Send to UI
        now = time.perf_counter()
        #if playbackEndTime > now:
        #   print(f"[🔇] Ignoring transcript during playback: '{text}'")
        #   return
        
        # For interim transcripts, send immediately (they don't trigger actions anyway)
        if not is_final:
             await safe_send_json(ws,{"type": "transcript", "text": text, "final": is_final})
             # If user starts speaking, stop any current TTS
             if len(text.split()) >= 1 and not tts_interrupted:
                await safe_send_json(ws,{"type": "cancel_tts"})
                tts_interrupted = True
                print("[🔇] TTS interrupted due to user speech")

             # Store interim for comparison (optional, but harmless to keep)
             last_interim = text
             last_interim_ts = now
             return
        
        if waiting_for_callback_ack and is_final:
            normalized = text.lower().strip()
            print(f"INSIDE GOT CUSTOMER RESPONSE FOR CALLBACK - {text}")
            if any(kw in normalized for kw in ["yes", "ok", "sure", "callback", "call me", "please do"]):
                print("[📞] User accepted callback suggestion.")
                ack_msg = "Got it! We'll call you back soon. Thank you."
                sid = str(uuid.uuid4())
                handshake_ts = time.perf_counter()
                gen = await tts_client.speak_sentence(ack_msg, ws, chosen_lang_el, handshake_ts)
                pending_tts[sid] = (gen, handshake_ts)
                tts_queue.put_nowait(sid)
                await safe_send_json(ws,{
                    "type": "tts_start",
                    "sid": sid,
                })

               # await safe_send_json(ws,{
               #     "type": "full_response",
               #    "data": {
               #         "message": ack_msg,
               #         "viewType": 1,
               #         "viewTypeName": "simpleIncomingText",
               #         "from": "agent",
               #         "showMessage": True,
               #         "time": int(time.time() * 1000)
               #     }
               # })

                async def wait_for_tts_completion(sid: str):
                    while sid in pending_tts:
                        await asyncio.sleep(0.1)
                    print(f"[✅] TTS sid={sid} playback completed.")
                
                # ✅ Wait for all queued TTS to finish playing
                await wait_for_tts_completion(sid)
 
                # Optional: log this to Supabase notes
                await insert_mid_call_note_immediate(f"User requested callback: '{text}'")

                # Clear the state so normal flow can resume
                
                waiting_for_callback_ack = False

                try:
                    await asyncio.to_thread(stt.finish)  # only if finish is a sync function
                    print("[📴] STT closed after callback confirmation.")
                except Exception as e:
                    print(f"[❌] Error closing STT: {e}")
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(ws.close())
                    #await ws.close()
                    print("[📴] WebSocket closed after callback confirmation.")
                except Exception as e:
                    print(f"[❌] Error closing WebSocket: {e}")

                return
                        
        # SMARTER ECHO FILTERING FOR FINAL TRANSCRIPTS  
        if is_final and now < tts_playing_until:
            text_lower = text.lower().strip()
            echo_detected = False
            
            # Simply check if user's words match any recent TTS content
            for tts_phrase in recent_tts_content:
                tts_lower = tts_phrase.lower()
                
                # Direct substring check - if what user said is part of what bot is saying
                if text_lower in tts_lower:
                    echo_detected = True
                    print(f"[🔇 ECHO] Matches TTS content: '{text}'")
                    break
                    
                # For partial matches, check word-level overlap
                user_words = set(text_lower.split())
                tts_words = set(tts_lower.split())
                
                # Calculate overlap ratio (ignoring very short words)
                meaningful_user = {w for w in user_words if len(w) > 2}
                meaningful_tts = {w for w in tts_words if len(w) > 2}
                
                if meaningful_user:
                    overlap = len(meaningful_user & meaningful_tts) 
                    overlap_ratio = overlap / len(meaningful_user)
                    
                    if overlap_ratio >= 0.8:  # 80% of user's words appear in TTS
                        echo_detected = True
                        print(f"[🔇 ECHO] High overlap ({overlap_ratio:.0%}) with TTS: '{text}'")
                        break
            
            if echo_detected:
                return  # Don't send to UI or process
            else:
                print(f"[✅] User input during TTS allowed: '{text}'") 
        # Only send to UI if it passed echo filtering
        await safe_send_json(ws,{"type": "transcript", "text": text, "final": is_final})
        # Final transcript received
        last_user_end_ts = now
        if not spec_launched:
            if not text.strip():
                print("[⚠️] Ignoring empty final transcript")
                return
            spec_launched = True
            print(f"[{now:.3f}] [STT] final transcript kickoff: '{text}'")
            await safe_send_json(ws,{"type": "stop_speech"})
            asyncio.create_task(launch_llm(text))
        else:
            print("[🚫] Skipping LLM call: spec_launched already True")


    # Starting appropriate STT in a background thread
    chosen_stt = ws.query_params.get("stt", "google").lower()
    print(f"CHOSEN_STT: {chosen_stt}")
    lang = "English" if chosen_language == "en-IN" else "Hindi"

    # Initialize the appropriate STT transcriber
    if chosen_stt == "azure":
        #print(f"[DEBUG] Initializing Azure STT...")
        from azure_stt import AzureTranscriber
        #print(f"[DEBUG] Azure credentials present: {bool(AZURE_SPEECH_KEY and AZURE_REGION)}")
        stt = AzureTranscriber(AZURE_SPEECH_KEY, AZURE_REGION, language="en-IN")
        #print(f"[DEBUG] Azure STT created, starting thread...")
        threading.Thread(target=lambda: asyncio.run(stt.start(on_transcript)), daemon=True).start() 
        #print(f"[DEBUG] Azure STT thread started")
    elif chosen_stt == "assembly" or chosen_stt == "assemblyai":
        from assemblyai_stt import AssemblyAITranscriber
        stt = AssemblyAITranscriber(ASSEMBLY_KEY, language=chosen_language)
        threading.Thread(target=lambda: asyncio.run(stt.start(on_transcript)), daemon=True).start()
            
    elif chosen_stt == "google":
        from google_stt import GoogleTranscriber
        stt = GoogleTranscriber("/Users/prashan.agarwal/Downloads/media-cdn-437609-ed2502095156.json", language="en-IN")
        threading.Thread(target=lambda: asyncio.run(stt.start(on_transcript)), daemon=True).start()
        
    else:
        from deepgram_stt import DeepgramTranscriber
        stt = DeepgramTranscriber(DG_KEY, language=chosen_language)
        # Sending initial greeting to LLM ("greet in English/Hindi")
        
        #asyncio.create_task(on_transcript(f"Can you help with health insurance", True))
        threading.Thread(target=lambda: asyncio.run(stt.start(on_transcript)), daemon=True).start()
    
    # Step 1: Create a synthetic agent message
    #initial_message = "Hello - I am Nina calling from Raksha Insurance. How can I help you?"
    initial_message = "Hello Good morning - Main Minraj bol raha hoon RakshaInsurance.com se...Aaj kaise hain aap?"

    # Send initial greeting through LLM's on_plans (MESSAGE PIPE)
    if hasattr(llm, '_on_plans') and llm._on_plans:
        await llm._on_plans({
            "message": initial_message,
            "viewType": 1,
            "viewTypeName": "simpleIncomingText",
            "from": "agent",
            "showMessage": True,
            "insuranceSummary": None,  # No summary at start
            "time": int(time.time() * 1000)
        })

    # Step 4: Trigger TTS the same way your normal buffer flush does
    sid = str(uuid.uuid4())
    handshake_ts = time.perf_counter()
    gen = await tts_client.speak_sentence(initial_message, ws, chosen_lang_el, handshake_ts)
    pending_tts[sid] = (gen, handshake_ts)
    tts_queue.put_nowait(sid)
    await safe_send_json(ws,{
        "type": "tts_start",
        "sid": sid,
    })
    # ADD THESE TWO LINES HERE:
    #await asyncio.sleep(1.0)
    #print("[⏳] Audio systems initializing...")
    try:
        # main receive loop: handles audio bytes and control messages
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is not None:
                # print("Audio bytes received:", len(msg["bytes"]))
                stt.feed_audio(msg["bytes"]) # sends mic PCM to Deepgram received from client

            if msg.get("text"):
                raw = msg["text"]
                print("🧐 TEXT payload:", raw)
                data = json.loads(msg["text"])
                print("🧐 TEXT payload JSON :", data)
                if data.get("type") == "cancel_tts":
                    print("🧐 raw cancel_tts payload:", data)
                    sid = data["sid"]
                    played = data["playedSeconds"]
                    playback_rate = data["playbackRate"]
                    total = total_seconds_map.get(sid)
                    print("------", prev_sid)
                    print("------", sid, played, total)
                    if sid and prev_sid != sid:
                        if total and playback_rate:
                            # scale server duration to client rate
                            adjusted_total = total * (playback_rate/44100)
                            fraction = min(1.0, played / adjusted_total)
                        else:
                         fraction = 1.0
                        print(fraction)
                        if fraction < 1:
                            # character‑level substring:
                            txt = current_tts_sentence or ""
                            print(f"[Combined TTS Sentence] {combined_tts_sentence}")
                            parts = re.split(r'(?<=[.!?|])\s', combined_tts_sentence)
                            last_sentence = parts.pop()
                            char_cutoff = max(1, int(len(txt) * fraction))
                            partial = txt[:char_cutoff].rstrip()
                            if fraction < 1:
                                partial = partial + "..."
                            prev_sid = sid
                            print(f"[INTERRUPT] partial text (before): {partial}")
                            last_sentence = partial
                            parts.append(last_sentence)
                            partial = " ".join(parts)
                            print(f"[INTERRUPT] partial text (after): {partial}")
                            #await insert_mid_call_note_immediate(partial) # adds a row with spoken text only with interruption in supabase lead_notes table
                elif data.get("type") == "user_message":
                    text = data.get("message")
                    print(f"Calling the Launch LLM Text")
                    #if (mute ) call llm text else call llm audio...
                    asyncio.create_task(launch_llm(text))
            elif msg.get("type") == "websocket.disconnect":
                break
            #await on_transcript("I want to buy insurance", True)
    except WebSocketDisconnect:
        print("[❌] WebSocket client disconnected.")
        return
    finally:
        # 4. Enhance the finally block with better cleanup
        print(f"[WS CLEANUP] Cleaning up connection {connection_id} for lead {lead_id}")
        # ADD THIS AFTER THE ABOVE PRINT:
        if connection_id in active_connections:
            del active_connections[connection_id]
            
        # Remove this WebSocket from lead_clients
        if lead_id in lead_clients:
            lead_clients[lead_id].discard(ws)
            if not lead_clients[lead_id]:  # If no more connections for this lead
                del lead_clients[lead_id]
                print(f"[WS CLEANUP] Removed lead {lead_id} from lead_clients")
        
        # Clean up STT
        if stt:
            try:
                # Clean up STT with more aggressive shutdown
                print(f"[WS CLEANUP] Stopping STT for connection {connection_id}")
                stt.finish()
                # For Azure STT, give it a moment to clean up
                if chosen_stt == "azure":
                    await asyncio.sleep(0.5)
            except Exception as e:
                print(f"[WS CLEANUP] Error finishing STT: {e}")
        
        # Reset LLM
        if llm:
            try:
                # Save conversation state
                await llm._save_conversation_state_async()
                llm.reset()
            except Exception as e:
                print(f"[WS CLEANUP] Error resetting LLM: {e}")
        
        # Close WebSocket
        try:
            await ws.close()
        except RuntimeError:
            pass
        
        print(f"[WS CLEANUP] Cleanup complete for lead {lead_id}")
        print(f"[WS CLEANUP] Cleanup complete for connection {connection_id}")
