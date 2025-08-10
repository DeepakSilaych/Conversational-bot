import asyncio
import os
import uuid
import threading
import re
import json
import time
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator, Dict, List
from supabase import create_client, ClientOptions
from shared import db_executor
from supabase import create_client, ClientOptions
from dotenv import load_dotenv
from httpx import Client, Timeout
import google.generativeai as genai
from get_recommendation import handle_recommendations
from get_recommendation import format_response, COMPANY_SHORT_NAMES
from helpers import check_recommendation_cache

load_dotenv()
url, key, GEMINI_KEY = os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"], os.environ["GEMINI_API_KEY"]




def extract_json_block(text: str) -> dict:
    # Try to extract code block ```json ... ```
    match = re.search(r'\(JSON\):\s*({.*?})\s*(end_call|$)', text, re.DOTALL)
    if match:
        print(f"If match {match}")
        json_str = match.group(1)
    else:
        # fallback: try to find first {...} block
        match = re.search(r"(\{.*?\})", text, re.DOTALL)
        print(f"Else match {match}")
        if match:
            json_str = match.group(1)
        else:
            print("[❌] No JSON object found.")
            return {}
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[❌] Failed to parse JSON: {e}")
        return {}

def get_all_info_collected_flag(text: str) -> bool:
    """
    Extracts JSON block from the given text and checks if all required fields are present and valid.
    Returns True if all expected info is collected, else False.
    """
    try:
        # Extract JSON block
        match = re.search(r'\(JSON\):\s*({.*?})\s*(end_call|$)', text, re.DOTALL)
        if not match:
            return False
        
        data = json.loads(match.group(1))

        # Required fields and validations
        required_keys = ["who_to_insure", "ages", "sum_assured", "budget", "city"]
        for key in required_keys:
            if key not in data or data[key] in [None, "", []]:
                return False

        if not isinstance(data["ages"], dict) or not all(isinstance(v, int) for v in data["ages"].values()):
            return False

        if not isinstance(data["sum_assured"], int) or not isinstance(data["budget"], int):
            return False

        if not isinstance(data["city"], str) or not data["city"].strip():
            return False

        return True

    except Exception:
        return False
    
class LLMClient:
    def __init__(self, 
                lead_id: str,
                session_id: str,
                sender_id: str,
                conversation_id: str = str(uuid.uuid4()),
                assistant_name: str = "",
                customer_name: str = "", 
                customer_email: str = "",
                customer_phone: str = "", 
                on_plans: Optional[AsyncGenerator] = None):
        # Conversation
        self.session_id = session_id
        self.sender_id = sender_id
        self.conversation_id = conversation_id
        self._on_plans = on_plans
        self.is_all_info_collected = False
        self.postRecoPhase = False
        # Add these for tracking cart/compare state
        self.cart_items = []        # Current cart items
        self.compare_items = []     # Current comparison items

        # Caller & Assistant
        self.assistant_name, self.customer_name, self.customer_email, self.customer_phone = assistant_name, customer_name, customer_email, customer_phone
        # Supabase
        timeout = Timeout(15.0, connect=15.0)
        httpx_client = Client(timeout=timeout)
        # Define user_details with initial values (before any database operations)
        user_details = f"Name: {self.customer_name}, Phone: {self.customer_phone}, Email: {self.customer_email}"

        # Gemini init
        #GEMINI_KEY = os.environ("GEMINI_API_KEY")
        genai.configure(api_key=GEMINI_KEY)

        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config={
                "temperature": 0.2,  # Lower for better instruction following
                "top_p": 0.8,       # More focused
                "top_k": 20,        # Stricter vocabulary
                "max_output_tokens": 1024,
                "candidate_count": 1,
                "stop_sequences": ["Translation:", "This means", "In English", "In Hindi"]
            },
            # Add system instruction if your Gemini version supports it
            system_instruction="""You are an insurance sales agent who MUST follow these rules:
            1. DEFAULT to code-mixed Hindi-English (Hinglish) for all responses
            2. ONLY switch to pure English if customer explicitly asks "Please speak in English" or similar
            3. Natural code-mixing is expected - mix Hindi and English fluidly जैसे normal conversation में होता है
            4. Use English for technical terms (premium, coverage) with Hindi explanations
            5. NEVER provide translations in parentheses - just speak naturally in Hinglish
            Breaking these rules will cause system failure."""
        )
        # 5. LOAD THE PROMPT FILES HERE
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            self.base_prompt = f.read().strip()
        
        with open("minraj_behaviors.txt", "r", encoding="utf-8") as f:
            self.persona = f.read().strip()

        # Set minimal system prompt (no user data yet)
        self.system_prompt = self.base_prompt.replace("{{persona_behaviors}}", self.persona).replace("{{user_details}}", "Name: Unknown, Phone: Unknown, Email: Unknown").replace("{{collected_info}}", "{}")
        # Leave {{user_details}} and {{collected_info}} as placeholders for now
                    
        #Use httpx_client instead of http_client
        with open("system_prompt_advanced.txt", "r", encoding="utf-8") as f:
            raw_prompt_advanced = f.read().strip()

       # Use httpx_client instead of http_client
        options = ClientOptions(httpx_client=httpx_client)

        self.supabase = create_client(url, key, options=options)
        # Lead
        self.lead_id = lead_id
        self.is_loaded = False
        self.conversation_phase = "info_collection"  # Default phase
        
        # Searching in the Leads table in the supabase if there exists any row with the initialized lead_id, if exists, it extracts customer info else create a new row with default customer info.
        # res = self.supabase.from_("leads").select("id, name, email, phone_number,recommended_plans, backup_plans, created_date").eq("id", self.lead_id).limit(1).execute()
        # notes_res = self.supabase.from_("lead_notes").select("transcript_summary, transcript").eq("lead_id", self.lead_id).order("created_at", desc=False).limit(20).execute()
        # conversation_state = self.supabase.from_("conversation_states").select("*").eq("lead_id", self.lead_id).order("created_at", desc=False).limit(1).execute()

        # State
        self.history = []
        self.transcript = []
        self.collected_info = {}
        self.recommended_plans = {}

        #if res.data:
         #   existing = res.data[0]
         #  self.lead_id = existing["id"]  # Overwrite if phone matched but ID was new
         #   print(f"Existing Row: {existing}")
         #   self.customer_name  = existing.get("name",  self.customer_name)
         #   self.customer_email = existing.get("email", self.customer_email)
        3#   self.customer_phone = existing.get("phone_number", self.customer_phone)
            # if existing.get("recommended_plans") is not None:
            #     self.recommended_plans = existing["recommended_plans"]
         #   print(f"[SUPABASE] Found existing lead {self.lead_id}, created at {existing.get('created_date')}")
             # Construct user details string
         #   user_details = f"Name: {self.customer_name}, Phone: {self.customer_phone}, Email: {self.customer_email}"

            # Replace placeholder
            #self.system_prompt = raw_prompt.replace("{{user_details}}", user_details)
            #print(f"[System Prompt] : {self.system_prompt}")
        #else:
         #   print(f"Lead does not exist")
            #self._create_lead_record()
    
        

     #   if notes_res.data:
     #       for note in notes_res.data[::-1]:
     #          try:
     #               transcript = note.get("transcript")
     #              #print(f"[TRANSCRIPT] {transcript}")
     #               if isinstance(transcript, str):
     #                  turns = json.loads(transcript)
     #               else:
     #                  turns = transcript  # already parsed
                    
     #               for turn in turns:
     #                  role = "user" if turn["role"] == "user" else "assistant"
     #                 content = turn["message"]
     #                   self.history.append({"role": role, "content": content})
     #          except Exception as e:
     #               print(f"[❌] Error parsing transcript in note: {e}")
        #print(f"[LEAD NOTES] Data : {self.history}")

        
        # This is the correct block that should remain (current lines 201-207)
      #  if conversation_state.data and conversation_state.data[0]["collected_info"] is not None:
      #     self.collected_info = conversation_state.data[0]["collected_info"]
      #     self.is_all_info_collected = conversation_state.data[0]["is_all_info_collected"]
      #    print(f"Self Collected info from Supabase : {self.collected_info} : {self.is_all_info_collected}")
      #   self.system_prompt = raw_prompt.replace("{{user_details}}", user_details).replace("{{collected_info}}", json.dumps(self.collected_info)).replace("{{persona_behaviors}}", persona_behaviors)
      #  else:
      #    self.system_prompt = raw_prompt.replace("{{user_details}}", user_details).replace("{{persona_behaviors}}", persona_behaviors)

      #  print(f"[NORMAL PROMPT] {len(self.system_prompt)} - {self.system_prompt[:100]}")
            
        self.reset()
    
    async def _lazy_load_lead_data(self):
        """Load lead data from database only when needed"""
        if self.is_loaded:
            return
            
        print(f"[LAZY LOAD] Loading data for lead {self.lead_id}")
        
        try:
            # Move the 3 DB queries here
            res = self.supabase.from_("leads").select("id, name, email, phone_number,recommended_plans, backup_plans, created_date").eq("id", self.lead_id).limit(1).execute()
            notes_res = self.supabase.from_("lead_notes").select("transcript_summary, transcript").eq("lead_id", self.lead_id).order("created_at", desc=False).limit(20).execute()
            conversation_state = self.supabase.from_("conversation_states").select("*").eq("lead_id", self.lead_id).order("created_at", desc=False).limit(1).execute()
            
            # Process leads data
            if res.data:
                existing = res.data[0]
                self.lead_id = existing["id"]
                print(f"Existing Row: {existing}")
                self.customer_name = existing.get("name", self.customer_name)
                self.customer_email = existing.get("email", self.customer_email)
                self.customer_phone = existing.get("phone_number", self.customer_phone)
                print(f"[SUPABASE] Found existing lead {self.lead_id}, created at {existing.get('created_date')}")
                
                # Construct user details string
                user_details = f"Name: {self.customer_name}, Phone: {self.customer_phone}, Email: {self.customer_email}"
            else:
                print(f"Lead does not exist")
                user_details = f"Name: {self.customer_name}, Phone: {self.customer_phone}, Email: {self.customer_email}"
            
            # We'll handle notes and conversation state in next step
           
            # Process notes (conversation history)
            if notes_res.data:
                for note in notes_res.data[::-1]:
                    try:
                        transcript = note.get("transcript")
                        if isinstance(transcript, str):
                            turns = json.loads(transcript)
                        else:
                            turns = transcript
                        
                        for turn in turns:
                            role = "user" if turn["role"] == "user" else "assistant"
                            content = turn["message"]
                            self.history.append({"role": role, "content": content})
                    except Exception as e:
                        print(f"[❌] Error parsing transcript in note: {e}")
            
            # Process conversation state
            if conversation_state.data and conversation_state.data[0]["collected_info"] is not None:
                self.collected_info = conversation_state.data[0]["collected_info"]
                self.is_all_info_collected = conversation_state.data[0].get("is_all_info_collected", False)
                print(f"Self Collected info from Supabase : {self.collected_info} : {self.is_all_info_collected}")
                # Update last interaction timestamp
                asyncio.create_task(self._save_conversation_state_async())
                # Restore conversation phase from database
                if conversation_state.data[0].get("current_phase"):
                    self.conversation_phase = conversation_state.data[0]["current_phase"]
                    print(f"[RESTORE] Loaded conversation phase from DB: {self.conversation_phase}")
                # ===== END ADDITION =====
            
            # Update system prompt with loaded data
            self.system_prompt = self.base_prompt.replace("{{user_details}}", user_details).replace("{{collected_info}}", json.dumps(self.collected_info)).replace("{{persona_behaviors}}", self.persona)
            print(f"[SYSTEM PROMPT] Length: {len(self.system_prompt)}")
            print(f"[SYSTEM PROMPT] First 100 chars: {self.system_prompt[:100]}...")

            # Determine conversation phase based on loaded data
            if res.data and res.data[0].get("recommended_plans"):
                # Only set to post_reco if not already loaded from DB
                if self.conversation_phase != "post_reco":
                    self.conversation_phase = "post_reco"
                    print(f"[PHASE] Lead has recommendations - setting phase to post_reco")
                # Save phase transition
                asyncio.create_task(self._save_conversation_state_async())
            elif self.is_all_info_collected:
                self.conversation_phase = "ready_for_reco"
                print(f"[PHASE] All info collected - setting phase to ready_for_reco")
            else:
                self.conversation_phase = "info_collection"
                print(f"[PHASE] Still collecting info")
        
        except Exception as e:
            print(f"[ERROR] Failed to load lead data: {e}")
            # Continue with defaults if load fails
        
        self.is_loaded = True


    def _fast_parse_response(self, full_message: str) -> dict:
        """Fast single-pass parsing of LLM response"""
        result = {
            'message': '',
            'json': {},
            'flags': {
                'end_call': False,
                'is_all_info_collected': False,
                'handover_to_human': False
            }
        }
        
        # DEBUG: Log raw input
        print(f"[PARSE DEBUG] ===== PARSING START =====")
        print(f"[PARSE DEBUG] Raw message length: {len(full_message)}")
        print(f"[PARSE DEBUG] Raw message preview:\n{full_message[:300]}...")
        
        # Handle different response formats
        if "ExtractedInfo" in full_message:
            print(f"[PARSE DEBUG] Found 'ExtractedInfo' in message")
            
            # Split by ExtractedInfo to separate message from JSON
            parts = full_message.split("ExtractedInfo", 1)
            print(f"[PARSE DEBUG] Split into {len(parts)} parts")
            
            # First part contains the message (possibly duplicated)
            message_part = parts[0].strip()
            print(f"[PARSE DEBUG] Message part: {message_part[:100]}...")
            
            # Check if message is duplicated with "Message:" prefix
            if "Message:" in message_part:
                # Special case: Check if text before "Message:" is same as after
                parts_by_message = message_part.split("Message:", 1)
                if len(parts_by_message) == 2:
                    text_before = parts_by_message[0].strip()
                    text_after = parts_by_message[1].strip()
                    
                    # Remove any trailing flags from text_after
                    lines_after = text_after.split('\n')
                    clean_lines_after = []
                    for line in lines_after:
                        if (line.strip().startswith("end_call=") or 
                            line.strip().startswith("is_all_info_collected=") or
                            line.strip().startswith("handover_to_human=")):
                            break
                        clean_lines_after.append(line)
                    
                    text_after_clean = '\n'.join(clean_lines_after).strip()
                    
                    # If they're the same, just use the text after "Message:"
                    if text_before == text_after_clean or text_before in text_after_clean:
                        result['message'] = text_after_clean
                        print(f"[PARSE DEBUG] Detected duplicate, using text after 'Message:'")
                    else:
                        # Otherwise, just take what's after "Message:"
                        result['message'] = text_after_clean
            else:
                # No "Message:" prefix, use the whole first part
                result['message'] = message_part
                # But remove any trailing flags
                lines = result['message'].split('\n')
                clean_lines = []
                for line in lines:
                    line_stripped = line.strip()
                    if (line_stripped.startswith("end_call=") or 
                        line_stripped.startswith("is_all_info_collected=") or
                        line_stripped.startswith("handover_to_human=")):
                        break
                    clean_lines.append(line)
                result['message'] = '\n'.join(clean_lines).strip()
            
            # IMPROVED JSON EXTRACTION
            if len(parts) > 1:
                json_section = parts[1]
                print(f"[PARSE DEBUG] JSON section length: {len(json_section)}")
                print(f"[PARSE DEBUG] JSON section preview: {json_section[:100]}...")
                
                # Try multiple patterns for JSON extraction
                # Pattern 1: Look for (JSON): with optional whitespace
                if "(JSON):" in json_section or " (JSON):" in json_section:
                    print(f"[PARSE DEBUG] Found '(JSON):' marker")
                    try:
                        # Extract JSON using the helper function
                        extracted = extract_json_block(full_message)
                        if extracted:
                            result['json'] = extracted
                            print(f"[PARSE DEBUG] Successfully extracted JSON: {result['json']}")
                        else:
                            print(f"[PARSE DEBUG] extract_json_block returned empty dict")
                    except Exception as e:
                        print(f"[PARSE DEBUG ERROR] JSON extraction failed: {e}")
                        result['json'] = {}
                else:
                    # Pattern 2: Try to find raw JSON object without (JSON): marker
                    print(f"[PARSE DEBUG] No '(JSON):' marker found, trying raw JSON extraction")
                    # Look for JSON object pattern
                    json_match = re.search(r'\{[^}]*\}', json_section, re.DOTALL)
                    if json_match:
                        try:
                            json_str = json_match.group(0)
                            result['json'] = json.loads(json_str)
                            print(f"[PARSE DEBUG] Extracted raw JSON: {result['json']}")
                        except json.JSONDecodeError as e:
                            print(f"[PARSE DEBUG ERROR] Raw JSON decode failed: {e}")
                            result['json'] = {}
                    else:
                        print(f"[PARSE DEBUG] No JSON pattern found in section")
        else:
            # FALLBACK: Plain text response (when no structured format)
            result['message'] = full_message.strip()
            print(f"[PARSE DEBUG] No 'ExtractedInfo' found - using plain text fallback")
            
            # Even in fallback, check if there's a Message: prefix
            if result['message'].startswith("Message:"):
                result['message'] = result['message'][8:].strip()
        
        # Check flags more robustly - search entire message
        lines = full_message.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if line_stripped == "is_all_info_collected=true":
                result['flags']['is_all_info_collected'] = True
                print(f"[PARSE DEBUG] Found flag: is_all_info_collected=true")
            elif line_stripped == "end_call=true":
                result['flags']['end_call'] = True
                print(f"[PARSE DEBUG] Found flag: end_call=true")
            elif line_stripped == "handover_to_human=true":
                result['flags']['handover_to_human'] = True
                print(f"[PARSE DEBUG] Found flag: handover_to_human=true")
        
        # Final cleanup - ensure no "Message:" at start
        if result['message'].startswith("Message:"):
            result['message'] = result['message'][8:].strip()
        
        print(f"[PARSE DEBUG] ===== PARSING COMPLETE =====")
        print(f"[PARSE DEBUG] Final message: {result['message'][:100]}...")
        print(f"[PARSE DEBUG] Final JSON: {result['json']}")
        print(f"[PARSE DEBUG] Flags: {result['flags']}")
        
        return result

    async def _generate_recommendations_async(self):
        """Generate recommendations in background without blocking conversation"""
        try:
            print(f"[ASYNC RECO] Starting background recommendation generation")
            
            # Skip cache check for now - there's a function signature issue
            # TODO: Fix cache implementation
            cache_result = False
            
            if cache_result:
                print(f"[ASYNC RECO] Using cached recommendations")
                self.conversation_phase = "post_reco"
                asyncio.create_task(self._save_conversation_state_async())
                return
            
            # Generate recommendations using existing flow
            recommendation_response = await handle_recommendations(
                lead_id=self.lead_id,
                collected_info=self.collected_info,
                supabase=self.supabase
            )
            
            print(f"[ASYNC RECO] Got recommendations: {recommendation_response['type']}")
            print(f"[ASYNC RECO] Response keys: {recommendation_response.keys()}")
            
            if recommendation_response["type"] == "recommendations":
                # Update phase
                self.conversation_phase = "post_reco"
                # Get the voice summary
                voice_summary = recommendation_response.get("voiceResponse", "")
               #  Inject voice summary into the conversation stream
               #  if voice_summary and self._on_speech:  # Check if speech callback exists
                    # Queue the summary to be spoken after current TTS finishes
               #     await self._on_speech({
               #         "type": "assistant_speech",
               #         "text": voice_summary,
               #         "priority": "high"
               #    })
                asyncio.create_task(self._save_conversation_state_async())
                
                # Build insurance summary
                insurance_summary = {
                    "members": self.collected_info.get("who_to_insure", []),
                    "ages": self.collected_info.get("ages", {}),
                    "coverage": self.collected_info.get("sum_assured", 0),
                    "budget": self.collected_info.get("budget", 0),
                    "city": self.collected_info.get("city", "")
                }
                
                # STEP 1: Send top 3 featured recommendations
                recommendation_response_data = {
                    "type": recommendation_response["type"],
                    "voiceResponse": recommendation_response.get("voiceResponse", ""),
                    "message": recommendation_response.get("message", ""),
                    "data": recommendation_response.get("data", []),  # Top 3 featured plans
                    "planMentions": recommendation_response.get("planMentions", [])
                }
                
                # Format response for frontend
                result_json = format_response(recommendation_response_data, insurance_summary, False)
                
                # Debug logging
                print(f'✪ [ASYNC RECO] Formatted response:')
                print(f'  - viewType: {result_json.get("viewType")}')
                print(f'  - message: {result_json.get("message", "")[:50]}...')
                print(f'  - items exists: {bool(result_json.get("items"))}')
                print(f'  - number of plans: {len(recommendation_response.get("data", []))}')
                
                # Send to frontend via WebSocket
                await self._on_plans(result_json)
                print(f'✪ [ASYNC RECO] Sent TOP 3 recommendations to frontend')
                
                # STEP 2: Wait and send ALL plans (if available)
                if recommendation_response.get("allData"):
                    all_plans_count = len(recommendation_response.get('allData', []))
                    #print(f"[ASYNC RECO] Waiting 2 seconds before sending all {all_plans_count} plans...")
                    
                    # Prepare all plans response
                    recommendation_response_allData = {
                        "type": "allplans",  # Different type for viewType 5
                        "voiceResponse": recommendation_response.get("voiceResponse", ""),
                        "message": recommendation_response.get("message", ""),
                        "data": recommendation_response.get("allData", []),  # ALL plans
                        "planMentions": recommendation_response.get("planMentions", [])
                    }
                    
                    # Format all plans response
                    result_json_all = format_response(recommendation_response_allData, insurance_summary, True)
                    
                    # Debug logging
                    print(f'✪ [ASYNC RECO] All plans formatted response:')
                    print(f'  - viewType: {result_json_all.get("viewType")}')
                    print(f'  - number of all plans: {all_plans_count}')
                    
                    # Send all plans to frontend
                    await self._on_plans(result_json_all)
                    print(f'✪ [ASYNC RECO] Sent ALL plans to frontend')
                else:
                    print(f"[ASYNC RECO] No allData found, skipping backup plans")
                    
            else:
                # Handle non-recommendation responses (errors, no plans found, etc.)
                print(f"[ASYNC RECO] Non-recommendation response: {recommendation_response.get('message', '')}")
                # Handle text responses (like "team will contact you")
                result_json = format_response(recommendation_response, None, False)
                await self._on_plans(result_json)
                
        except Exception as e:
            print(f"[ASYNC RECO ERROR] Failed to generate recommendations: {e}")
            import traceback
            traceback.print_exc()
            self.conversation_phase = "ready_for_reco"  # Reset phase on error
    
    async def _save_conversation_state_async(self, extra_fields=None):
        """Save conversation state to DB in background"""
        try:
            # Add detailed logging
            print(f"[DB SAVE START] Attempting to save state for lead_id: {self.lead_id}")
            print(f"[DB SAVE] collected_info: {json.dumps(self.collected_info, indent=2)}")
            print(f"[DB SAVE] conversation_phase: {self.conversation_phase}")
            print(f"[DB SAVE] is_all_info_collected: {self.is_all_info_collected}")
            
            data = {
                "lead_id": self.lead_id,
                "collected_info": self.collected_info,
                "current_phase": self.conversation_phase,
                "unanswered_questions": getattr(self, 'unanswered_questions', []),  # Add this
                "is_all_info_collected": self.is_all_info_collected,
                "last_interaction": datetime.now(timezone.utc).isoformat()
            }
            if extra_fields:
                data.update(extra_fields)
                
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                db_executor,
                lambda: self.supabase.from_("conversation_states").upsert(
                    data, 
                    on_conflict="lead_id"
                ).execute()
            )
        
            print(f"[DB SAVED] Conversation state updated")
            if hasattr(result, 'data') and result.data:
                data = result.data[0] if result.data else {}
                print(f"[DB SAVE RESULT] Phase: {data.get('current_phase')}, Info collected: {data.get('is_all_info_collected')}")
        except Exception as e:
            print(f"[DB ERROR] Failed to save state: {e}")
            import traceback
            traceback.print_exc()

    async def _load_plan_context_for_sales(self):
        """Load detailed plan context for ALL recommended plans"""
        try:
            print("[SALES CONTEXT] Loading detailed plan context for post-reco phase")
            
            # Skip if already loaded
            if hasattr(self, 'sales_context_loaded') and self.sales_context_loaded:
                return
            
            # Get recommendation_data from conversation_states
            loop = asyncio.get_event_loop()
            conv_result = await loop.run_in_executor(
                db_executor,
                lambda: self.supabase.from_('conversation_states')
                    .select('recommendation_data')
                    .eq('lead_id', self.lead_id)
                    .limit(1)
                    .execute()
            )
            
            if not conv_result.data or not conv_result.data[0].get('recommendation_data'):
                print("[SALES CONTEXT] No recommendation data found")
                return
            
            recommendation_data = conv_result.data[0]['recommendation_data']
            # ADD THIS LINE to store it for later use:
            self.recommendation_data = recommendation_data
            
            # Extract all unique product IDs from the structure
            plan_ids = []
            for company_group in recommendation_data:
                # Add featured product
                plan_ids.append(company_group['product_id'])
                # Add all sub_items
                for sub_item in company_group.get('sub_items', []):
                    plan_ids.append(sub_item['product_id'])
            
            print(f"[SALES CONTEXT] Found {len(plan_ids)} total plans to load context for")
            
            # Single DB query for benefits/addons/premium_matrix
            result = await loop.run_in_executor(
                db_executor,
                lambda: self.supabase.from_('india_insurance_products')
                    .select('id, pid, product_name, company, benefits, addons, premium_matrix')
                    .in_('id', plan_ids)
                    .execute()
            )
            
            if not result.data:
                print("[SALES CONTEXT] No product data found")
                return
            
            # Initialize sales context
            self.plan_details = {}
            
            # Get user's sum assured and budget (no defaults)
            user_sa = str(self.collected_info.get('sum_assured', ''))
            user_budget = int(self.collected_info.get('budget', 0))
            
            # Create a map of ALL products (featured + sub_items) by product_id
            reco_map = {}
            
            for company_group in recommendation_data:
                # Add featured product
                reco_map[str(company_group['product_id'])] = company_group
                
                # Add all sub_items
                for sub_item in company_group.get('sub_items', []):
                    # Important: sub_items might not have company name, so inherit it
                    sub_item_with_company = {**sub_item, 'company': company_group['company']}
                    reco_map[str(sub_item['product_id'])] = sub_item_with_company
            
            # Process each plan
            for plan in result.data:
                plan_id = str(plan['id'])
                
                # Get USPs from already processed recommendation data
                reco_item = reco_map.get(plan_id, {})
                existing_usps = reco_item.get('usp', [])
                
                # Extract core benefits (8 fields like JS version)
                benefits_detail = self._extract_core_benefits(plan.get('benefits', {}))
                
                # Extract key addons with label and explanation
                addons_detail = self._extract_key_addons(plan.get('addons', {}))
                
                # Get premium from recommendation data or matrix
                current_premium = reco_item.get('premium')
                if not current_premium and user_sa and plan.get('premium_matrix'):
                    current_premium = plan.get('premium_matrix', {}).get('1', {}).get(user_sa, 0)
                
                # Store detailed context
                self.plan_details[plan_id] = {
                    'name': plan['product_name'],
                    'company': plan['company'],
                    'sum_assured': reco_item.get('sum_assured'),  # ADD THIS LINE
                    'current_premium': current_premium,
                    'monthly_premium': round(current_premium / 12) if current_premium else 0,
                    'usps': existing_usps[:5],  # Limit to 5 USPs
                    'benefits': benefits_detail,
                    'addons': addons_detail,
                    'premium_matrix': plan.get('premium_matrix', {})
                }
            
            # Identify best plan for this user
            # self._identify_best_plan_for_user()
            
            # Create full context for LLM
            self._create_sales_context_for_llm()
            
            self.sales_context_loaded = True
            print(f"[SALES CONTEXT] Loaded context for {len(self.plan_details)} plans")
            
        except Exception as e:
            print(f"[SALES CONTEXT ERROR] Failed to load plan context: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_core_benefits(self, benefits):
        """Extract only the 8 core benefits matching JS version"""
        if not benefits:
            return {}
        
        # Extract exactly like JS version
        core_benefits = {
            'room_rent': benefits.get('room_rent', 'Not specified'),
            'pre_existing': benefits.get('pre_existing_coverage', 'Not covered'),
            'maternity': benefits.get('maternity_benefits', 'Yes') if benefits.get('maternity_option') == 'yes' else 'No',
            'cashless': benefits.get('cashless_facility', 'Not specified'),
            'restoration': 'Yes' if benefits.get('restoration_benefit') == 'yes' else 'No',
            'copay': benefits.get('co_pay', 'Not specified'),
            'opd': benefits.get('opd_limit', 'Yes') if benefits.get('opd_treatment') == 'yes' else 'No',
            'ncb': benefits.get('cumulative_bonus') or benefits.get('no_claim_bonus') or 'Not specified'
        }
        
        return core_benefits
    
    def _extract_key_addons(self, addons):
        """Extract key addons with label and explanation only"""
        if not addons:
            return []
        
        # Key addons to check (from JS version + acute_care and deductible)
        key_addons = [
            'pa_cover', 'ci_cover', 'acute_care', 'maternity_expenses',
            'dm_rider', 'room_modification', 'deductible_addon'
        ]
        
        extracted_addons = []
        
        for addon_key in key_addons:
            if addon_key in addons and addons[addon_key]:
                addon_data = addons[addon_key]
                if isinstance(addon_data, dict):
                    label = addon_data.get('label', addon_key.replace('_', ' ').title())
                    explanation = addon_data.get('explanation', 'Available')
                    extracted_addons.append({
                        'name': label,
                        'details': explanation
                    })
                else:
                    # If addon is just True/string value
                    extracted_addons.append({
                        'name': addon_key.replace('_', ' ').title(),
                        'details': 'Available'
                    })
        
        return extracted_addons
    
 #   def _identify_best_plan_for_user(self):
 #      """Identify the best plan based on user profile"""
 #      if not self.plan_details:
 #           return
        
 #       user_budget = int(self.collected_info.get('budget', 0))
        
        # Simple logic: Find plan closest to budget
 #       best_plan_id = None
 #       best_diff = float('inf')
        
  #      for plan_id, details in self.plan_details.items():
  #          premium = details['current_premium']
  #         if premium and premium <= user_budget * 1.1:  # Allow 10% over budget
  #              diff = abs(user_budget - premium)
  #             if diff < best_diff:
  #                  best_diff = diff
  #                  best_plan_id = plan_id
        
  #     self.best_plan_id = best_plan_id or list(self.plan_details.keys())[0]

    def _create_sales_context_for_llm(self):
        """Create full context with all plan details for LLM"""
        context = {
            "current_recommendations": {
                "user_requirement": {
                    "sum_assured": self.collected_info.get('sum_assured'),
                    "budget": self.collected_info.get('budget'),
                    "family": self.collected_info.get('who_to_insure')
                },
                "recommended_plans": {}
            }
        }
        
        # Add ALL plan details
        for plan_id, details in self.plan_details.items():
            context["current_recommendations"]["recommended_plans"][plan_id] = {
                "name": details['name'],
                "company": details['company'],
                "monthly_premium": details['monthly_premium'],
                "annual_premium": details['current_premium'],
                "usps": details['usps'],
                "benefits": details['benefits'],  # All 8 core benefits
                "addons": details['addons'],      # All key addons
               # "is_best_match": plan_id == self.best_plan_id
            }
        
        # Convert to readable text or JSON string
        self.sales_context_full = json.dumps(context, indent=2)
            
    def _create_sales_context_summary(self):
        """Create a compressed context summary for LLM"""
        if not self.plan_details:
            self.sales_context_summary = "No plans available"
            return
        
        summary_lines = []
        
        # Add best plan first
        # Add inline plans first (these are featured or budget-closest)
        inline_plans = []
        for plan_id, details in self.plan_details.items():
            # Check if this plan is marked as inline in the original recommendation data
            for rec in self.recommendation_data:
                if rec.get('product_id') == plan_id and rec.get('inline'):
                    inline_plans.append(details)
                    break
                # Also check sub_items
                for sub in rec.get('sub_items', []):
                    if sub.get('product_id') == plan_id and sub.get('inline'):
                        inline_plans.append(details)
                        break
        
        if inline_plans:
            summary_lines.append("TOP RECOMMENDED PLANS (shown in carousel):")
            for plan in inline_plans[:3]:  # Max 3 inline
                summary_lines.append(f"• {plan['company']} {plan['name']} - ₹{plan['monthly_premium']}/month")
            summary_lines.append("")
        
        # Add other plans in compressed format
        summary_lines.append("OTHER AVAILABLE PLANS:")
        
        for plan_id, details in self.plan_details.items():
            if plan_id == getattr(self, 'best_plan_id', None):
                continue
                
            # One line per plan with key info
            addon_count = len(details['addons'])
            summary_lines.append(
                f"• {details['company']} {details['name']} - ₹{details['monthly_premium']}/month"
                f" | OPD: {details['benefits'].get('opd', 'No')}"
                f" | Room: {details['benefits'].get('room_rent', 'N/A')}"
                f" | {addon_count} addons"
            )
        
        self.sales_context_summary = "\n".join(summary_lines)

    def _detect_natural_language_intent(self, user_text: str, agent_response: str) -> dict:
        """Detect compare/cart intent from user's natural language"""
        text_lower = user_text.lower()
        agent_lower = agent_response.lower()
        combined_text = text_lower + " " + agent_lower
                
        # Extract mentioned plans
        mentioned_plans = self._extract_plan_names_from_text(agent_response)
        
        # ============= COMPARE DETECTION =============
        has_compare_pattern = any(p in combined_text for p in [
            'compar', 'vs', 'versus', 'differen', 'better', 'than'
        ])
        
        has_which_pattern = (
            ('between' in combined_text and 'which' in combined_text) or  
            ('which' in combined_text and (' or ' in combined_text))
        )
        print(f"[COMPARE DEBUG] User text: {user_text}")
        print(f"[COMPARE DEBUG] Has compare pattern: {has_compare_pattern}")
        print(f"[COMPARE DEBUG] Has which pattern: {has_which_pattern}")
        
        if has_compare_pattern or has_which_pattern:
            print(f"[COMPARE DEBUG] Agent response: {agent_response[:100]}...")
            print(f"[COMPARE DEBUG] Extracted {len(mentioned_plans)} plans: {[p['name'] for p in mentioned_plans]}")
            if len(mentioned_plans) >= 2:
                return {'type': 'compare', 'detected': True, 'plans': mentioned_plans}
        
        # ============= CART DETECTION (from benefits.js) =============
        
        # Buying intent words
        buying_intent_words = [
            'buy', 'purchase', 'go with', 'take', 'want', 
            'proceed with', 'interested in', 'sign me up',
            'book', 'get me', 'add', 'select'
        ]
        
        has_buying_intent = any(word in combined_text for word in buying_intent_words)
        
        # Decision phrases
        decision_phrases = [
            "i'll take", "i want", "i'd like", "let me", 
            "let's go", "i'm interested", "i would like"
        ]
        
        has_decision_phrase = any(phrase in text_lower for phrase in decision_phrases)
        
        # Remove intent patterns
        has_remove_intent = (
            ('remove' in combined_text and ('cart' in combined_text or 'from' in combined_text)) or
            ('delete' in combined_text and 'cart' in combined_text) or
            ('take' in combined_text and 'out' in combined_text) or
            'changed my mind' in text_lower or
            ("don't want" in combined_text and 'anymore' in combined_text) or
            'cancel' in combined_text or
            'drop' in combined_text
        )
        
        # Show cart patterns
        show_cart_patterns = [
            'show cart', 'view cart', 'check cart', 'see cart',
            'display cart', 'open cart', "what's in cart", 
            'what is in cart', 'what have i selected',
            'show me my selections', 'what am i buying',
            'what did i add', 'my cart','kart dikha', 'cart dikha', 'kath dikha',
            'meri cart', 'mera cart', 'meri kath'
        ]

        # Agent patterns that indicate they're SHOWING cart contents
        agent_showing_cart = any(p in agent_lower for p in [
            'cart mein hai',           # "is in cart"
            'cart me hai',
            'your cart contains',
            'your cart has',
            'items in your cart',
            'cart is empty',
            'cart mein sirf',          # "only in cart"
            'nothing in cart',
            'cart summary',
            'here\'s your cart',
            'showing your cart'
        ])
        
        has_show_cart = (
              any(pattern in text_lower for pattern in show_cart_patterns) or 
              agent_showing_cart or  # ✅ Use the variable directly
              text_lower in ['cart', 'my cart', 'my cart?']
        )
        
        # Contextual phrases (yes, that, it, etc)
        contextual_phrases = [
            'that', 'it', 'this one', 'that one', 
            'yes', 'yes please', 'perfect', 'sounds good',
            'ok', 'okay', 'sure', 'great'
        ]
        
        has_contextual = any(phrase == text_lower.strip() for phrase in contextual_phrases)
        
        # Check for negations
        has_negation = "don't" in text_lower or "not" in text_lower

        # CART ADD LOGIC
        if (has_buying_intent or has_decision_phrase) and not has_negation and not has_remove_intent:
            if mentioned_plans:
                return {'type': 'cart_add', 'detected': True, 'plans': mentioned_plans}
            elif has_contextual:
                # Need to resolve from context
                return {'type': 'cart_add', 'detected': True, 'contextual': True}
        
        # CART REMOVE LOGIC  
        if has_remove_intent:
            if mentioned_plans:
                return {'type': 'cart_remove', 'detected': True, 'plans': mentioned_plans}
            else:
                # Remove most recent or resolve from context
                return {'type': 'cart_remove', 'detected': True, 'contextual': True}
        
        # SHOW CART
        if has_show_cart:
            return {'type': 'show_cart', 'detected': True}
        
        # CONTEXTUAL HANDLING (when user just says "yes" etc)
        if has_contextual and len(text_lower.split()) <= 5:
            # Check last agent message for what was offered
            resolved = self._resolve_contextual_plan()
            if resolved['found']:
                # Determine action type from context
                action_type = 'cart_add'  # default
                
                # Check if it was a remove action
                if resolved.get('action_type') == 'added_to_cart' and has_remove_intent:
                    action_type = 'cart_remove'
                
                return {
                    'type': action_type,
                    'detected': True,
                    'plans': [{
                        'id': resolved['plan_id'],
                        'name': resolved['plan_name'],
                        'company': resolved.get('company', '')
                    }],
                    'contextual': True
                }
            # If no plan found in context, return generic contextual
            return {'type': 'contextual_response', 'detected': True}
        
        return {'type': None, 'detected': False}

    def _resolve_contextual_plan(self) -> dict:
        """Resolve plan from recent conversation context"""
        # Look at last 3-4 messages in reverse order
        for i in range(min(5, len(self.history))):
            idx = -(i + 1)  # Start from -1, -2, etc
            msg = self.history[idx]
            
            if msg['role'] == 'assistant':
                msg_text = msg['content'].lower()
                
                # Look for plan mentions in agent's message
                for plan_id, details in self.plan_details.items():
                    plan_name = details['name'].lower()
                    company = details['company'].lower()
                    
                    # Check if plan was mentioned
                    if plan_name in msg_text or company in msg_text:
                        # Extra validation: was it in context of an offer/question?
                        offer_patterns = [
                           'would you like to add',  # More specific
                            'interested in adding',
                            'shall i add',
                            'add to cart',
                            'cart mein daal',
                            'suitable for you',      # OK
                            'best for you',
                        ]
                        
                        if any(pattern in msg_text for pattern in offer_patterns):
                            return {
                                'found': True,
                                'plan_id': plan_id,
                                'plan_name': details['name'],
                                'company': details['company'],
                                'context': msg_text[:100]  # For debugging
                            }
        
        # Also check for recent cart actions in the message
        for i in range(min(3, len(self.history))):
            idx = -(i + 1)
            msg = self.history[idx]
            
            if msg['role'] == 'assistant':
                msg_text = msg['content'].lower()
                
                # Pattern for "I've added X to cart"
                if 'added' in msg_text and 'cart' in msg_text:
                    for plan_id, details in self.plan_details.items():
                        if details['name'].lower() in msg_text:
                            return {
                                'found': True,
                                'plan_id': plan_id,
                                'plan_name': details['name'],
                                'company': details['company'],
                                'action_type': 'added_to_cart'
                            }
        
        return {'found': False}
        
    def _extract_plan_names_from_text(self, text: str) -> list:
        """Plan extraction with smart normalization + word boundaries"""
        found_plans = []
        text_lower = text.lower()
        
        sorted_plans = sorted(
            self.plan_details.items(), 
            key=lambda x: len(x[1]['name']), 
            reverse=True
        )
        
        for plan_id, details in sorted_plans:
            plan_name = details['name']
            plan_name_lower = plan_name.lower()
            
            # Create variations to check
            variations = [
                plan_name_lower,  # exact
                plan_name_lower.replace('+', 'plus'),  # + → plus
                plan_name_lower.replace('plus', '+'),  # plus → +
            ]
            
            for variant in variations:
                # Escape for regex safety
                escaped = re.escape(variant)
                # Word boundary pattern
                pattern = rf'\b{escaped}\b'
                
                if re.search(pattern, text_lower):
                    found_plans.append({
                        'id': plan_id,
                        'name': plan_name,
                        'company': details['company']
                    })
                    text_lower = re.sub(pattern, '[FOUND]', text_lower)
                    break  # Found this plan, move to next
        
        return found_plans
    
    async def _handle_ui_action_for_frontend(self, ui_action):
        """Format data using format_response, then send via _on_plans"""
        
        if ui_action['type'] == 'compare' and ui_action.get('plans'):
            plan_ids = [p['id'] for p in ui_action['plans']]

            print(f"[UI DEBUG] Compare plan_ids: {plan_ids}")
            
            # Update DB
            await self._update_compare_in_db(plan_ids)
            # Build comparison data using the new function
            comparison_data = await self._build_comparison_data_for_frontend(plan_ids)
            
            result = {
                "type": "compare",
                "data": comparison_data,
                "message": ""  # Add empty string instead of None
            }
            print(f"[UI DEBUG] Built comparison data with {len(comparison_data.get('headers', [])) - 1} plans")

            try:
                formatted = format_response(result, None, True)
                print(f"[UI DEBUG] Formatted response type: {type(formatted)}")
                print(f"[UI DEBUG] Formatted response: {formatted}")
                
                if self._on_plans:
                    print(f"[UI DEBUG] Calling _on_plans with formatted response")
                    await self._on_plans(formatted)
                else:
                    print("[UI ERROR] self._on_plans is None!")
                    
            except Exception as e:
                print(f"[UI ERROR] format_response failed: {e}")
                import traceback
                traceback.print_exc()

            
        elif ui_action['type'] in ['cart_add', 'cart_remove', 'show_cart']:
            # Handle cart operations
            if ui_action['type'] == 'cart_add':
                plan_ids = [p['id'] for p in ui_action['plans']]
                await self._update_cart_in_db('add', plan_ids)
            elif ui_action['type'] == 'cart_remove':
                plan_ids = [p['id'] for p in ui_action['plans']] if ui_action.get('plans') else []
                await self._update_cart_in_db('remove', plan_ids)
            
            # Build cart data from existing plan_details
            if self.cart_items:
                cart_items = []
                
                for plan_id in self.cart_items:
                    if plan_id in self.plan_details:
                        details = self.plan_details[plan_id]
                        
                        # Use imported COMPANY_SHORT_NAMES
                        short_company = COMPANY_SHORT_NAMES.get(details['company'], details['company'])
                        
                        cart_item = {
                            "id": plan_id,
                            "company": short_company,
                            "logo": "",  # We don't have logos in plan_details
                            "name": details['name'],
                            "sum_assured": self.collected_info.get('sum_assured'),  # No default
                            "premium": details['current_premium'],
                            "monthly_premium": details['monthly_premium'],
                            "usps": details['usps'][:3],
                            "inline": True,
                            "sub_items": []
                        }
                        cart_items.append(cart_item)
                
                result = {
                    "type": "cart",
                    "data": cart_items,
                    "message": f"You have {len(cart_items)} plan{'s' if len(cart_items) > 1 else ''} in your cart"  # ADD THIS
                }
            else:
                result = {
                    "type": "cart_update",
                    "data": {
                        "itemsSelected": len(self.cart_items),
                        "itemIds": self.cart_items
                    },
                    "message": "Your cart is empty"  # ADD THIS
                }
            
            formatted = format_response(result, None, True)
            await self._on_plans(formatted)


    async def _build_comparison_data_for_frontend(self, plan_ids):
        """Build comparison data in the format expected by frontend"""

        # Fetch logos first
        logo_map = await self._fetch_logos_for_comparison(plan_ids)
        headers = [{
            "companyName": "Comparison",
            "premium": "",
            "coverage": "",
            "companyLogo": ""
        }]
        
        planNames = []
        planIds = []
        premiums = []
        sumAssureds = []
        roomRent = []
        preExisting = []
        cashless = []
        ncb = []
        
        for plan_id in plan_ids:
            if plan_id not in self.plan_details:
                continue
                
            details = self.plan_details[plan_id]
            
            # Add header
            headers.append({
                "companyName": COMPANY_SHORT_NAMES.get(details['company'], details['company']),
                "premium": f"₹{details.get('current_premium', 0):,}",
                "coverage": f"₹{(details.get('sum_assured', 0)/100000):.1f}L",
                "companyLogo": logo_map.get(plan_id, "")  # Use the fetched logo
            })
            
            # Add row values
            planNames.append(details['name'])
            planIds.append(plan_id)
            premiums.append(f"₹{details.get('current_premium', 0):,}")
            sumAssureds.append(f"₹{details.get('sum_assured', 0):,}")
            
            # Extract benefits
            benefits = details.get('benefits', {})
            roomRent.append(benefits.get('room_rent', 'Not specified'))
            preExisting.append(benefits.get('pre_existing', 'Not specified'))
            cashless.append(benefits.get('cashless', 'Not specified'))
            ncb.append(benefits.get('ncb', 'Not specified'))
        
        return {
            "headers": headers,
            "rows": [
                {"label": "Plan", "values": planNames},
                {"label": "Premium", "values": premiums},
                {"label": "Sum Insured", "values": sumAssureds},
                {"label": "Room Rent Limit", "values": roomRent},
                {"label": "Pre-existing Disease", "values": preExisting},
                {"label": "Cashless Network", "values": cashless},
                {"label": "No Claim Bonus", "values": ncb},
                {"label": "Plan Ids", "values": planIds, "hidden": True}
            ],
            "action": [{"label": "Add To Cart", "action": "addToCart"} for _ in planIds],
            "itemIds": plan_ids  # ADD THIS LINE
        }

    async def _fetch_logos_for_comparison(self, plan_ids):
        """Fetch company logos for comparison plans"""
        try:
            # First, get the insurer_ids from the database for these plans
            plan_ids_str = [str(pid) for pid in plan_ids]
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                db_executor,
                lambda: self.supabase
                    .from_('india_insurance_products')
                    .select('id, company, insurer_id')
                    .in_('id', plan_ids_str)
                    .execute()
            )
            
            if not response.data:
                return {}
            
            # Extract unique insurer_ids
            insurer_ids = list(set(
                str(item['insurer_id']) 
                for item in response.data 
                if item.get('insurer_id')
            ))
            
            # Use the existing fetch_company_logos function
            from helpers import fetch_company_logos
            logo_map_by_id = await fetch_company_logos(self.supabase, insurer_ids)
            
            # Create a map of plan_id -> logo_url
            logo_map = {}
            for item in response.data:
                if item.get('insurer_id'):
                    logo_url = logo_map_by_id.get(str(item['insurer_id']), "")
                    logo_map[item['id']] = logo_url
            
            return logo_map
            
        except Exception as e:
            print(f"[LOGO ERROR] Failed to fetch logos: {e}")
            return {}

    async def _update_cart_in_db(self, action='add', plan_ids=None):
        """Update cart in leads_selections table"""
        try:
            loop = asyncio.get_event_loop()
            
            # Get existing cart
            result = await loop.run_in_executor(
                db_executor,
                lambda: self.supabase.from_('leads_selections')
                    .select('cart')
                    .eq('lead_id', self.lead_id)
                    .eq('chat_id', '1')
                    .limit(1)
                    .execute()
            )
            
            existing_cart = []
            if result.data and result.data[0].get('cart'):
                # Use filter to remove empty strings
                existing_cart = [id for id in result.data[0]['cart'].split(',') if id]
            
            # Update cart based on action
            if action == 'add' and plan_ids:
                for plan_id in plan_ids:
                    if plan_id not in existing_cart:
                        existing_cart.append(plan_id)
            elif action == 'remove' and plan_ids:
                existing_cart = [pid for pid in existing_cart if pid not in plan_ids]
            
            # Update database
            cart_data = {
                'lead_id': self.lead_id,
                'chat_id': '1',
                'cart': ','.join(existing_cart)
            }
            
            if result.data:
                # Update existing record
                await loop.run_in_executor(
                    db_executor,
                    lambda: self.supabase.from_('leads_selections')
                        .update({'cart': ','.join(existing_cart)})
                        .eq('lead_id', self.lead_id)
                        .eq('chat_id', '1')
                        .execute()
                )
            else:
                # Insert new record
                cart_data['compare'] = ''  # Initialize empty compare
                await loop.run_in_executor(
                    db_executor,
                    lambda: self.supabase.from_('leads_selections')
                        .insert(cart_data)
                        .execute()
                )
            
            # Update local state
            self.cart_items = existing_cart
            print(f"[DB] Updated cart: {len(self.cart_items)} items")
            
        except Exception as e:
            print(f"[DB ERROR] Failed to update cart: {e}")

    async def _update_compare_in_db(self, plan_ids):
        """Update compare list in leads_selections table"""
        try:
            loop = asyncio.get_event_loop()
            
            # Similar pattern for compare
            compare_data = {
                'lead_id': self.lead_id,
                'chat_id': '1',
                'compare': ','.join(plan_ids)
            }
            
            # Check if record exists
            result = await loop.run_in_executor(
                db_executor,
                lambda: self.supabase.from_('leads_selections')
                    .select('compare')
                    .eq('lead_id', self.lead_id)
                    .eq('chat_id', '1')
                    .limit(1)
                    .execute()
            )
            
            if result.data:
                # Update
                await loop.run_in_executor(
                    db_executor,
                    lambda: self.supabase.from_('leads_selections')
                        .update({'compare': ','.join(plan_ids)})
                        .eq('lead_id', self.lead_id)
                        .eq('chat_id', '1')
                        .execute()
                )
            else:
                # Insert
                compare_data['cart'] = ''
                await loop.run_in_executor(
                    db_executor,
                    lambda: self.supabase.from_('leads_selections')
                        .insert(compare_data)
                        .execute()
                )
            
            self.compare_items = plan_ids
            print(f"[DB] Updated compare: {len(self.compare_items)} items")
            
        except Exception as e:
            print(f"[DB ERROR] Failed to update compare: {e}")
            
    # Create new row in leads table
    def _create_lead_record(self):
        now = datetime.now(timezone.utc).isoformat()
        self.supabase.from_("leads").insert({"id":self.lead_id,
                                             "name":self.customer_name,
                                             "phone_number":self.customer_phone,
                                             "email": self.customer_email,
                                             "created_date":now}).execute()
        
        print(f"[SUPABASE] Created lead {self.lead_id}")

                
    # Resets the conversation history on websocket close.
    def reset(self) -> None:
        #self.history = [{"role":"system","content":self.system_prompt}]

        if not getattr(self, "transcript", None) or len(self.transcript) == 0:
            print("[ℹ️] Skipping transcript insert — transcript is empty.")
            return
    
        note = { 
            "lead_id": self.lead_id,
            "content": "",
            "type": "web_call",
            "call_sid": None,
            "transcript_summary": None,
            "transcript": json.dumps(self.transcript)
        }
        print(f"[DATABASE] inserting lead_notes {note}")
        # Insert into Supabase Lead Notes
        try:            
            res = self.supabase.from_("lead_notes").insert([note]).execute()
        except Exception as e:
            print(f"[❌ SUPABASE] Exception during transcript insert: {e}")

        print(f"[DATABASE] inserting conversation states {self.collected_info}")
        # Insert into Supabase Conversation States
        try:            
            res_1 = self.supabase.from_("conversation_states").upsert({"lead_id":self.lead_id, "collected_info":self.collected_info},on_conflict="lead_id").execute()
        except Exception as e:
            print(f"[❌ SUPABASE] Exception during conversation insert: {e}")

    async def restore_state_after_error(self):
        """Restore LLM state after WebSocket errors"""
        print("[LLM] Restoring state after error...")
        
        # Reload system prompt with current collected info
        try:
            # Ensure we have the latest data
            await self._lazy_load_lead_data()
            # ============ ADD THIS SECTION ============
            # Check if we're in post-reco phase based on recommendations
            if hasattr(self, 'recommended_plans') and self.recommended_plans:
                self.conversation_phase = "post_reco"
                print(f"[RESTORE] Detected recommendations, setting phase to post_reco")
                
                # Load sales context if not already loaded
                if not hasattr(self, 'sales_context_loaded') or not self.sales_context_loaded:
                    print(f"[RESTORE] Loading sales context...")
                    await self._load_plan_context_for_sales()
            # ============ END ADDITION ============
            
            # Rebuild system prompt with current state
            user_details = f"Name: {self.customer_name}, Phone: {self.customer_phone}, Email: {self.customer_email}"
            self.system_prompt = self.base_prompt.replace(
                "{{user_details}}", user_details
            ).replace(
                "{{collected_info}}", json.dumps(self.collected_info)
            ).replace(
                "{{persona_behaviors}}", self.persona
            )
            
            # Reinitialize Gemini model with YOUR EXACT configuration
            genai.configure(api_key=GEMINI_KEY)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-latest",  # YOUR model name
                generation_config={
                    "temperature": 0.2,  # Lower for better instruction following
                    "top_p": 0.8,       # More focused
                    "top_k": 20,        # Stricter vocabulary
                    "max_output_tokens": 1024,
                    "candidate_count": 1,
                    "stop_sequences": ["Translation:", "This means", "In English", "In Hindi"]
                },
                # Add system instruction if your Gemini version supports it
                system_instruction="""You are an insurance sales agent who MUST follow these rules:
                1. DEFAULT to code-mixed Hindi-English (Hinglish) for all responses
                2. ONLY switch to pure English if customer explicitly asks "Please speak in English" or similar
                3. Natural code-mixing is expected - mix Hindi and English fluidly जैसे normal conversation में होता है
                4. Use English for technical terms (premium, coverage) with Hindi explanations
                5. NEVER provide translations in parentheses - just speak naturally in Hinglish
                Breaking these rules will cause system failure."""
            )
            
            print(f"[LLM] State restored. System prompt length: {len(self.system_prompt)}")
            print(f"[LLM] Collected info: {self.collected_info}")
            
        except Exception as e:
            print(f"[ERROR] Failed to restore LLM state: {e}")
    
    async def stream_response(self, user_text: str) -> AsyncGenerator[str, None]:
        # Add state validation
        if not self.system_prompt or "ExtractedInfo" not in self.system_prompt:
            print("[⚠️] System prompt corrupted, restoring state...")
            await self.restore_state_after_error()
        
        t_llm_start = time.perf_counter()
        
        # Ensure lead data is loaded
        await self._lazy_load_lead_data()

        # DEBUG: Check what's loaded
        print(f"[HISTORY CHECK] History length: {len(self.history)}")
        if len(self.history) > 0:
            print(f"[HISTORY CHECK] Last 2 messages:")
            for msg in self.history[-2:]:
                print(f"  - {msg['role']}: {msg['content'][:50]}...")
        print(f"[HISTORY CHECK] Collected info keys: {list(self.collected_info.keys())}")

        # ===== SMART CONTEXT INJECTION =====
        # Check if this might be a context-less message in an ongoing conversation
        if len(self.history) > 1:  # We have conversation history
            # Check if message is short and might lack context
            words = user_text.strip().split()
            if len(words) <= 3:  # Short messages like "Hello", "Yes", "Tell me more"
                # Add conversation state as a system message to Gemini
                context_summary = []
                
                if self.collected_info:
                    # Build context from what we know
                    who = self.collected_info.get('who_to_insure', [])
                    if who:
                        # Convert ["self", "spouse"] to "yourself and spouse"
                        who_text = []
                        for person in who:
                            if person == "self":
                                who_text.append("yourself")
                            else:
                                who_text.append(person)
                        context_summary.append(f"Insurance for {' and '.join(who_text)}")
                    
                    # Add coverage amount if available
                    if 'sum_assured' in self.collected_info:
                        amount = self.collected_info['sum_assured']
                        amount_text = f"{amount//100000}L" if amount >= 100000 else f"{amount//1000}K"
                        context_summary.append(f"{amount_text} coverage")
                    
                    # Add phase-specific context
                    if self.conversation_phase == "post_reco":
                        context_summary.append("reviewing plans")
                    elif self.conversation_phase == "info_collection":
                        # What are we still collecting?
                        required = ['who_to_insure', 'ages', 'sum_assured', 'budget', 'city']
                        missing = [k for k in required if k not in self.collected_info]
                        if missing:
                            # Make it conversational
                            missing_text = {
                                'ages': 'ages',
                                'budget': 'budget',
                                'city': 'location',
                                'sum_assured': 'coverage amount',
                                'who_to_insure': 'family members'
                            }
                            context_summary.append(f"need {missing_text.get(missing[0], missing[0])}")
                
                if context_summary:
                    # Create a natural context string
                    self.conversation_context = " - ".join(context_summary)
                    print(f"[CONTEXT] Added conversation context: {self.conversation_context}")
        # ===== END SMART CONTEXT =====

        try:
            # Check if we need to transition to post_reco phase
            if self.conversation_phase == "ready_for_reco":
                loop = asyncio.get_event_loop()
                lead_res = await loop.run_in_executor(
                    db_executor,
                    lambda: self.supabase.from_("leads")
                        .select("recommended_plans")
                        .eq("id", self.lead_id)
                        .limit(1)
                        .execute()
                )
                if lead_res.data and lead_res.data[0].get("recommended_plans"):
                    print("[PHASE] Transitioning from ready_for_reco to post_reco")
                    self.conversation_phase = "post_reco"
                    asyncio.create_task(self._save_conversation_state_async())
            
            # ============ POST-RECO HANDLING ============
            if self.conversation_phase == "post_reco":
                # Ensure plan context is loaded
                if not hasattr(self, 'sales_context_loaded') or not self.sales_context_loaded:
                    await self._load_plan_context_for_sales()
                
                # Track post-reco interactions
                if not hasattr(self, 'post_reco_interaction_count'):
                    self.post_reco_interaction_count = 0
                
                if self.post_reco_interaction_count == 0:
                    self.post_reco_interaction_count += 1
                    # Proactive nudge
                    nudge_message = "Ye plans aapke budget mein best hain. "
                    # Speak about all inline plans (up to 3)
                    inline_plans = []
                    for plan_id, details in self.plan_details.items():
                        # Check if this plan is inline
                        for rec in self.recommendation_data:
                            if rec.get('product_id') == plan_id and rec.get('inline'):
                                inline_plans.append(details)
                                break
                            for sub in rec.get('sub_items', []):
                                if sub.get('product_id') == plan_id and sub.get('inline'):
                                    inline_plans.append(details)
                                    break
                    
                    if inline_plans:
                        if len(inline_plans) == 1:
                            nudge_message += f"{inline_plans[0]['company']} ka {inline_plans[0]['name']} aapke budget mein best rahega. "
                        elif len(inline_plans) == 2:
                            nudge_message += f"{inline_plans[0]['name']} aur {inline_plans[1]['name']} dono aapke liye suitable hain. "
                        else:  # 3 plans
                            nudge_message += f"Ye teen plans - {inline_plans[0]['name']}, {inline_plans[1]['name']} aur {inline_plans[2]['name']} - aapke budget mein best hain. "
                    
                    nudge_message += "Koi specific question hai?"
                    
                    # Yield the complete message
                    for token in nudge_message.split():
                        yield token + " "
                    
                    # Store in history/transcript for consistency
                    self.history.append({"role": "assistant", "content": nudge_message})
                    self.transcript.append({
                        "role": "agent", 
                        "message": nudge_message, 
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
                    return
            # ============ END POST-RECO HANDLING ============
    
            print("[LLM] Calling Gemini Flash...", user_text)
            # Append user message to history
            self.history.append({"role": "user", "content": user_text})
            self.transcript.append({"role": "user", "message": user_text, "timestamp": datetime.now(timezone.utc).isoformat()})
            
            # Build Gemini history WITH system prompt
            gemini_history = []
            
            # Add system prompt as first message
            gemini_history.append({
                "role": "model",
                "parts": [{"text": self.system_prompt}]
            })
            
            # Add model acknowledgment
            gemini_history.append({
                "role": "model",
                "parts": [{"text": "Ji bilkul, main samajh gaya. Main naturally Hinglish mein baat karunga - Hindi-English mix karke, jaise normal conversation hoti hai. Sirf agar customer English mein baat karne ko kahe tabhi pure English use karunga."}]
            })

            # ===== ADD CONVERSATION CONTEXT HERE (MOVE FROM LINE 68-84) =====
            if hasattr(self, 'conversation_context') and self.conversation_context:
                # Inject the context we built earlier
                context_message = f"[Continuing conversation - Context: {self.conversation_context}]"
                gemini_history.append({
                    "role": "model",
                    "parts": [{"text": context_message}]
                })
                # Clear it so it's only used once
                self.conversation_context = None
                print(f"[CONTEXT] Injected context into Gemini history: {context_message}")
            # ===== END CONTEXT INJECTION =====           
            # ============ ADD POST-RECO CONTEXT HERE ============
            # Inject sales context for post-reco phase
            if self.conversation_phase == "post_reco" and hasattr(self, 'sales_context_full'):
                print(f"[CONTEXT INJECTION] Injecting {len(self.sales_context_full)} chars of context")  # Add this
                context_message = f"""[CONTEXT] 
            IMPORTANT: Insurance plans have ALREADY been displayed to the user. They can see the following plans on their screen:
            
            {self.sales_context_full}
            
            The user is now asking follow-up questions about these already-shown plans. Do NOT act like you need to find or check plans - discuss the specific plans shown above."""
                
                gemini_history.append({
                    "role": "user",
                    "parts": [{"text": context_message}]
                })
                gemini_history.append({
                    "role": "model", 
                    "parts": [{"text": "Samajh gaya! I have all the plan details and will help answer questions about these recommendations."}]
                })
            else:
                print(f"[CONTEXT INJECTION] ❌ NOT injecting - phase: {self.conversation_phase}, has context: {hasattr(self, 'sales_context_full')}")  # ADD
            # ============ END POST-RECO CONTEXT ============
            # Add conversation history
            for m in self.history[:-1]:  # Exclude the last user message we just added
                role = "user" if m["role"] == "user" else "model"
                content = m["content"]
                # Clean content by removing timestamps if they exist
                if content and isinstance(content, str):
                    content = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*?Z?', '', content).strip()
                gemini_history.append({"role": role, "parts": [{"text": content}]})

            # Stream Gemini response
            print(f"[GEMINI DEBUG] System prompt length: {len(self.system_prompt)}")
            print(f"[GEMINI DEBUG] System prompt preview (first 500 chars):\n{self.system_prompt[:500]}")
            print(f"[GEMINI DEBUG] Does prompt contain 'ExtractedInfo'? {'ExtractedInfo' in self.system_prompt}")
            print(f"[GEMINI DEBUG] Does prompt contain 'Message:'? {'Message:' in self.system_prompt}")
            print(f"[GEMINI CALL] Phase: {self.conversation_phase}, Has context: {hasattr(self, 'sales_context_full')}")
            
            stream = self.model.start_chat(history=gemini_history).send_message(user_text, stream=True)
    
            full_message = ""
            for chunk in stream:
                if hasattr(chunk, "text") and chunk.text:
                    full_message += chunk.text
                    print(f"[CHUNK] Got text: {chunk.text[:50]}...")
                else:
                    print(f"[CHUNK] No text in chunk: {chunk}")
                    
            print(f"[FULL MESSAGE] Length: {len(full_message)}")
            print(f"[FULL MESSAGE] Content: {full_message[:200]}...")
            # ADD THE DEBUG LOG HERE - RIGHT AFTER FULL MESSAGE IS COMPLETE
            print(f"[LLM OUTPUT CHECK] Looking for format markers:")
            print(f"  - Contains 'Message:'? {'Message:' in full_message}")
            print(f"  - Contains 'ExtractedInfo'? {'ExtractedInfo' in full_message}")
            print(f"  - Contains '(JSON):'? {'(JSON):' in full_message}")
            print(f"  - Contains 'end_call='? {'end_call=' in full_message}")
            
            # Parse response
            parsed = self._fast_parse_response(full_message)
            extracted_message = parsed['message']
            extracted_json = parsed['json']
            flags = parsed['flags']
            
            print(f"[PARSED] Message: {extracted_message[:100]}...")
            print(f"[PARSED] JSON: {extracted_json}")
            print(f"[PARSED] Flags: {flags}")
            
            # Handle flags
            is_end_call = flags['end_call']
            is_handover_to_human = flags['handover_to_human']
            
            # Handle is_all_info_collected flag
            if flags['is_all_info_collected']:
                if not self.is_all_info_collected:
                    self.is_all_info_collected = True
                    asyncio.create_task(self._save_conversation_state_async())
            else:
                if not self.is_all_info_collected:
                    # Check if we have all required fields
                    required_fields = ['who_to_insure', 'ages', 'sum_assured', 'budget', 'city']
                    all_fields_present = all(
                        self.collected_info.get(field) and 
                        self.collected_info[field] not in [None, "", []] 
                        for field in required_fields
                    )
                    self.is_all_info_collected = all_fields_present
    
            # Store response in history
            self.history.append({"role": "assistant", "content": extracted_message})
            self.transcript.append({"role": "agent", "message": extracted_message, "timestamp": datetime.now(timezone.utc).isoformat()})
            
            # Update collected info
            if not hasattr(self, "collected_info"):
                self.collected_info = {}

             # Handle unanswered questions (for post-reco phase)
            if extracted_json.get('unanswered_questions'):
                if not hasattr(self, 'unanswered_questions'):
                    self.unanswered_questions = []
                # Extend the list with new questions
                self.unanswered_questions.extend(extracted_json['unanswered_questions'])
                print(f"[UNANSWERED] Added questions: {extracted_json['unanswered_questions']}")
                # Remove from extracted_json so it doesn't go into collected_info
                extracted_json.pop('unanswered_questions', None)
                
            if extracted_json:
                self.collected_info.update(extracted_json)
                print(f"[INFO UPDATE] Collected: {extracted_json}")
                # Save after extraction
                if self.conversation_phase == "info_collection":
                    asyncio.create_task(self._save_conversation_state_async())
            
            # Build insurance summary for frontend
            members_covered = []
            if self.collected_info.get("who_to_insure") and self.collected_info.get("ages"):
                for member in self.collected_info["who_to_insure"]:
                    member_data = {
                        "relationship": member,
                        "age": self.collected_info["ages"].get(member)
                    }
                    # Add gender if available
                    if "genders" in self.collected_info and member in self.collected_info["genders"]:
                        member_data["gender"] = self.collected_info["genders"][member]
                    members_covered.append(member_data)
    
            # Calculate budget and sum assured
            requested_budget = int(self.collected_info.get("budget") or 0)
            requested_sum_assured = int(self.collected_info.get("sum_assured") or 0)
    
            def format_sum_assured(value):
                if value >= 1_00_00_000:
                    return f"₹{value // 10000000} Cr"
                elif value >= 1_00_000:
                    return f"₹{value // 100000} L"
                else:
                    return f"₹{value:,}"
                    
            insurance_summary = {
                "membersCovered": members_covered,
                "city": self.collected_info.get("city", "Not specified"),
                "coverage": {
                    "value": requested_sum_assured,
                    "unit": "INR",
                    "label": format_sum_assured(requested_sum_assured),
                    "slider": {
                        "min": 500000,
                        "max": 50000000
                    }
                },
                "budget": {
                    "annual": requested_budget,
                    "currency": "INR",
                    "slider": {
                        "min": 5000,
                        "max": 100000
                    }
                },
                "keyPreferences": {
                    "preferredInsurers": "",
                    "primaryConcerns": "",
                    "implicitNeeds": "",
                    "originalQuery": ""
                }
            }
            
            print(f"All Info Collected: {self.is_all_info_collected}")
            print(f"Current Phase: {self.conversation_phase}")
            
            # Handle phase transition when all info is collected
            if self.conversation_phase == "info_collection" and self.is_all_info_collected:
                # Verify we actually have all required fields
                required_fields = ['who_to_insure', 'ages', 'sum_assured', 'budget', 'city']
                all_fields_present = all(self.collected_info.get(field) for field in required_fields)
                
                if all_fields_present:
                    print(f"[PHASE TRANSITION] All info collected, moving to ready_for_reco")
                    self.conversation_phase = "ready_for_reco"
                    # Save phase transition
                    asyncio.create_task(self._save_conversation_state_async())
                    
                    # Start generating recommendations in background (non-blocking)
                    asyncio.create_task(self._generate_recommendations_async())
                    
                    # Don't block! Continue streaming response
                else:
                    missing = [f for f in required_fields if not self.collected_info.get(f)]
                    print(f"[WARNING] LLM flagged complete but missing: {missing}")
                    self.is_all_info_collected = False  # Reset flag
                    # Save the reset state
                    asyncio.create_task(self._save_conversation_state_async({
                        "missing_info": missing
                    }))
    
            # Stream the clean message from LLM
            clean_message = extracted_message
            # Remove bold markers
            clean_message = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_message)
            
            # Remove single asterisks (italics and bullet points)
            clean_message = re.sub(r'\*([^*]+)\*', r'\1', clean_message)
            
            # Remove bullet point asterisks at start of lines
            clean_message = re.sub(r'^\* ', '', clean_message, flags=re.MULTILINE)
            
            # Remove any remaining asterisks
            clean_message = clean_message.replace('*', '')
            
            # Remove extra colons from headers (like "Maternity::")
            clean_message = re.sub(r'([A-Za-z\s]+):\s*:', r'\1:', clean_message)
            
            # Remove extra whitespace
            clean_message = re.sub(r'\s+', ' ', clean_message).strip()
            for token in clean_message.split():
                yield token + " "

            # ============ POST-RECO UI ACTION DETECTION ============
            if self.conversation_phase == "post_reco" and hasattr(self, 'plan_details') and self.plan_details:
                try:
                    # Store the complete agent response
                    agent_response = extracted_message
                    
                    # Detect intent from user message + agent response
                    ui_action = self._detect_natural_language_intent(user_text, agent_response)
                    
                    if ui_action['detected']:
                        print(f"[UI ACTION] Detected {ui_action['type']} - sending UI update")
                        
                        # Handle contextual resolution if needed
                        if ui_action.get('contextual'):
                            resolved = self._resolve_contextual_plan()
                            if resolved['found']:
                                ui_action['plans'] = [{
                                    'id': resolved['plan_id'],
                                    'name': resolved['plan_name'],
                                    'company': resolved.get('company', '')
                                }]
                            else:
                                print("[UI ACTION] Could not resolve contextual reference")
                                return  # Exit if we can't resolve
                        
                        # Send UI update WITHOUT interrupting TTS
                        await self._handle_ui_action_for_frontend(ui_action)
                        
                except Exception as e:
                    print(f"[UI ACTION ERROR] Failed to handle UI action: {e}")
                    import traceback
                    traceback.print_exc()
            # ============ END UI ACTION DETECTION ============

            # Store the final response if not already done
            if extracted_message and len(self.history) > 0 and self.history[-1]["content"] != extracted_message:
                self.history.append({"role": "assistant", "content": extracted_message})
                self.transcript.append({
                    "role": "agent", 
                    "message": extracted_message, 
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
            t_llm_end = time.perf_counter()
            print(f"[TIMING] LLM inference took {t_llm_end - t_llm_start:.3f} sec")
    
        except Exception as e:
            print(f"[LLM] Error: {e}")
            yield "Sorry, something went wrong with the assistant."
            
            
    # Inserting a row in lead_notes table for each response from user or agent
    async def _insert_mid_call_note_immediate(self):
        try:
            print(f"LINE 1")
            message = self.history[-1]
            print(f"LINE 2")
            entry = [{
                "role": "agent" if message['role'] == 'assistant' else "user",
                "message": message['content'],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
            content = "User Call Message" if message['role'] == 'user' else "Agent Call Message"
            print(f"LINE 3")
            immediate_transcript = json.dumps(entry)
            print(f"LINE 4")
            print(f"Transcript Immediate JSON:\n{immediate_transcript} {message['role']}")

            note = { 
                "lead_id": self.lead_id,
                "content": content,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "type": "web_call",
                "call_sid": None,
                "transcript_summary": None,
                "transcript": immediate_transcript
            }
            print(f"LEAD NOTE: {note}")
            res = self.supabase.from_("lead_notes").insert([note]).execute()
            print(f"LINE 5")
            print("Raw response from Supabase:", res)

            # Check for error in response
            if res.get("status_code", 200) >= 400:
                print(f"[❌ SUPABASE] Insert failed: {res}")
            else:
                print(f"[✔ SUPABASE] Row inserted in lead_notes: {res['data']}")
        
        except Exception as e:
            import traceback
            print("[❌ SUPABASE] Exception during insert:")
            traceback.print_exc()