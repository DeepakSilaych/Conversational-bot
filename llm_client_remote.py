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
from get_recommendation import format_response
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
            
            # Update system prompt with loaded data
            self.system_prompt = self.base_prompt.replace("{{user_details}}", user_details).replace("{{collected_info}}", json.dumps(self.collected_info)).replace("{{persona_behaviors}}", self.persona)
            print(f"[SYSTEM PROMPT] Length: {len(self.system_prompt)}")
            print(f"[SYSTEM PROMPT] First 100 chars: {self.system_prompt[:100]}...")

            # Determine conversation phase based on loaded data
            if res.data and res.data[0].get("recommended_plans"):
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
                    await self._on_plans(result_json_all) #MESSAGE VIEW
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
            if hasattr(result, 'data'):
                print(f"[DB SAVE RESULT] {result.data}")
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
                    'current_premium': current_premium,
                    'monthly_premium': round(current_premium / 12) if current_premium else 0,
                    'usps': existing_usps[:5],  # Limit to 5 USPs
                    'benefits': benefits_detail,
                    'addons': addons_detail,
                    'premium_matrix': plan.get('premium_matrix', {})
                }
            
            # Identify best plan for this user
            self._identify_best_plan_for_user()
            
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
    
    def _identify_best_plan_for_user(self):
        """Identify the best plan based on user profile"""
        if not self.plan_details:
            return
        
        user_budget = int(self.collected_info.get('budget', 0))
        
        # Simple logic: Find plan closest to budget
        best_plan_id = None
        best_diff = float('inf')
        
        for plan_id, details in self.plan_details.items():
            premium = details['current_premium']
            if premium and premium <= user_budget * 1.1:  # Allow 10% over budget
                diff = abs(user_budget - premium)
                if diff < best_diff:
                    best_diff = diff
                    best_plan_id = plan_id
        
        self.best_plan_id = best_plan_id or list(self.plan_details.keys())[0]

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
                "is_best_match": plan_id == self.best_plan_id
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
        if hasattr(self, 'best_plan_id') and self.best_plan_id in self.plan_details:
            best_plan = self.plan_details[self.best_plan_id]
            summary_lines.append(f"RECOMMENDED: {best_plan['name']} - ₹{best_plan['monthly_premium']}/month")
            summary_lines.append(f"Key benefits: {', '.join(best_plan['usps'][:3])}")
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
                    if hasattr(self, 'best_plan_id') and self.best_plan_id in self.plan_details:
                        best_plan = self.plan_details[self.best_plan_id]
                        nudge_message += f"{best_plan['company']} ka plan sabse suitable lag raha hai. "
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

                        
            # ============ ADD POST-RECO CONTEXT HERE ============
            # Inject sales context for post-reco phase
            if self.conversation_phase == "post_reco" and hasattr(self, 'sales_context_full'):
                gemini_history.append({
                    "role": "user",
                    "parts": [{"text": f"[CONTEXT] Current insurance plan details:\n{self.sales_context_full}"}]
                })
                gemini_history.append({
                    "role": "model", 
                    "parts": [{"text": "Samajh gaya! I have all the plan details and will help answer questions about these recommendations."}]
                })
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
            
            print(f"[PARSED] Message: {extracted_message[:300]}...")
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

            await self._on_plans({
                        "message": clean_message,
                        "viewType": 1,
                        "viewTypeName": "simpleIncomingText",
                        "from": "agent",
                        "showMessage": True,
                        "insuranceSummary" : insurance_summary,
                        "time": int(time.time() * 1000)
                    });

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