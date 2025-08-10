# get-recommendation.py

import asyncio
from helpers import extract_family_scenario, find_matching_scenarios, get_recommendations, re_rank_by_original_intent,check_recommendation_cache
from fastapi.responses import JSONResponse
from shared import db_executor
from datetime import datetime, timezone
import time
import json

corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Access-Control-Allow-Credentials": "true",
  "Content-Type": "application/json"
};
CONVERSATION_PHASES = {
  "NEW": 'new',
  "INFO_COLLECTION": 'info_collection',
  "RECOMMENDATIONS_SHOWN": 'recommendations_shown',
  "QA_ACTIVE": 'qa_active',
  "SCHEDULED": 'scheduled',
  "COMPLETED": 'completed'
};

RESPONSE_TYPES = {
  "TEXT": 'text',
  "MESSAGE": 'message',
  "RECOMMENDATIONS": 'recommendations',
  "ALLPLANS": 'allplans',
  "CAROUSEL": 'carousel',
  "COMPARE": 'compare',
  "COMPARE_UPDATE": 'compare_update',
  "CART": 'cart',
  "CART_UPDATE": 'cart_update'
};
VIEW_TYPE_MAP = {
    RESPONSE_TYPES["TEXT"]: { "viewType": 1, "viewTypeName": "simpleIncomingText" },
    RESPONSE_TYPES["MESSAGE"]: { "viewType": 1, "viewTypeName": "simpleIncomingText" },
    RESPONSE_TYPES["RECOMMENDATIONS"]: { "viewType": 2, "viewTypeName": "productCarousel" },
    RESPONSE_TYPES["ALLPLANS"]: { "viewType": 5, "viewTypeName": "productCarousel" },
    RESPONSE_TYPES["CAROUSEL"]: { "viewType": 2, "viewTypeName": "productCarousel" },
    RESPONSE_TYPES["COMPARE"]: { "viewType": 3, "viewTypeName": "compare" },
    RESPONSE_TYPES["COMPARE_UPDATE"]: { "viewType": 3, "viewTypeName": "compare" },
    RESPONSE_TYPES["CART"]: { "viewType": 4, "viewTypeName": "addToCart" },
    RESPONSE_TYPES["CART_UPDATE"]: { "viewType": 4, "viewTypeName": "addToCart" }
}

# Add this company name mapping at the top of the file
COMPANY_SHORT_NAMES = {
    "Niva Bupa Health Insurance Company Limited": "Niva Bupa",
    "Aditya Birla Health Insurance Company Limited": "Aditya Birla",
    "Care Health Insurance Limited": "Care",
    "Star Health And Allied Insurance Company Limited": "Star Health",
    "ManipalCigna Health Insurance Company Limited": "ManipalCigna",
    "HDFC ERGO General Insurance Company Limited": "HDFC Ergo",
    "ICICI Lombard General Insurance Company Limited": "ICICI Lombard",
    "Bajaj Allianz General Insurance Company Limited": "Bajaj Allianz"
}



import re
from typing import List, Dict

def escape_regex(string: str) -> str:
    """Escape special regex characters for safe usage in regex patterns."""
    return re.escape(string)

def linkify_plans(text: str, plan_mentions: List[Dict[str, str]]) -> str:
    if not text or not plan_mentions:
        print("🔗 linkifyPlans: No text or planMentions, returning original text")
        return text

    linked_text = text
    total_replacements = 0

    # Remove duplicates based on productId
    unique_plans = list({p["productId"]: p for p in plan_mentions}.values())

    # Sort by product name length (longest first)
    unique_plans.sort(key=lambda p: len(p["textReference"]), reverse=True)

    for plan in unique_plans:
        product_name = plan["textReference"]
        product_id = plan["productId"]

        if not product_name or product_name not in linked_text:
            lower_text = linked_text.lower()
            lower_product = product_name.lower()
            if lower_product in lower_text:
                print(f"⚠️ Found case-insensitive match for: {product_name}")
            continue

        escaped_name = escape_regex(product_name)
        regex = re.compile(escaped_name)

        def replacement(match):
            nonlocal total_replacements
            offset = match.start()
            before = linked_text[max(0, offset - 50):offset]
            after = linked_text[offset:offset + 50]

            if "](https://ct.com/product_id=" in before or ("[" in before and "](" in after):
                return match.group()

            char_before = linked_text[offset - 1] if offset > 0 else ' '
            char_after = linked_text[offset + len(match.group())] if offset + len(match.group()) < len(linked_text) else ' '

            is_word_boundary = re.compile(r'[\s,.\-!?:;()\[\]]')
            if not is_word_boundary.match(char_before) and offset != 0:
                return match.group()

            total_replacements += 1
            return f"[{match.group()}](https://ct.com/product_id={product_id})"

        linked_text = regex.sub(replacement, linked_text)

    return linked_text

def format_response(result: dict, summary=None, is_ui_action=False):
    view_info = VIEW_TYPE_MAP.get(result.get("type"), VIEW_TYPE_MAP["text"])

    # Process message to add hyperlinks if planMentions exist
    processed_message = result.get("message") or ""
    if processed_message and result.get("planMentions"):
        print(f"🔗 Adding hyperlinks for {len(result['planMentions'])} plan mentions")
        #processed_message = linkify_plans(processed_message, result["planMentions"])
    response = {
        "success": True,
        **view_info,
        "time": int(time.time() * 1000),
        "from": "agent",
        "message": processed_message,
        "showMessage": False if is_ui_action else bool(processed_message and processed_message.strip()),
        "headers": corsHeaders,
    }
    # Preserve triggerNext if present
    if "triggerNext" in result:
        response["triggerNext"] = result["triggerNext"]
    # Handle different response types
    result_type = result.get("type")
    data = result.get("data")
    if result_type in [RESPONSE_TYPES["COMPARE_UPDATE"], RESPONSE_TYPES["CART_UPDATE"]]:
        if data:
            response["itemsSelected"] = data.get("itemsSelected")
            response["itemIds"] = data.get("itemIds")
    elif result_type == RESPONSE_TYPES["COMPARE"]:
        if data:
            if "itemIds" in data:
                response["itemIds"] = data["itemIds"]
                data_for_items = data.copy()
                data_for_items.pop("itemIds", None)
                response["items"] = json.dumps(data_for_items)
            else:
                response["items"] = json.dumps(data)

    elif result_type == RESPONSE_TYPES["CART"]:
        response["cartMessage"] = result.get("message")
        if data:
            response["items"] = json.dumps(data)
        # ADD THIS LINE:
        if "itemIds" in result:
            response["itemIds"] = result["itemIds"]
        print(f"🛒 Cart response viewType: {response.get('viewType')}")

    else:
        if data:
            response["items"] = json.dumps(data)

    # Add insurance summary if present
    if summary:
        response["insuranceSummary"] = summary

    # Add voice message if message exists
    if response.get("message"):
        response["voiceMessage"] = result.get("voiceResponse")
    return response

async def handle_recommendations(lead_id, collected_info, supabase):
    print(f'✪ Generating recommendations for lead: {lead_id}')

    try:
        t_llm_start = time.perf_counter()
        loop = asyncio.get_event_loop()  # ADD THIS LINE
        # state = supabase.from_('conversation_states').select('*').eq('lead_id', lead_id).maybe_single().execute()
        # conversation_state = getattr(state, "data", None)

        # print(f'✪ [RECO] Got conversation_state')
        # # Check if cached
        # if conversation_state and check_recommendation_cache(lead_id, conversation_state):
        #     print('✅ Using cached recommendations')
        #     t_llm_end = time.perf_counter()
        #     print(f"[TIMING] RETURN FROM RECO CACHE {t_llm_end - t_llm_start:.3f} sec")
        #     return {
        #         "type": VIEW_TYPE_MAP[RESPONSE_TYPES["RECOMMENDATIONS"]],
        #         "message": conversation_state['recommendation_summary'],
        #         "voiceMessage": conversation_state.get('voice_summary'),
        #         "data": conversation_state['recommendation_data'],
        #         "planMentions": conversation_state.get('plan_mentions', [])
        #     }
        print(f'✪ [RECO] No cache found')
        family_scenario = await extract_family_scenario(collected_info)
        family_scenario['lead_id'] = lead_id

        if not family_scenario['is_ready']:
            return {
                "type": VIEW_TYPE_MAP[RESPONSE_TYPES["TEXT"]],
                "message": f"We're missing some info: {', '.join(family_scenario['missing_info'])}.",
                "data": None
            }
        matching_scenarios = await find_matching_scenarios(supabase, family_scenario)
        if not matching_scenarios:
            await loop.run_in_executor(
                db_executor,
                lambda: supabase.from_('leads').update({'scenario_ids': ''}).eq('id', lead_id).execute()
            )
            return {
                "type": "text",
                "message": "No matching plans found. Our team will contact you shortly.",
                "data": None
            }
        
        scenario_ids = ','.join([str(s['id']) for s in matching_scenarios])
        
        print(f'✪ [RECO] CALLING get_recommendations {scenario_ids}')
        recommendations = await get_recommendations(
            supabase,
            scenario_ids,
            int(family_scenario['preferred_sum_assured']),
            int(collected_info['budget'])
        )
        t_llm_end = time.perf_counter()
        print(f"[TIMING] AFTER GET RECOMMENDATION {t_llm_end - t_llm_start:.3f} sec")
        if not recommendations:
            return {
                "type": "text",
                "message": "No plans found within your budget. Would you like to see slightly higher premium options?",
                "data": None
            }

        # try:
        #     result = await re_rank_by_original_intent(
        #         recommendations,
        #         conversation_state.get('original_query'),
        #         conversation_state.get('query_intent', {}),
        #         family_scenario
        #     )
            
        #     final_recommendations = result['recommendations']
        #     summary = result.get('summary')
        #     voice_summary = result.get('voiceSummary')
        #     plan_mentions = result.get('planMentions', [])
        #     t_llm_end = time.perf_counter()
        #     print(f"[TIMING] AFTER RERANK {t_llm_end - t_llm_start:.3f} sec")
        # except Exception as e:
        final_recommendations = recommendations
        budget = family_scenario['budget']
        
        # Inline selection logic
        try:
            # Step 1: Count featured products
            featured_products = []
            all_products = []
            
            for rec in final_recommendations:
                # Collect featured products
                if rec.get('featured', False):
                    featured_products.append(rec)
                all_products.append(rec)
                
                # Also check sub_items
                for sub_item in rec.get('sub_items', []):
                    all_products.append(sub_item)
            
            print(f"[INLINE] Found {len(featured_products)} featured products out of {len(all_products)} total")
            
            # Step 2: If we have featured products, mark them as inline (up to 3)
            inline_count = 0
            if featured_products:
                for product in featured_products:
                    if inline_count < 3:
                        product['inline'] = True
                        inline_count += 1
                        print(f"[INLINE] Marked featured: {product['company']} - {product['product_name']}")
                        
            # Step 3: If we need more inline products, find closest to budget
            if inline_count < 3:
                user_budget = budget  # Already available from family_scenario['budget']
                
                # Create list of (product, premium, budget_diff) tuples for non-inline products
                remaining_products = []
                for product in all_products:
                    if not product.get('inline', False):
                        premium = product['premium']
                        # Calculate absolute difference from budget
                        budget_diff = abs(user_budget - premium)
                        remaining_products.append((product, premium, budget_diff))
                
                # Sort by budget difference (closest first)
                remaining_products.sort(key=lambda x: x[2])
                
                # Mark closest to budget as inline until we have 3
                for product, premium, diff in remaining_products:
                    if inline_count < 3:
                        product['inline'] = True
                        inline_count += 1
                        print(f"[INLINE] Marked budget-closest: {product.get('company', 'Unknown')} - {product.get('product_name', 'Unknown')} at ₹{premium} (diff: ₹{diff})")
            
            # Step 4: Ensure all other products have inline=False
            for product in all_products:
                if 'inline' not in product:
                    product['inline'] = False
            print(f"[INLINE] Total inline products: {inline_count}")
            
            # ADD DEBUG LOGGING HERE:
            print("[INLINE DEBUG] Products marked as inline:")
            for rec in final_recommendations:
                if rec.get('inline'):
                    print(f"  - Featured: {rec['company']} - {rec['product_name']} (ID: {rec['product_id']})")
                for sub in rec.get('sub_items', []):
                    if sub.get('inline'):
                        print(f"  - Sub-item: {sub.get('company', rec['company'])} - {sub['product_name']} (ID: {sub['product_id']})")
            
        except Exception as e:
            print(f"[ERROR] Inline selection failed: {e}")
            # Fallback: mark all as not inline
            for rec in final_recommendations:
                rec['inline'] = False
                for sub in rec.get('sub_items', []):
                    sub['inline'] = False
                    
        # GENERATE SUMMARY AND PLAN_MENTIONS
        inline_products = []
        plan_mentions = []

        for rec in final_recommendations:
            if rec.get('inline', False):
                inline_products.append(rec)
                plan_mentions.append({
                    "textReference": COMPANY_SHORT_NAMES.get(rec['company'], rec['company'].split()[0]),
                    "productId": rec['product_id'],
                    "productName": rec.get('product_name', ''),
                    "premium": rec.get('premium', 0)
                })
            
            for sub in rec.get('sub_items', []):
                if sub.get('inline', False):
                    inline_products.append(sub)
                    plan_mentions.append({
                        "textReference": COMPANY_SHORT_NAMES.get(rec['company'], rec['company'].split()[0]),
                        "productId": sub['product_id'],
                        "productName": sub.get('product_name', ''),
                        "premium": sub.get('premium', 0)
                    })

        # BETTER SUMMARY
        if inline_products:
            inline_products.sort(key=lambda x: x['premium'])
            summary_parts = []
            for i, product in enumerate(inline_products[:3]):
                short_name = COMPANY_SHORT_NAMES.get(product.get('company', ''), product.get('company', '').split()[0])
                product_name = product.get('product_name', '').replace(product.get('company', ''), '').strip()
                summary_parts.append(f"{short_name}'s {product_name} at ₹{product['premium']:,}")
            
            coverage_lakhs = family_scenario['preferred_sum_assured'] / 100000
            family_desc = family_scenario['family_composition']
            
            summary = f"Based on your need for ₹{coverage_lakhs:.0f}L coverage for {family_desc}, I found these best options: {', '.join(summary_parts)}. Each offers comprehensive coverage within your ₹{budget:,} budget."
            # Shorter voice summary
            first_product = inline_products[0]
            first_short_name = COMPANY_SHORT_NAMES.get(first_product.get('company', ''), first_product.get('company', '').split()[0])
            voice_summary = f"Main aapke liye {len(inline_products)} best plans dhundhe hain. {first_short_name} ka plan sabse affordable hai ₹{first_product['premium']:,} mein."
        else:
            summary = "Based on your requirements, here are the best health insurance plans."
            voice_summary = summary

        # PREPARE DATA (KEEPING ORIGINAL STRUCTURE)
        recommended_plan_ids = []
        recommended_plans = []
        backup_plan_ids = []
        backup_plans = []
        
        product_map = {}
      
        # Step 1: Flatten all unique product_id → item (first seen wins)
        for rec in final_recommendations:
           top_item = {**rec}
           top_item.pop("sub_items", None)
           pid = top_item["product_id"]
           if pid not in product_map:
              product_map[pid] = top_item

           for sub in rec.get("sub_items", []):
              sub_pid = sub["product_id"]
              if sub_pid not in product_map:
                 product_map[sub_pid] = sub

        for pid, item in product_map.items():
           if item.get("inline") is True and len(recommended_plan_ids) < 3:
              recommended_plan_ids.append(pid)
              recommended_plans.append(item)
              print(f"[RECO DEBUG] Adding to recommended_plans: {item.get('company')} - {item.get('product_name')} (ID: {pid})")
           else:
              backup_plan_ids.append(pid)
              backup_plans.append(item)

        # ADD THIS AFTER THE LOOP:
        print(f"[RECO DEBUG] Final recommended_plan_ids: {recommended_plan_ids}")

        # print(f"Flat {product_map} {len(product_map)}")
        # print(f"Recommended Ids {recommended_plan_ids}")
        # print(f"BackUp Ids {backup_plan_ids}")
        # print(f"[RECO] Recommended Plans {len(recommended_plans)}")
        # print(f"[RECO] BackUp Plans {len(backup_plans)}")
        # ✅ Update Supabase - # Fire and forget

        # UPDATE ONLY LEADS TABLE - llm_client handles conversation state
        await loop.run_in_executor(
            db_executor,
            lambda: supabase.from_('leads').update({
                'scenario_ids': scenario_ids,
                'preferred_sum_assured': int(family_scenario['preferred_sum_assured']),
                'budget': int(collected_info['budget']),
                'recommended_plans': ','.join(recommended_plan_ids),
                'backup_plans': ','.join(backup_plan_ids)
            }).eq('id', lead_id).execute()
        )    

        # Fire and forget - store recommendation cache in conversation_states
        loop.run_in_executor(
            db_executor,
            lambda: supabase.from_('conversation_states')
                .update({
                    'recommendation_data': final_recommendations,
                    'recommendation_summary': summary,
                    'voice_summary': voice_summary,
                    'plan_mentions': plan_mentions,
                    'last_recommendation_timestamp': datetime.now(timezone.utc).isoformat(),
                    'last_interaction': datetime.now(timezone.utc).isoformat(),  # Add this
                    'scenario_ids': scenario_ids  # Optional - add if you want it here too
                })
                .eq('lead_id', lead_id)
                .execute()
        )
        t_llm_end = time.perf_counter()
        print(f"[TIMING] AFTER TABLE UPDATE {t_llm_end - t_llm_start:.3f} sec")
        return {
            "type": "recommendations",
            "message": summary,
            "voiceResponse": voice_summary,
            "data": recommended_plans,     # Full grouped structure for viewType 2
            "allData": final_recommendations,  # Same full grouped structure for viewType 5
            "planMentions": plan_mentions
        }

    except Exception as e:
        print('Recommendation error:', e)
        return {
            "type": RESPONSE_TYPES["TEXT"],
            "message": "Something went wrong while generating recommendations.",
            "data": None
        }
