# helpers.py

COMPANY_PRIORITY = {
    "Niva Bupa Health Insurance Company Limited": 1,
    "Aditya Birla Health Insurance Company Limited": 2,
    "Care Health Insurance Limited": 3,
    "Star Health And Allied Insurance Company Limited": 4,
    "ManipalCigna Health Insurance Company Limited": 5
}
import os
import re
import datetime
import google.generativeai as genai
import json
from dotenv import load_dotenv
import time
   # Add these imports
import asyncio
from shared import db_executor

load_dotenv()
GEMINI_KEY = os.environ["GEMINI_API_KEY"]

# Gemini init
#GEMINI_KEY = os.environ("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)

async def fetch_company_logos(supabase, insurer_ids):
    valid_ids = [str(i) for i in insurer_ids if i is not None]

    if not valid_ids:
        return {}

    print(f" 🖼️ Fetching logos for companies {valid_ids}")

    try:
        # Use non-blocking pattern like the rest of the code
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            db_executor,
            lambda: supabase \
                .from_('india_insurance_company') \
                .select('id, name, logo') \
                .in_('id', valid_ids) \
                .execute()
        )

        data = response.data if hasattr(response, 'data') else response.get('data')

        if not data:
            print("⚠️ No data found or error occurred")
            return {}

        logo_map = {str(item['id']): item['logo'] for item in data if item.get('logo')}
        return logo_map

    except Exception as e:
        print(f"❌ Failed to fetch company logos: {e}")
        return {}

def check_recommendation_cache(conversation_state, lead):
    if not lead or not lead.get('recommended_plans'):
         return False
    if not conversation_state.get('recommendation_summary') or not conversation_state.get('recommendation_data'):
        return False
    if conversation_state.get('last_recommendation_timestamp'):
        cache_age = datetime.datetime.now().timestamp() - datetime.datetime.fromisoformat(conversation_state['last_recommendation_timestamp']).timestamp()
        if cache_age > 86400:
            return False
    original = conversation_state.get('original_query')
    last_ranked = conversation_state.get('last_ranked_query')

    if original is None and last_ranked is None:
    # handle special case, if needed
        return True  # or False depending on your needs

    if original != last_ranked:
        return False
    return True

async def find_matching_scenarios(supabase, family_scenario):
    
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        db_executor,
        lambda: supabase.from_('india_scenarios') \
            .select('id, city, eldest_age, has_spouse, num_children, sa_value') \
            .ilike('city', family_scenario['city']) \
            .eq('eldest_age', family_scenario['scenario_age']) \
            .eq('has_spouse', family_scenario['has_spouse']) \
            .eq('num_children', family_scenario['num_children']) \
            .lte('sa_value', int(family_scenario['max_sa'])) \
            .gte('sa_value', int(family_scenario['preferred_sum_assured'] * 0.8)).execute()
    )
    
    return response.data if response else []

async def extract_family_scenario(collected_info):
    missing_info = []

    if not collected_info.get("city"):
        missing_info.append("city")
    if not collected_info.get("sum_assured"):
        missing_info.append("sum assured")
    if not collected_info.get("budget"):
        missing_info.append("budget")

    if missing_info:
        return {"is_ready": False, "missing_info": missing_info}

    # Get eldest age
    eldest_age = 0
    if collected_info.get("ages"):
        eldest_age = max(collected_info["ages"].values())

    if not eldest_age or eldest_age == 0:
        return {"is_ready": False, "missing_info": ["ages of family members"]}

    # Map age bucket
    def map_to_scenario_age(actual_age):
        # Special handling for young adults
        if 18 <= actual_age <= 30:
            return 26  # All ages 18-30 map to bucket 26
        
        buckets = [(31, 35), (36, 40), (41, 45), (46, 50),
                   (51, 55), (56, 60), (61, 65), (66, 200)]
        for b in buckets:
            if b[0] <= actual_age <= b[1]:
                return b[0]
        return None

    scenario_age = map_to_scenario_age(eldest_age)
    if not scenario_age:
        return {"is_ready": False, "missing_info": ["valid age range (25+ years)"]}

    adults, children = 0, 0
    if collected_info.get("ages"):
        for age in collected_info["ages"].values():
            if age >= 18:
                adults += 1
            else:
                children += 1
    elif collected_info.get("family_size"):
        adults = collected_info["family_size"]
    elif collected_info.get("who_to_insure"):
        for p in collected_info["who_to_insure"]:
            if p == "child":
                children += 1
            else:
                adults += 1

    has_spouse = adults >= 2
    family_comp = f"{adults} Adult{'s' if adults > 1 else ''}"
    if children > 0:
        family_comp += f" + {children} Child{'ren' if children > 1 else ''}"

    return {
        "is_ready": True,
        "city": collected_info["city"],
        "eldest_age": eldest_age,
        "scenario_age": scenario_age,
        "has_spouse": has_spouse,
        "num_children": children,
        "preferred_sum_assured": int(collected_info["sum_assured"]),
        "budget": int(collected_info["budget"]),
        "family_composition": family_comp,
        "max_sa": round(int(collected_info["sum_assured"]) * 1.5),
        "max_premium": round(int(collected_info["budget"]) * 1.2)
    }

def convert_usp_to_array(usp_string):
    if isinstance(usp_string, list):
        return usp_string
    if not usp_string:
        return []

    # Handle numbered lists: "1. " → "1) "
    processed = re.sub(r"(\d+)\.\s+", r"\1) ", usp_string)
    
    # Split on delimiters
    usp_array = re.split(r"[,;•]|\s+(?=\d+\))", processed)
    
    # Clean and filter
    return [
        re.sub(r"\.$", "", re.sub(r"^\d+\)\s*", "", item.strip()))
        for item in usp_array
        if item.strip()
    ]

def sort_recommendations_by_company_priority(recommendations):
    return sorted(recommendations, key=lambda x: COMPANY_PRIORITY.get(x['company'], 999))


async def get_recommendations(supabase, scenario_ids, sum_assured, budget):
    t_llm_start = time.perf_counter()
    print(f'✪ [RECO] CALLING SUPABASE {scenario_ids}')
    scenario_ids_int = [int(x) for x in scenario_ids.split(",")]

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        db_executor,
        lambda: supabase.from_('india_product_availability') \
            .select("* , india_insurance_products:product_id (id, pid, product_name, company, insurer_id, product_type, sa_available, variants, benefits, usp, premium_matrix, featured, active, created_at)") \
            .in_('scenario_id', scenario_ids_int).execute()
    )  
    # Filter in Python
    all_recs = [
        row for row in response.data 
        if row.get("india_insurance_products")
    ]
    if not all_recs:
        return []
    t_llm_end = time.perf_counter()
    print(f"[TIMING] TIME FOR QUERY {t_llm_end - t_llm_start:.3f} sec")
    max_premium = round(budget * 1.2)

    # Fixed code (SA RANGE 0.8x to 1.5x)
    min_sa = int(sum_assured * 0.8)  # 800,000
    max_sa = int(sum_assured * 1.5)  # 1,500,000
    product_map = {}
    print(f"[FILTER] Budget: {budget}, Max Premium: {max_premium}")
    print(f"[FILTER] SA Range: {min_sa} to {max_sa}")
    
    for item in all_recs:
        product = item.get('india_insurance_products')
        if not product:
            continue

        premium_matrix_1 = product.get('premium_matrix', {}).get('1', {})
        
        # Find best premium in SA range
        best_premium = None
        best_sa = None
        
        for sa_str, premium_value in premium_matrix_1.items():
            try:
                sa_val = int(sa_str)
                if min_sa <= sa_val <= max_sa and premium_value and premium_value <= max_premium:
                    # Pick the cheapest premium or closest to requested SA
                    if best_premium is None or premium_value < best_premium:
                        best_premium = premium_value
                        best_sa = sa_val
            except (ValueError, TypeError):
                continue
        
        if not best_premium:
            continue
        
        premium = best_premium
        base_name = product['product_name'].replace(" - Direct", "").replace(" Direct", "")
        key = f"{product['company']}_{base_name}"
        if key not in product_map:
            product_map[key] = {"regular": None, "direct": None, "company": product['company'], "base_name": base_name}
        variant = {"product": product, "premium": premium}
        if "Direct" in product['product_name']:
            product_map[key]['direct'] = variant
        else:
            product_map[key]['regular'] = variant
   #unique_companies = {item['india_insurance_products']['insurer_id'] for item in all_recs if item['india_insurance_products']}
    # NEW (correct - uses products that passed filters)
    unique_companies = set()
    for key, variants in product_map.items():
        for variant_type in ['regular', 'direct']:
            if variants[variant_type]:
                product = variants[variant_type]['product']
                if product.get('insurer_id'):
                    unique_companies.add(product['insurer_id'])
    #print(f"Unique Insurers {unique_companies}")
    logo_map = await fetch_company_logos(supabase, list(unique_companies))
    #print(f"Logo Map {logo_map}")

    grouped = {}
    for key, variants in product_map.items():
        for variant_type in ['regular', 'direct']:
            variant_data = variants[variant_type]
            if not variant_data:
                continue
            product = variant_data['product']
            company = product['company']
            insurer_id = product['insurer_id']
            item_data = {
                "company": company,
                "company_logo": logo_map.get(str(insurer_id), ""),  
                "variant_name": "Direct" if variant_type == "direct" else "Regular",
                "product_id": product['id'],
                "pid": product['pid'],
                "product_name": product['product_name'],
                "usp": convert_usp_to_array(product['usp'])[:5],
                "premium": variant_data['premium'],
                "discount": 0,
                "sum_assured": sum_assured,
                "base_product_name": product['product_name'],
                "tenure": 1,
                "featured": product['featured'],
                "inline": False,  # Will be set later by selection logic
                "buttons": [
                    {"label": "Compare", "action": "compare", "subAction": "", "isMessage": False, "message": ""},
                    {"label": "Add to Cart", "action": "addToCart", "variant": "primary", "subAction": "", "isMessage": False, "message": ""}
                ]
            }
            if company not in grouped:
                grouped[company] = {
                    "company": company,
                    "company_logo": logo_map.get(str(insurer_id), ""),
                    **item_data,
                    "featured": product['featured'],
                    "inline":product['featured'],
                    "sub_items": []
                }
            else:
                current = grouped[company]
                if abs(variant_data['premium'] - budget) < abs(current['premium'] - budget):
                    grouped[company]['sub_items'].append({k: current[k] for k in item_data})
                    grouped[company].update(item_data)
                else:
                    grouped[company]['sub_items'].append(item_data)
    #print(f"Recommendations JSON : {list(grouped.values())}")
    return list(grouped.values()); #sort_recommendations_by_company_priority(list(grouped.values()))

async def re_rank_by_original_intent(recommendations, original_query, query_intent, family_scenario):
    all_products = []
    for rec in recommendations:
        #rec['isOriginallyFeatured'] = True
      
        all_products.append(rec)
        for sub in rec.get('sub_items', []):
            sub['company'] = rec['company']
            #sub['company_logo'] = rec['company_logo']
            #sub['isOriginallyFeatured'] = False
            all_products.append(sub)
            
    prompt = f"""
You are an insurance expert. Analyze and re-rank products for a user.
Query: {original_query}
Concerns: {query_intent}
Budget: ₹{family_scenario['budget']}
Family: {family_scenario['family_composition']}

Products:
{chr(10).join([f"{i+1}. {p['company']} - {p['product_name']} (₹{p['premium']})" for i, p in enumerate(all_products)])}

Return JSON:
{{
  "newOrder": [1,2,...],
  "summary": "text...",
  "voiceSummary": "voice...",
  "planMentions": [{{"textReference": "ReAssure", "productId": "uuid"}}]
}}
"""
    try:
        geminiModel = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config={
                "temperature": 0.6,
                "top_p": 1,
                "top_k": 32,
                "max_output_tokens": 1024
            }
        )

        stream = geminiModel.start_chat().send_message(prompt)
        raw = stream.text
         # Step 1: Clean triple backticks and optional "json" tag
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()

        # Step 2: Extract the actual JSON using regex
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if not json_match:
            raise ValueError("No JSON block found in response")
        
        json_str = json_match.group(0)

        # Step 3: Parse JSON safely
        try:
            response = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON: {e}")
        
        order = response.get('newOrder', list(range(1, len(all_products)+1)))
        reordered = [all_products[i-1] for i in order if 0 < i <= len(all_products)]

        top3_product_ids = [p["product_id"] for p in reordered[:3]]
        # Regroup
        company_map = {}
        for p in reordered:
            is_top3 = p["product_id"] in top3_product_ids
            item = {
                **p,
                "inline": is_top3,
                "featured": False
            }

            if p["company"] not in company_map:
                # First product of the company becomes featured
                company_map[p["company"]] = {
                    **p,
                    "inline": is_top3,
                    "featured": True,
                    "sub_items": []
                }
            else:
                company_map[p["company"]]["sub_items"].append(item)

        return {
            "recommendations": sort_recommendations_by_company_priority(list(company_map.values())),
            "summary": response.get('summary'),
            "voiceSummary": response.get('voiceSummary'),
            "planMentions": response.get('planMentions', [])
        }
    except Exception as e:
        print('Re-rank failed:', e)
        return {
            "recommendations": recommendations,
            "summary": "Based on your preferences, here are some plans.",
            "voiceSummary": "Here are some plans that might suit you.",
            "planMentions": []
        }
