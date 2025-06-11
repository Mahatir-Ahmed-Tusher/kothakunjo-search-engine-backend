import aiohttp
import json
import serpapi
import re
from typing import Dict, List
from utils import format_serp_results
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def search_and_present(
    query: str,
    mistral_api_key: str,
    openrouter_api_key: str,
    serpapi_api_key: str,
    language: str = "bn"
) -> Dict[str, str | List[str]]:
    """
    Search the web using SerpAPI, generate a summary with Mistral, translate to Bengali with Grok.
    Returns: {"summary": "...", "sources": [...]}
    """
    # Step 1: Search authoritative sources
    search_query = f"lang_bn:{query}"  # Force Bengali results where possible
    
    try:
        client = serpapi.Client(api_key=serpapi_api_key)
        results = client.search(
            q=search_query,
            num=5,
            hl=language,
            engine="google"
        )
        search_results = results.get("organic_results", [])
    except Exception as e:
        raise Exception(f"সার্চ ত্রুটি: {str(e)}")

    # Step 2: Format results for LLM
    formatted_results = format_serp_results(search_results[:3])

    # Step 3: Generate English summary with Mistral
    specialist_prompt = f"""You are a search result summarizer. Analyze the following search results and provide a concise summary in English, including key points and citations where applicable:
    {formatted_results}
    Return ONLY a valid JSON object with:
    {{
        "summary": "1-2 paragraph summary in English",
        "sources": "List of up to 3 URLs"
    }}"""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {mistral_api_key}",
            },
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": specialist_prompt}],
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.9
            }
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Mistral API error: {response.status} - {error_text}")
            data = await response.json()
            logger.info(f"Raw Mistral Response: {data}")  # Debug log
            content = data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise Exception("প্রক্রিয়াকরণ ত্রুটি: Mistral প্রতিক্রিয়া খালি")
            try:
                english_response = json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    english_response = json.loads(json_match.group(0))
                else:
                    raise Exception("প্রক্রিয়াকরণ ত্রুটি: বৈধ JSON পাওয়া যায়নি")

    # Step 4: Translate English summary to Bengali with Grok
    translation_prompt = f"""Translate the following English summary into natural and accurate Bengali. Include the sources as provided:
    {json.dumps(english_response, ensure_ascii=False)}
    Return ONLY a valid JSON object with:
    {{
        "summary": "1-2 paragraph summary in Bengali",
        "sources": "List of up to 3 URLs"
    }}"""

    async with aiohttp.ClientSession() as session:
        try:
            logger.info(f"OPENROUTER_API_KEY: {openrouter_api_key}")  # Debug log
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "HTTP-Referer": "https://kothakunjo-search-engine-backend.onrender.com",  
                    "X-Title": "Kothakunjo Search Engine",
                },
                json={
                    "model": "xai/grok",
                    "messages": [{
                        "role": "system",
                        "content": "You are a JSON-only Bengali translator. Return valid JSON with: summary, sources. No additional text."
                    }, {
                        "role": "user",
                        "content": translation_prompt
                    }],
                    "temperature": 0.1,
                    "max_tokens": 2000
                }
            ) as response:
                status = response.status
                response_text = await response.text()
                logger.info(f"OpenRouter API response: Status {status}, Body: {response_text}")  # Debug log
                if response.status != 200:
                    raise Exception(f"OpenRouter API error: {status} - {response_text}")
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise Exception("প্রক্রিয়াকরণ ত্রুটি: Grok প্রতিক্রিয়া খালি")
                try:
                    # Clean potential trailing incomplete JSON
                    content = re.sub(r',\s*]', ']', content)  # Fix unterminated arrays
                    content = re.sub(r',\s*}', '}', content)  # Fix unterminated objects
                    llm_response = json.loads(content)
                except json.JSONDecodeError as e:
                    json_match = re.search(r'\{(?:[^{}]|(?R))*\}', content, re.DOTALL)  # Robust JSON extraction
                    if json_match:
                        llm_response = json.loads(json_match.group(0))
                    else:
                        raise Exception("প্রক্রিয়াকরণ ত্রুটি: বৈধ JSON পাওয়া যায়নি")
                return llm_response
        except Exception as e:
            error_msg = str(e).replace('\n', ' ').replace('\r', ' ')[:200]
            logger.error(f"OpenRouter processing error: {error_msg}")
            return {
                "summary": "প্রক্রিয়াকরণ ত্রুটি: সার্ভারে সমস্যা হয়েছে। অনুগ্রহ করে পরে আবার চেষ্টা করুন।",
                "sources": []
            }
