import aiohttp
import json
import serpapi
import re
from typing import Dict, List
from utils import format_serp_results
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def search_and_present(
    query: str,
    mistral_api_key: str,
    serpapi_api_key: str,
    language: str = "bn"
) -> Dict[str, str | List[str]]:
    """
    Search the web using SerpAPI and generate a Bengali summary with Mistral.
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
        logger.error(f"SerpAPI error: {str(e)}")
        return {
            "summary": "সার্চ ত্রুটি: তথ্য অনুসন্ধানে সমস্যা হয়েছে।",
            "sources": []
        }

    # Step 2: Format results for LLM
    formatted_results = format_serp_results(search_results[:3])

    # Step 3: Generate English summary and translate to Bengali with Mistral
    specialist_prompt = f"""You are a search result summarizer and translator. Analyze the following search results and provide a concise 1-2 paragraph summary in English, including key points and citations. Then, translate the summary into natural and accurate Bengali. Return ONLY a valid JSON object with:
    {{
        "summary": "1-2 paragraph summary in Bengali",
        "sources": "List of up to 3 URLs"
    }}
    Search results:
    {formatted_results}"""

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {mistral_api_key}",
                },
                json={
                    "model": "mistral-small-latest",
                    "messages": [{"role": "user", "content": specialist_prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            ) as response:
                status = response.status
                response_text = await response.text()
                logger.info(f"Mistral API request: {specialist_prompt[:500]}...")
                logger.info(f"Mistral API response: Status {status}, Body: {response_text}")
                if response.status == 429:
                    logger.error("Mistral API rate limit exceeded")
                    return {
                        "summary": "প্রক্রিয়াকরণ ত্রুটি: API কোটা অতিক্রম করেছে। অনুগ্রহ করে পরে আবার চেষ্টা করুন।",
                        "sources": []
                    }
                if response.status != 200:
                    raise Exception(f"Mistral API error: {status} - {response_text}")
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise Exception("প্রক্রিয়াকরণ ত্রুটি: Mistral প্রতিক্রিয়া খালি")
                try:
                    content = re.sub(r',\s*]', ']', content)
                    content = re.sub(r',\s*}', '}', content)
                    llm_response = json.loads(content)
                except json.JSONDecodeError as e:
                    json_match = re.search(r'\{(?:[^{}]|(?R))*\}', content, re.DOTALL)
                    if json_match:
                        llm_response = json.loads(json_match.group(0))
                    else:
                        raise Exception("প্রক্রিয়াকরণ ত্রুটি: বৈধ JSON পাওয়া যায়নি")
                return llm_response
        except aiohttp.ClientConnectorError as e:
            error_msg = f"Mistral connection error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "summary": "সংযোগ ত্রুটি: Mistral API-এর সাথে সংযোগ করা যায়নি। অনুগ্রহ করে পরে চেষ্টা করুন।",
                "sources": []
            }
        except Exception as e:
            error_msg = str(e).replace('\n', ' ').replace('\r', ' ')[:200]
            logger.error(f"Mistral processing error: {error_msg}\n{traceback.format_exc()}")
            return {
                "summary": "প্রক্রিয়াকরণ ত্রুটি: সার্ভারে সমস্যা হয়েছে। অনুগ্রহ করে পরে আবার চেষ্টা করুন।",
                "sources": []
            }
