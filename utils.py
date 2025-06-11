from typing import Dict, List

def format_serp_results(results: List[Dict]) -> str:
    """Convert SerpAPI results to a readable string for the LLM."""
    formatted = []
    for idx, result in enumerate(results[:3], 1):  # Top 3 results
        formatted.append(
            f"উৎস {idx}:\n"
            f"- শিরোনাম: {result.get('title', 'N/A')}\n"
            f"- URL: {result.get('link', 'N/A')}\n"
            f"- সারাংশ: {result.get('snippet', 'কোনো সারাংশ নেই')}\n"
        )
    return "\n".join(formatted) if formatted else "কোনো প্রাসঙ্গিক উৎস পাওয়া যায়নি."