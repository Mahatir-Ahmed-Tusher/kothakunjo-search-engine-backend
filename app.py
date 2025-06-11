import os
import logging
import json
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from inference import search_and_present
from starlette.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load and validate environment variables
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not MISTRAL_API_KEY or not SERPAPI_API_KEY:
    logger.error("Missing required API keys in .env file")
    raise RuntimeError("API keys not configured properly")

app = FastAPI(
    title="Kothakunjo Search Engine",
    description="A Bengali search engine using SerpAPI and Mistral for summarization and translation",
    version="1.0.0",
    contact={
        "name": "Support",
        "email": "support@kothakunjo.example.com"
    },
    license_info={
        "name": "MIT",
    },
)

# Custom JSONResponse with UTF-8 encoding
class UTF8JSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False).encode("utf-8")

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return UTF8JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)},
        headers=exc.headers
    )

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://kothakunjo.vercel.app", "https://kothakunjo.onrender.com"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
    max_age=600,
)
logger.info("CORS configured with origins: %s", ["http://localhost:5173", "https://kothakunjo.vercel.app", "https://kothakunjo.onrender.com"])

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, example="জীবনের অর্থ কী?")
    language: str = Field("bn", min_length=2, max_length=5, example="bn")

class SearchResponse(BaseModel):
    summary: str = Field(..., example="জীবনের অর্থ সম্পর্কে বিভিন্ন দর্শন রয়েছে...")
    sources: list[str] = Field(..., example=["https://example.com"])

@app.post(
    "/api/search",
    response_model=SearchResponse,
    response_class=UTF8JSONResponse,
    summary="Perform a Bengali web search",
    description="Searches the web and returns results in Bengali.",
    responses={
        200: {"description": "Successful search"},
        400: {"description": "Invalid input"},
        429: {"description": "API rate limit exceeded"},
        500: {"description": "Internal server error"}
    },
    tags=["Search"]
)
async def perform_search(request: SearchRequest):
    """
    Endpoint to perform a web search in Bengali.
    
    - **query**: The search query (2-500 characters)
    - **language**: Language code for results (default: 'bn')
    """
    try:
        logger.info(f"Search request received for: {request.query[:50]}...")
        
        result = await search_and_present(
            query=request.query,
            mistral_api_key=MISTRAL_API_KEY,
            serpapi_api_key=SERPAPI_API_KEY,
            language=request.language
        )
        
        logger.info(f"Search completed for: {request.query[:50]}...")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
        if "Mistral API error: 429" in str(e):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="API কোটা অতিক্রম করেছে। অনুগ্রহ করে পরে আবার চেষ্টা করুন।"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="সার্চ অনুরোধ প্রক্রিয়াকরণে ব্যর্থ"
        )

@app.get(
    "/health",
    summary="Service health check",
    description="Returns the current status of the API service",
    tags=["Monitoring"]
)
async def health_check():
    """
    Health check endpoint for monitoring and uptime verification.
    """
    return {
        "status": "healthy",
        "version": app.version,
        "services": {
            "mistral": bool(MISTRAL_API_KEY),
            "serpapi": bool(SERPAPI_API_KEY)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8025,
        log_level="info",
        timeout_keep_alive=30
    )
