from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys
import os

# Add your chatbot to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

from chatbot.groq_rag_chatbot import GroqRAGChatbot

app = FastAPI(title="Wei Ming Chatbot API", version="1.0.0")

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        "https://vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    max_tokens: int = 500

class ChatResponse(BaseModel):
    response: str
    success: bool
    model_used: str

# Initialize chatbot
vector_store_dir = project_root / "data" / "vector_store"
chatbot = GroqRAGChatbot(str(vector_store_dir))

@app.get("/")
async def root():
    return {"message": "Wei Ming Chatbot API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is operational"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Get response from your chatbot
        result = chatbot.chat(
            query=request.query,
            top_k=request.top_k,
            max_tokens=request.max_tokens
        )

        return ChatResponse(
            response=result['response'],
            success=True,
            model_used=result['model_used']
        )

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
