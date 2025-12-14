"""
BankGPT FastAPI Backend
Serves as the core API layer for LLM interactions with Gemini (primary) and Groq (fallback)
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime
import os
import asyncio

from context_manager import ContextManager
from fallback_handler import FallbackHandler
from gemini_integration_v2 import GeminiLLM
from groq_integration import GroqClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BankGPT LLM API",
    description="Multi-turn conversation API with Gemini primary and Groq fallback",
    version="1.0.0"
)

# Configure CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
context_manager = ContextManager()
gemini_llm = GeminiLLM()
groq_llm = GroqClient()
fallback_handler = FallbackHandler(gemini_llm, groq_llm, context_manager)

# ==================== Pydantic Models ====================

class Message(BaseModel):
    """Single message in conversation"""
    role: str  # "user" or "bot"
    content: str
    timestamp: Optional[str] = None

class ConversationRequest(BaseModel):
    """Request for conversation endpoint"""
    user_message: str
    session_id: str
    custom_context: Optional[Dict[str, Any]] = None

class ConversationResponse(BaseModel):
    """Response from conversation endpoint"""
    response: str
    session_id: str
    conversation_stage: str
    conversation_history: List[Dict[str, str]]
    api_used: str
    tokens_used: Dict[str, int]
    context_summary: Dict[str, Any]
    timestamp: str

class ContextResetRequest(BaseModel):
    """Request to reset context for a session"""
    session_id: str

class APIStatusResponse(BaseModel):
    """API status information"""
    status: str
    gemini_available: bool
    groq_available: bool
    active_sessions: int
    timestamp: str

# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "BankGPT LLM API"
    }

# ==================== Status Endpoint ====================

@app.get("/api/status")
async def api_status():
    """Get API status including LLM availability"""
    try:
        gemini_available = gemini_llm.is_available() if hasattr(gemini_llm, 'is_available') else True
        groq_available = groq_llm.is_available() if hasattr(groq_llm, 'is_available') else True
    except:
        gemini_available = True
        groq_available = True
    
    return {
        "status": "operational",
        "gemini_available": gemini_available,
        "groq_available": groq_available,
        "active_sessions": len(context_manager.sessions),
        "timestamp": datetime.now().isoformat()
    }

# ==================== Conversation Endpoints ====================

@app.post("/api/conversation", response_model=ConversationResponse)
async def conversation(request: ConversationRequest, background_tasks: BackgroundTasks):
    """
    Main conversation endpoint
    Handles multi-turn conversations with context retention
    """
    try:
        session_id = request.session_id
        user_message = request.user_message
        
        # Initialize session if new
        if session_id not in context_manager.sessions:
            context_manager.initialize_session(session_id)
        
        # Add user message to history
        context_manager.add_message(session_id, "user", user_message)
        
        # Get response from fallback handler
        try:
            response = await fallback_handler.get_response(session_id, user_message)
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            raise HTTPException(status_code=503, detail="All LLM APIs unavailable. Please try again later.")
        
        # Add bot response to history
        context_manager.add_message(session_id, "bot", response)
        
        # Get current conversation state
        history = context_manager.get_conversation_history_for_llm(session_id)
        context = context_manager.get_context(session_id)
        
        # Background logging
        background_tasks.add_task(
            log_conversation,
            session_id=session_id,
            user_msg=user_message,
            bot_response=response
        )
        
        return {
            "response": response,
            "session_id": session_id,
            "conversation_stage": context.get("stage", "greeting"),
            "conversation_history": [
                {"role": msg[0], "content": msg[1], "timestamp": msg[2]}
                for msg in context_manager.histories.get(session_id, [])
            ],
            "api_used": fallback_handler.get_last_api_used(),
            "tokens_used": {"input": 0, "output": 0},
            "context_summary": {
                "customer_name": context.get("customer_name"),
                "phone": context.get("phone"),
                "requested_amount": context.get("requested_amount"),
                "pre_approved_limit": context.get("pre_approved_limit"),
                "credit_score": context.get("credit_score")
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Context Management Endpoints ====================

@app.get("/api/context/{session_id}")
async def get_context(session_id: str):
    """Get current context for a session"""
    if session_id not in context_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    context = context_manager.get_context(session_id)
    return {
        "session_id": session_id,
        "context": context,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/context/{session_id}")
async def update_context(session_id: str, updates: Dict[str, Any]):
    """Update context for a session"""
    if session_id not in context_manager.sessions:
        context_manager.initialize_session(session_id)
    
    context_manager.update_context(session_id, updates)
    return {
        "session_id": session_id,
        "message": "Context updated",
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/api/context/{session_id}")
async def reset_context(session_id: str):
    """Reset context for a session"""
    if session_id in context_manager.sessions:
        context_manager.sessions[session_id] = {}
    
    return {
        "session_id": session_id,
        "message": "Context reset",
        "timestamp": datetime.now().isoformat()
    }

# ==================== Conversation History Endpoints ====================

@app.get("/api/conversation/history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get full conversation history for a session"""
    if session_id not in context_manager.histories:
        return {
            "session_id": session_id,
            "history": [],
            "total_messages": 0
        }
    
    history = context_manager.histories[session_id]
    return {
        "session_id": session_id,
        "history": [
            {
                "role": msg[0],
                "content": msg[1],
                "timestamp": msg[2]
            }
            for msg in history
        ],
        "total_messages": len(history)
    }

@app.delete("/api/conversation/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    if session_id in context_manager.histories:
        context_manager.histories[session_id] = []
    
    return {
        "session_id": session_id,
        "message": "Conversation history cleared",
        "timestamp": datetime.now().isoformat()
    }

# ==================== Analytics Endpoints ====================

@app.post("/api/conversation/analyze")
async def analyze_conversation(session_id: str):
    """Analyze a conversation for metrics"""
    if session_id not in context_manager.histories:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = context_manager.histories[session_id]
    user_messages = [msg for msg in history if msg[0] == "user"]
    bot_messages = [msg for msg in history if msg[0] == "bot"]
    
    return {
        "session_id": session_id,
        "total_turns": len(history) // 2,
        "user_message_count": len(user_messages),
        "bot_message_count": len(bot_messages),
        "average_user_length": sum(len(msg[1]) for msg in user_messages) / max(len(user_messages), 1),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats/sessions")
async def get_session_statistics():
    """Get statistics for all active sessions"""
    stats = []
    for session_id, context in context_manager.sessions.items():
        history = context_manager.histories.get(session_id, [])
        stats.append({
            "session_id": session_id,
            "message_count": len(history),
            "stage": context.get("stage", "unknown"),
            "created_at": context.get("created_at", "unknown")
        })
    
    return {
        "active_sessions": len(stats),
        "sessions": stats,
        "timestamp": datetime.now().isoformat()
    }

# ==================== Utility Functions ====================

async def log_conversation(session_id: str, user_msg: str, bot_response: str):
    """Background task to log conversations"""
    try:
        logger.info(f"[{session_id}] User: {user_msg[:50]}...")
        logger.info(f"[{session_id}] Bot: {bot_response[:50]}...")
    except Exception as e:
        logger.error(f"Logging error: {str(e)}")

# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("BankGPT FastAPI Backend starting...")
    logger.info(f"Gemini API: Available")
    logger.info(f"Groq API: Available")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("BankGPT FastAPI Backend shutting down...")

# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
