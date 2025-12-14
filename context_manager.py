"""Context Manager for session state and conversation history"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages session context, conversation history, and customer data"""
    
    def __init__(self, max_messages_per_session: int = 100):
        self.sessions = {}  # {session_id: {context_data}}
        self.histories = {}  # {session_id: [(role, message, timestamp), ...]}
        self.metadata = {}  # {session_id: {created_at, updated_at, duration}}
        self.max_messages = max_messages_per_session
    
    def initialize_session(self, session_id: str) -> Dict[str, Any]:
        """Initialize a new session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "stage": "greeting",
                "customer_name": None,
                "phone": None,
                "requested_amount": None,
                "pre_approved_limit": None,
                "credit_score": None,
                "income": None,
                "employment": None,
                "created_at": datetime.now().isoformat()
            }
            self.histories[session_id] = []
            self.metadata[session_id] = {
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "message_count": 0
            }
            logger.info(f"Session initialized: {session_id}")
        
        return self.sessions[session_id]
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get current context for a session"""
        if session_id not in self.sessions:
            return self.initialize_session(session_id)
        return self.sessions[session_id]
    
    def update_context(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update context for a session"""
        if session_id not in self.sessions:
            self.initialize_session(session_id)
        
        self.sessions[session_id].update(updates)
        self.metadata[session_id]["updated_at"] = datetime.now()
        logger.info(f"Context updated for {session_id}: {updates}")
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add message to conversation history"""
        if session_id not in self.histories:
            self.histories[session_id] = []
        
        timestamp = datetime.now().isoformat()
        self.histories[session_id].append((role, content, timestamp))
        
        # Trim if exceeds max messages
        if len(self.histories[session_id]) > self.max_messages:
            self.histories[session_id] = self.histories[session_id][-self.max_messages:]
        
        self.metadata[session_id]["message_count"] = len(self.histories[session_id])
        self.metadata[session_id]["updated_at"] = datetime.now()
    
    def get_conversation_history_for_llm(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history formatted for LLM"""
        if session_id not in self.histories:
            return []
        
        return [
            {"role": msg[0], "content": msg[1]}
            for msg in self.histories[session_id]
        ]
    
    def clear_session(self, session_id: str) -> None:
        """Clear all data for a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.histories:
            del self.histories[session_id]
        if session_id in self.metadata:
            del self.metadata[session_id]
        logger.info(f"Session cleared: {session_id}")
    
    def get_session_duration(self, session_id: str) -> float:
        """Get duration of session in seconds"""
        if session_id not in self.metadata:
            return 0
        
        created = self.metadata[session_id]["created_at"]
        updated = self.metadata[session_id]["updated_at"]
        return (updated - created).total_seconds()
    
    def export_session(self, session_id: str) -> Dict[str, Any]:
        """Export complete session data"""
        return {
            "session_id": session_id,
            "context": self.sessions.get(session_id, {}),
            "history": self.histories.get(session_id, []),
            "metadata": {
                "created_at": str(self.metadata[session_id]["created_at"]),
                "updated_at": str(self.metadata[session_id]["updated_at"]),
                "duration_seconds": self.get_session_duration(session_id)
            } if session_id in self.metadata else {}
        }
    
    def import_session(self, session_data: Dict[str, Any]) -> None:
        """Import session data"""
        session_id = session_data["session_id"]
        self.sessions[session_id] = session_data.get("context", {})
        self.histories[session_id] = session_data.get("history", [])
        logger.info(f"Session imported: {session_id}")
