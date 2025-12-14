"""FastAPI client for Streamlit frontend communication"""

import asyncio
import logging
from typing import Optional, Dict, List, Any
import httpx

logger = logging.getLogger(__name__)

class FastAPIClient:
    """HTTP client for FastAPI backend communication"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.client = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client"""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout)
        return self.client
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
    
    async def send_message(self, message: str, session_id: str) -> Optional[Dict]:
        """Send message to backend and get response"""
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/api/conversation",
                json={
                    "user_message": message,
                    "session_id": session_id
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Message send error: {str(e)}")
            return None
    
    async def get_context(self, session_id: str) -> Optional[Dict]:
        """Get context for a session"""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/api/context/{session_id}"
            )
            
            if response.status_code == 200:
                return response.json().get("context")
            return None
            
        except Exception as e:
            logger.error(f"Get context error: {str(e)}")
            return None
    
    async def update_context(self, session_id: str, updates: Dict) -> bool:
        """Update context for a session"""
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/api/context/{session_id}",
                json=updates
            )
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Update context error: {str(e)}")
            return False
    
    async def reset_context(self, session_id: str) -> bool:
        """Reset context for a session"""
        try:
            client = await self._get_client()
            response = await client.delete(
                f"{self.base_url}/api/context/{session_id}"
            )
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Reset context error: {str(e)}")
            return False
    
    async def get_conversation_history(self, session_id: str) -> Optional[List]:
        """Get conversation history"""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/api/conversation/history/{session_id}"
            )
            
            if response.status_code == 200:
                return response.json().get("history", [])
            return None
            
        except Exception as e:
            logger.error(f"Get history error: {str(e)}")
            return None
    
    async def clear_conversation_history(self, session_id: str) -> bool:
        """Clear conversation history"""
        try:
            client = await self._get_client()
            response = await client.delete(
                f"{self.base_url}/api/conversation/history/{session_id}"
            )
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Clear history error: {str(e)}")
            return False
    
    async def get_api_status(self) -> Optional[Dict]:
        """Get API status"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/status")
            
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            logger.error(f"Get status error: {str(e)}")
            return None
    
    async def analyze_conversation(self, session_id: str) -> Optional[Dict]:
        """Analyze conversation"""
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/api/conversation/analyze",
                params={"session_id": session_id}
            )
            
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            logger.error(f"Analyze conversation error: {str(e)}")
            return None
    
    async def get_session_statistics(self) -> Optional[Dict]:
        """Get session statistics"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/stats/sessions")
            
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            logger.error(f"Get statistics error: {str(e)}")
            return None
    
    async def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return False
