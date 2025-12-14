"""Fallback handler managing Gemini → Groq API switching"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class FallbackHandler:
    """Manages fallback from Gemini to Groq API"""
    
    def __init__(self, gemini_llm, groq_llm, context_manager, rate_limit_cooldown: int = 60):
        self.gemini = gemini_llm
        self.groq = groq_llm
        self.context_manager = context_manager
        self.rate_limit_cooldown = rate_limit_cooldown  # seconds
        
        # Fallback state
        self.gemini_rate_limited = False
        self.gemini_rate_limit_until = None
        self.groq_rate_limited = False
        self.groq_rate_limit_until = None
        
        # Error tracking for circuit breaker
        self.error_counts = {"gemini": 0, "groq": 0}
        self.circuit_breaker_threshold = 5
        self.circuits_open = {"gemini": False, "groq": False}
        
        # Call history for fallback stats
        self.api_call_history = []
        self.last_api_used = None
    
    async def get_response(self, session_id: str, user_message: str) -> str:
        """Get response using fallback strategy"""
        context = self.context_manager.get_context(session_id)
        history = self.context_manager.get_conversation_history_for_llm(session_id)
        
        # Try Gemini first (primary)
        if not self.circuits_open["gemini"] and not self._is_rate_limited("gemini"):
            try:
                response = await self._call_gemini(user_message, history, context)
                self.last_api_used = "gemini"
                self.error_counts["gemini"] = 0
                self._record_api_call("gemini", True)
                return response
            except Exception as e:
                logger.warning(f"Gemini error: {str(e)}")
                
                # Check if rate limit error
                if "429" in str(e) or "rate" in str(e).lower():
                    self._set_rate_limit("gemini")
                    logger.warning("Gemini rate limited, switching to Groq")
                
                self.error_counts["gemini"] += 1
                if self.error_counts["gemini"] >= self.circuit_breaker_threshold:
                    self.circuits_open["gemini"] = True
                    logger.error(f"Gemini circuit breaker opened after {self.circuit_breaker_threshold} errors")
        
        # Try Groq (fallback)
        if not self.circuits_open["groq"] and not self._is_rate_limited("groq"):
            try:
                response = await self._call_groq(user_message, history, context)
                self.last_api_used = "groq"
                self.error_counts["groq"] = 0
                self._record_api_call("groq", True)
                return response
            except Exception as e:
                logger.error(f"Groq error: {str(e)}")
                
                if "429" in str(e) or "rate" in str(e).lower():
                    self._set_rate_limit("groq")
                
                self.error_counts["groq"] += 1
                if self.error_counts["groq"] >= self.circuit_breaker_threshold:
                    self.circuits_open["groq"] = True
                    logger.error(f"Groq circuit breaker opened after {self.circuit_breaker_threshold} errors")
        
        # Both APIs unavailable
        logger.error("All LLM APIs unavailable")
        raise Exception("All LLM APIs unavailable. Please try again later.")
    
    async def _call_gemini(self, message: str, history: List[Dict], context: Dict) -> str:
        """Call Gemini API"""
        system_prompt = self._build_system_prompt(context.get("stage", "greeting"))
        
        try:
            response = await asyncio.to_thread(
                self.gemini.get_response,
                message,
                history,
                system_prompt
            )
            return response
        except Exception as e:
            if "429" in str(e):
                raise Exception("Rate limit error")
            raise
    
    async def _call_groq(self, message: str, history: List[Dict], context: Dict) -> str:
        """Call Groq API"""
        system_prompt = self._build_system_prompt(context.get("stage", "greeting"))
        
        try:
            response = await asyncio.to_thread(
                self.groq.get_response,
                message,
                history,
                system_prompt
            )
            return response
        except Exception as e:
            if "429" in str(e):
                raise Exception("Rate limit error")
            raise
    
    def _build_system_prompt(self, stage: str) -> str:
        """Build stage-specific system prompt"""
        prompts = {
            "greeting": "You are a friendly BankGPT loan officer. Greet the customer and ask how you can help with their loan needs.",
            "amount": "Ask the customer about the loan amount they need. Be professional and helpful.",
            "phone": "Ask the customer for their phone number to proceed with the application.",
            "income": "Ask about the customer's monthly income to assess loan eligibility.",
            "credit_score": "Ask the customer about their credit score to determine pre-approved limits.",
            "eligibility": "Evaluate the customer's eligibility based on their information.",
            "approval": "Inform the customer about their pre-approved loan amount and next steps.",
            "documents": "Request necessary documents for final verification.",
            "completed": "Confirm the loan application is complete and provide next steps."
        }
        return prompts.get(stage, prompts["greeting"])
    
    def _is_rate_limited(self, api: str) -> bool:
        """Check if API is rate limited"""
        if api == "gemini":
            if self.gemini_rate_limited and self.gemini_rate_limit_until:
                if datetime.now() < self.gemini_rate_limit_until:
                    return True
                else:
                    self.gemini_rate_limited = False
                    self.gemini_rate_limit_until = None
        elif api == "groq":
            if self.groq_rate_limited and self.groq_rate_limit_until:
                if datetime.now() < self.groq_rate_limit_until:
                    return True
                else:
                    self.groq_rate_limited = False
                    self.groq_rate_limit_until = None
        
        return False
    
    def _set_rate_limit(self, api: str) -> None:
        """Set rate limit for API"""
        cooldown_until = datetime.now() + timedelta(seconds=self.rate_limit_cooldown)
        
        if api == "gemini":
            self.gemini_rate_limited = True
            self.gemini_rate_limit_until = cooldown_until
        elif api == "groq":
            self.groq_rate_limited = True
            self.groq_rate_limit_until = cooldown_until
    
    def _record_api_call(self, api: str, success: bool) -> None:
        """Record API call for statistics"""
        self.api_call_history.append({
            "api": api,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 calls
        if len(self.api_call_history) > 100:
            self.api_call_history = self.api_call_history[-100:]
    
    def get_last_api_used(self) -> str:
        """Get the last API used"""
        return self.last_api_used or "gemini"
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        return {
            "gemini_rate_limited": self.gemini_rate_limited,
            "groq_rate_limited": self.groq_rate_limited,
            "gemini_circuit_open": self.circuits_open["gemini"],
            "groq_circuit_open": self.circuits_open["groq"],
            "error_counts": self.error_counts,
            "api_calls_recorded": len(self.api_call_history)
        }
