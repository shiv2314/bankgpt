"""Enhanced Gemini integration for multi-turn conversations"""

import logging
from typing import List, Dict, Optional, Any
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

class GeminiLLM:
    
    def __init__(self):
        self.api_key = None
        # Use confirmed available models from latest Gemini lineup
        # Using stable releases and latest aliases for better compatibility
        self.model_names = [
            "gemini-2.5-flash",        # Latest stable Flash (Recommended)
            "gemini-2.5-pro",          # Latest stable Pro (More capable)
            "gemini-2.0-flash",        # Fallback to stable 2.0 Flash
            "gemini-pro-latest",       # Fallback to latest Pro alias
        ]
        self.current_model_name = self.model_names[0]
        self.temperature = 0.7
        self.max_tokens = 1024
        self.configure()
    
    def configure(self):
        """Configure Gemini API"""
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.api_key = api_key
            logger.info("Gemini API configured")
        else:
            logger.warning("GEMINI_API_KEY not found in environment")
    
    def get_response(self, message: str, history: Optional[List[Dict]] = None, 
                     system_prompt: Optional[str] = None) -> str:
        """Get response from Gemini with conversation history"""
        
        # Try models in order until one works
        for model_name in self.model_names:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    }
                )
                
                # Build conversation messages
                messages = []
                
                # Add system prompt as initial context
                if system_prompt:
                    messages.append({
                        "role": "user",
                        "parts": [system_prompt]
                    })
                    messages.append({
                        "role": "model",
                        "parts": ["Understood. I'll follow these guidelines."]
                    })
                
                # Add conversation history
                if history:
                    for msg in history:
                        role = "user" if msg.get("role") == "user" else "model"
                        content = msg.get("content", "")
                        if content: # Only add non-empty messages
                            messages.append({
                                "role": role,
                                "parts": [content]
                            })
                
                # Add current message
                messages.append({
                    "role": "user",
                    "parts": [message]
                })
                
                # Get response
                response = model.generate_content(messages)
                
                if response and response.text:
                    self.current_model_name = model_name # Remember the working model
                    return response.text
                else:
                    logger.warning(f"Empty response from Gemini model {model_name}")
                    continue
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Gemini error with model {model_name}: {error_msg}")
                
                # If it's the last model in the list, raise the exception to trigger Groq fallback
                if model_name == self.model_names[-1]:
                    logger.error("All Gemini models failed. Switching to fallback.")
                    raise e
                
                # Continue to next model on 404 (Not Found) or 503 (Overloaded)
                if "404" in error_msg or "503" in error_msg:
                    continue
                # For other errors (like auth), failing fast might be better, but we continue here to be safe
                continue
        
        return "I couldn't generate a response. Please try again."

    def generate_sales_pitch(self, customer_data: Dict) -> str:
        """Generate personalized sales pitch"""
        prompt = f"""
        Generate a personalized loan offer pitch for this customer:
        - Name: {customer_data.get('customer_name', 'Customer')}
        - Requested Amount: {customer_data.get('requested_amount', 'Unknown')} rupees
        - Pre-approved Limit: {customer_data.get('pre_approved_limit', 'Unknown')} rupees
        - Credit Score: {customer_data.get('credit_score', 'Unknown')}
        
        Make it professional, friendly, and compelling. Keep it to 2-3 sentences.
        """
        try:
            # Use the last known working model
            model = genai.GenerativeModel(model_name=self.current_model_name)
            response = model.generate_content(prompt)
            return response.text if response else "Special loan offer available for you!"
        except Exception as e:
            logger.error(f"Sales pitch generation error: {str(e)}")
            return "We have special loan offers available for you!"

    def extract_information(self, message: str) -> Dict[str, Any]:
        """Extract customer information from message"""
        prompt = f"""
        Extract customer information from this message in JSON format:
        Message: {message}
        
        Look for:
        - customer_name (string)
        - phone (string, 10 digits)
        - requested_amount (number)
        - credit_score (number 300-900)
        - income (number)
        
        Return only JSON, no other text.
        """
        try:
            # Use the last known working model
            model = genai.GenerativeModel(model_name=self.current_model_name)
            response = model.generate_content(prompt)
            
            import json
            import re
            if response and response.text:
                try:
                    # Clean up json markdown if present
                    text = response.text
                    if "```json" in text:
                        text = re.search(r'```json\n(.*?)\n```', text, re.DOTALL).group(1)
                    elif "```" in text:
                        text = re.search(r'```\n(.*?)\n```', text, re.DOTALL).group(1)
                    return json.loads(text)
                except:
                    return {}
            return {}
        except Exception as e:
            logger.error(f"Information extraction error: {str(e)}")
            return {}

    def is_available(self) -> bool:
        """Check if API is available"""
        return self.api_key is not None

# Backward compatibility alias
class GeminiClient(GeminiLLM):
    """Backward compatible alias for GeminiLLM"""
    pass