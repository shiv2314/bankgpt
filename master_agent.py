# master_agent.py - Unified LLM-Driven Conversation Agent
"""
Single, continuous conversation flow powered by LLM.
No phases - one natural conversation that handles:
- Greeting and loan understanding
- Phone verification and profile lookup
- Loan amount and eligibility
- Document verification if needed
- Approval and sanction

Language-aware responses (English, Hindi, Hinglish)
Smart context tracking to avoid duplicate questions
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from agents import load_db, verification_agent, fraud_agent, underwriting_agent, sanction_agent
from language_helper import detect_language
from groq_integration import GroqClient

DATA_PATH = Path('data/mock_db.json')


def load_mock_db():
    """Load mock CRM database."""
    if DATA_PATH.exists():
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def run_unified_agent(user_input: str, state: Dict[str, Any], conversation_history: list) -> Dict[str, Any]:
    """
    Unified conversation agent - handles entire loan application flow naturally.
    
    Key improvements:
    - Flexible conversation flow that adapts to user input
    - Handles unexpected/off-topic inputs gracefully
    - Extracts information opportunistically (doesn't rigidly ask in sequence)
    - Uses context to avoid asking for already-provided information
    - Natural progression without forcing stages
    
    Args:
        user_input: User's message
        state: Conversation state tracking
        conversation_history: List of previous messages for context
        
    Returns:
        Dict with message and updated state info
    """
    
    detected_language = detect_language(user_input)
    
    # FIRST TURN: Show greeting
    if not conversation_history:
        greeting = _get_adaptive_greeting(detected_language)
        return {
            'message': greeting,
            'detected_language': detected_language,
            'conversation_stage': 'loan_understanding'
        }
    
    # SUBSEQUENT TURNS: Use LLM to drive conversation flexibly
    try:
        client = GroqClient()
        
        # Extract ALL possible information from user input (not stage-dependent)
        extracted_info = _extract_all_information(user_input, state, detected_language)
        
        # Build conversation context
        history_context = _build_history_context(conversation_history, detected_language)
        
        # Build state context with what we know
        state_context = _build_flexible_state_context(state, extracted_info, detected_language)
        
        # Determine what information we still need
        missing_info = _determine_missing_info(state, extracted_info)
        
        # Build adaptive system prompt based on what we have and need
        system_prompt = _get_adaptive_system_prompt(
            detected_language, 
            state, 
            extracted_info, 
            missing_info
        )
        
        # Build the full prompt
        full_prompt = f"""{system_prompt}

CONVERSATION HISTORY:
{history_context}

WHAT WE KNOW:
{state_context}

WHAT WE STILL NEED:
{missing_info if missing_info else '- All essential information has been collected'}

Customer's latest message: "{user_input}"

Your response should:
1. Acknowledge the customer's input naturally
2. If they mentioned their loan need, ask clarifying questions
3. Extract information naturally without rigid questioning
4. If they go off-topic, politely redirect to loan assistance
5. Keep responses under 150 words
6. Be warm, professional, and conversational

Generate only the bot's response, no explanations."""
        
        # Generate response with Groq LLM
        response = GroqClient.generate_text(full_prompt, max_tokens=300)
        
        if not response or len(response.strip()) < 5:
            response = _get_fallback_response(missing_info, detected_language)
        
        # Determine conversation stage adaptively
        next_stage = _determine_adaptive_stage(state, extracted_info, missing_info)
        
        return {
            'message': response.strip(),
            'detected_language': detected_language,
            'conversation_stage': next_stage,
            **extracted_info  # Include phone, amount, name, etc. if extracted
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Fallback response
        return {
            'message': "I'm having trouble processing your request. Could you please try again?",
            'detected_language': detected_language,
            'error': str(e),
            'conversation_stage': state.get('conversation_stage', 'loan_understanding')
        }


def _get_greeting(language: str) -> str:
    """Get language-specific greeting."""
    greetings = {
        'hindi': (
            "🙏 नमस्ते! मैं BankGPT हूँ टाटा कैपिटल से। "
            "आप अपने लिए कौन सा लोन ढूंढ रहे हैं?"
        ),
        'hinglish': (
            "🙏 Namaste! Main BankGPT hoon Tata Capital se. "
            "Aap apne liye kaunsa loan dhundh rahe ho?"
        ),
        'english': (
            "🙏 Namaste! I am BankGPT from Tata Capital. "
            "What kind of loan are you looking for?"
        )
    }
    return greetings.get(language, greetings['english'])


def _determine_conversation_stage(state: Dict[str, Any], conversation_history: list) -> str:
    """Determine current conversation stage adaptively."""
    
    if not conversation_history:
        return 'greeting'
    
    has_phone = bool(state.get('phone'))
    has_amount = bool(state.get('requested_amount'))
    has_name = bool(state.get('customer_name'))
    
    # Flexible stage determination
    if has_amount and has_phone:
        return 'eligibility_check'
    elif has_phone:
        return 'amount_gathering'
    elif has_name or len(conversation_history) > 2:
        return 'phone_gathering'
    else:
        return 'loan_understanding'


def _determine_adaptive_stage(state: Dict[str, Any], extracted_info: Dict[str, Any], missing_info: str) -> str:
    """Determine stage adaptively based on what information we have."""
    
    has_phone = bool(state.get('phone') or extracted_info.get('phone'))
    has_amount = bool(state.get('requested_amount') or extracted_info.get('requested_amount'))
    has_name = bool(state.get('customer_name') or extracted_info.get('customer_name'))
    
    # Stage progression based on actual data, not rigid sequence
    if has_amount and has_phone:
        return 'eligibility_check'
    elif has_phone and not has_amount:
        return 'amount_gathering'
    elif has_name and not has_phone:
        return 'phone_gathering'
    else:
        return 'loan_understanding'


def _determine_next_stage(current_stage: str, extracted_info: Dict[str, Any], state: Dict[str, Any]) -> str:
    """Determine next conversation stage."""
    
    has_new_phone = bool(extracted_info.get('phone'))
    has_new_amount = bool(extracted_info.get('requested_amount'))
    
    stage_flow = {
        'greeting': 'phone_asked',
        'phone_asked': 'phone_provided' if has_new_phone else 'phone_asked',
        'phone_provided': 'amount_asked' if state.get('verified') else 'phone_provided',
        'amount_asked': 'amount_provided' if has_new_amount else 'amount_asked',
        'amount_provided': 'eligibility_check',
        'eligibility_check': 'approved' if (state.get('requested_amount', 0) <= state.get('pre_approved_limit', 0)) else 'document_needed',
        'approved': 'completed',
        'document_needed': 'document_uploaded',
        'document_uploaded': 'completed',
        'completed': 'completed',
        'loan_type': 'phone_asked'
    }
    
    return stage_flow.get(current_stage, current_stage)


def _get_stage_instructions(stage: str) -> str:
    """Get specific instructions for each conversation stage."""
    
    instructions = {
        'greeting': "You've already greeted the customer. Now ask about their loan needs and what type of loan they want.",
        'phone_asked': "The customer has NOT given their phone number yet. Ask for their 10-digit phone number to verify their identity.",
        'phone_provided': "The customer just provided their phone number. Acknowledge it and tell them you're verifying their profile.",
        'amount_asked': "The customer has provided their phone and been verified. Now ask how much loan amount they need.",
        'amount_provided': "The customer just told you the amount they need. Acknowledge it and say you're checking their eligibility.",
        'eligibility_check': "Check if the amount is within their pre-approved limit. Tell them the result.",
        'approved': "The amount is within their limit. Congratulate them on approval and provide EMI details.",
        'document_needed': "The amount exceeds their pre-approved limit. Ask them to upload their salary slip for verification.",
        'document_uploaded': "They've uploaded the document. Verify it and give them the final decision.",
        'completed': "The loan has been approved or the application is complete. Offer next steps or ask if they need anything else."
    }
    
    return instructions.get(stage, "Continue the conversation naturally based on the context provided.")


def _build_state_context_with_stage(state: Dict[str, Any], stage: str, language: str) -> str:
    """Build current state summary with stage awareness."""
    lines = []
    
    lines.append(f"CONVERSATION STAGE: {stage}")
    lines.append("")
    
    # What we know
    lines.append("VERIFIED INFORMATION:")
    if state.get('phone'):
        lines.append(f"  ✓ Phone: {state['phone']} (verified)")
    else:
        lines.append(f"  ✗ Phone: NOT YET PROVIDED")
    
    if state.get('customer_name'):
        lines.append(f"  ✓ Name: {state['customer_name']}")
    
    if state.get('credit_score'):
        lines.append(f"  ✓ Credit Score: {state['credit_score']}")
    
    if state.get('pre_approved_limit'):
        lines.append(f"  ✓ Pre-approved Limit: ₹{state['pre_approved_limit']:,}")
    
    if state.get('requested_amount'):
        lines.append(f"  ✓ Requested Amount: ₹{state['requested_amount']:,}")
    else:
        lines.append(f"  ✗ Requested Amount: NOT YET PROVIDED")
    
    if state.get('verified'):
        lines.append(f"  ✓ Identity: VERIFIED")
    else:
        lines.append(f"  ✗ Identity: NOT YET VERIFIED")
    
    # Eligibility
    if state.get('phone') and state.get('requested_amount'):
        amount = state.get('requested_amount', 0)
        limit = state.get('pre_approved_limit', 0)
        
        lines.append("")
        lines.append("ELIGIBILITY STATUS:")
        if amount <= limit:
            lines.append(f"  ✓ APPROVED - Amount ₹{amount:,} is within limit ₹{limit:,}")
        else:
            lines.append(f"  ⚠ NEEDS REVIEW - Amount ₹{amount:,} exceeds limit ₹{limit:,}")
    
    return "\n".join(lines)



def _build_history_context(conversation_history: list, language: str) -> str:
    """Build conversation history for LLM context."""
    if not conversation_history:
        return "No previous conversation yet."
    
    lines = []
    for msg in conversation_history[-6:]:  # Last 6 messages for context
        role = "Customer" if msg['role'] == 'user' else "BankGPT"
        lines.append(f"{role}: {msg['content'][:100]}")
    
    return "\n".join(lines)


def _build_state_context(state: Dict[str, Any], language: str) -> str:
    """Build current state summary for LLM context."""
    lines = []
    
    if state.get('phone'):
        lines.append(f"- Phone verified: {state['phone']}")
    
    if state.get('customer_name'):
        lines.append(f"- Customer name: {state['customer_name']}")
    
    if state.get('credit_score'):
        lines.append(f"- Credit score: {state['credit_score']}")
    
    if state.get('pre_approved_limit'):
        lines.append(f"- Pre-approved limit: ₹{state['pre_approved_limit']:,}")
    
    if state.get('requested_amount'):
        lines.append(f"- Requested amount: ₹{state['requested_amount']:,}")
    
    if state.get('eligibility_path'):
        lines.append(f"- Eligibility path: {state['eligibility_path']}")
    
    if state.get('document_uploaded'):
        lines.append(f"- Document status: {state['document_uploaded']}")
    
    if not lines:
        lines.append("- No customer information captured yet")
    
    return "\n".join(lines)


def _get_stage_aware_system_prompt(language: str, state: Dict[str, Any], stage: str) -> str:
    """Get language-specific system prompt with stage awareness."""
    
    base_prompt = f"""You are BankGPT, a professional and friendly loan officer at Tata Capital.
Your current task: You are in the '{stage}' stage of the loan application process.

CRITICAL RULES:
1. Only ask for information that is NOT yet in the VERIFIED INFORMATION section
2. DO NOT re-ask questions - check what's already been provided
3. Use verified data as facts - don't question or recalculate
4. Stay focused on the current stage
5. When moving to next stage, acknowledge progress

STAGE CONTEXT:
- phone_asked: Customer has NOT given phone number - ASK FOR IT
- phone_provided: Phone just given - VERIFY and LOOKUP PROFILE
- amount_asked: Customer has verified phone - ASK FOR AMOUNT
- amount_provided: Amount just given - CHECK ELIGIBILITY
- eligibility_check: Both phone and amount available - DETERMINE IF APPROVED
- approved: Amount within limit - CONGRATULATE and PROVIDE EMI
- document_needed: Amount exceeds limit - REQUEST SALARY SLIP
- completed: Application complete - PROVIDE NEXT STEPS

IMPORTANT:
- Phone number is ALWAYS 10 digits - never confuse with amounts
- Amounts are in RUPEES - can be "5 lakhs", "500000", etc
- Only look at verified information in the state - ignore user's rephrasing of old info
- BE CONVERSATIONAL but PRECISE
- Acknowledge customer's current message but act based on current stage

When in '{stage}' stage:
- Do NOT ask about information from earlier stages
- Focus ONLY on what this stage needs
- Use the stage instructions provided separately"""
    
    if language == 'hindi':
        return f"""{base_prompt}

LANGUAGE: Hindi/Hinglish
Always respond in the language the customer is using.
Keep professional tone but use warm, conversational Hindi.
Use rupee amounts like "पाँच लाख" or "5 लाख"."""
    elif language == 'hinglish':
        return f"""{base_prompt}

LANGUAGE: Hinglish (Hindi + English mix)
Match the customer's language preference.
Use simple, conversational Hinglish.
Say amounts like "5 lakhs" or "five lakhs"."""
    else:
        return f"""{base_prompt}

LANGUAGE: English
Use clear, professional English.
Say amounts like "5 lakhs" or "500,000 rupees"."""


def _extract_information(user_input: str, state: Dict[str, Any], 
                         conversation_history: list, language: str) -> Dict[str, Any]:
    """
    Extract structured information from user input (backward compatibility).
    """
    return _extract_all_information(user_input, state, language)


def _extract_all_information(user_input: str, state: Dict[str, Any], language: str) -> Dict[str, Any]:
    """
    Extract ALL possible information from user input, not stage-dependent.
    Opportunistically extracts: name, phone, amount, income, etc.
    """
    import re
    extracted = {}
    user_lower = user_input.lower()
    
    # Extract name: "My name is John" or "I'm John" or "Call me John"
    if not state.get('customer_name'):
        name_patterns = [
            r"(?:my )?name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"i'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"call me\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"you can call me\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, user_input)
            if match:
                extracted['customer_name'] = match.group(1)
                break
    
    # Extract phone number (10 digits)
    if not state.get('phone'):
        phone_matches = re.findall(r'\b\d{10}\b', user_input)
        if phone_matches:
            phone = phone_matches[0]
            try:
                db = load_mock_db()
                ver_status, record = verification_agent(phone, db)
                if record:
                    extracted['phone'] = phone
                    extracted['customer_name'] = record.get('name', extracted.get('customer_name', ''))
                    extracted['credit_score'] = record.get('credit_score', 700)
                    extracted['pre_approved_limit'] = record.get('approved_amount', 500000)
                    extracted['income'] = record.get('income', 50000)
                    extracted['verified'] = True
            except:
                extracted['phone'] = phone
    
    # Extract loan amount flexibly
    if not state.get('requested_amount'):
        # Pattern 1: "X lakhs" or "X lakh" or "X lac"
        lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lac)', user_lower)
        if lakh_match:
            amount_value = float(lakh_match.group(1))
            extracted['requested_amount'] = int(amount_value * 100000)
        
        # Pattern 2: "X crore"
        elif re.search(r'(\d+(?:\.\d+)?)\s*crore', user_lower):
            crore_match = re.search(r'(\d+(?:\.\d+)?)\s*crore', user_lower)
            amount_value = float(crore_match.group(1))
            extracted['requested_amount'] = int(amount_value * 10000000)
        
        # Pattern 3: "X rupees" or "Rs X"
        elif re.search(r'(?:rupees?|rs\.?)\s*(\d+(?:,\d{3})*)', user_lower, re.IGNORECASE):
            rupee_match = re.search(r'(?:rupees?|rs\.?)\s*(\d+(?:,\d{3})*)', user_lower, re.IGNORECASE)
            amount_str = rupee_match.group(1).replace(',', '')
            amount = int(amount_str)
            if 100000 <= amount <= 100000000:  # Reasonable loan amount
                extracted['requested_amount'] = amount
        
        # Pattern 4: Large 6-7 digit numbers
        else:
            large_numbers = re.findall(r'\b(\d{6,7})\b', user_input)
            for num_str in large_numbers:
                num = int(num_str)
                if 100000 <= num <= 100000000:  # Between 1 lakh and 1 crore
                    extracted['requested_amount'] = num
                    break
    
    # Extract income if mentioned
    if not state.get('income'):
        income_patterns = [
            r'(?:monthly |monthly income|income is|earn)\s*(?:rupees?|rs\.?)\s*(\d+(?:,\d{3})*)',
            r'(?:per month|monthly)\s*(?:rupees?|rs\.?)\s*(\d+(?:,\d{3})*)',
        ]
        for pattern in income_patterns:
            match = re.search(pattern, user_lower)
            if match:
                income_str = match.group(1).replace(',', '')
                extracted['income'] = int(income_str)
                break
    
    return extracted


def _build_flexible_state_context(state: Dict[str, Any], extracted_info: Dict[str, Any], language: str) -> str:
    """Build state context combining existing state and newly extracted info."""
    info_lines = []
    
    # Combine existing state with extracted info
    all_info = {**state, **extracted_info}
    
    if all_info.get('customer_name'):
        info_lines.append(f"- Customer Name: {all_info['customer_name']}")
    
    if all_info.get('phone'):
        info_lines.append(f"- Phone: {all_info['phone']} {'(Verified)' if all_info.get('verified') else '(Not verified)'}")
    
    if all_info.get('requested_amount'):
        info_lines.append(f"- Requested Loan: ₹{all_info['requested_amount']:,}")
    
    if all_info.get('pre_approved_limit'):
        info_lines.append(f"- Pre-approved Limit: ₹{all_info['pre_approved_limit']:,}")
    
    if all_info.get('credit_score'):
        info_lines.append(f"- Credit Score: {all_info['credit_score']}")
    
    if all_info.get('income'):
        info_lines.append(f"- Monthly Income: ₹{all_info['income']:,}")
    
    return "\n".join(info_lines) if info_lines else "- No information collected yet"


def _determine_missing_info(state: Dict[str, Any], extracted_info: Dict[str, Any]) -> str:
    """Determine what essential information is still missing."""
    combined = {**state, **extracted_info}
    missing = []
    
    if not combined.get('customer_name'):
        missing.append("- Customer's name")
    
    if not combined.get('phone'):
        missing.append("- Phone number for verification")
    
    if not combined.get('requested_amount'):
        missing.append("- Loan amount needed")
    
    return "\n".join(missing) if missing else "- All information collected"


def _get_adaptive_greeting(language: str) -> str:
    """Get friendly, natural greeting."""
    greetings = {
        'hindi': "नमस्ते! 👋 मैं आपका BankGPT लोन असिस्टेंट हूँ। आपको कितना लोन चाहिए? मैं आपको 10 मिनट में अनुमोदन दे सकता हूँ।",
        'english': "Hello! 👋 I'm BankGPT, your loan assistant. How much would you like to borrow? I can get you approved in just 10 minutes."
    }
    return greetings.get(language, greetings['english'])


def _get_adaptive_system_prompt(language: str, state: Dict[str, Any], 
                                extracted_info: Dict[str, Any], missing_info: str) -> str:
    """Get adaptive system prompt based on conversation state."""
    
    return f"""You are BankGPT, a friendly and professional loan officer. 
You help customers apply for personal loans quickly and easily.

Key guidelines:
1. Be conversational and natural - don't rigidly follow a script
2. If customer mentions their loan need, explore it naturally
3. Extract information opportunistically (don't ask rigidly for phone then amount then income)
4. If customer provides unexpected information (like income when asked about amount), acknowledge it and use it
5. If they go off-topic, politely redirect: "That sounds great! Let's get your loan application started."
6. Be warm, encouraging, and professional
7. If they ask questions, answer helpfully
8. Don't ask for information they already provided

Language: {'Hindi/Hinglish mix' if language == 'hindi' else 'English'}"""


def _get_fallback_response(missing_info: str, language: str) -> str:
    """Get fallback response when LLM fails."""
    if language == 'hindi':
        return "कृपया अपना प्रश्न फिर से दोहराएं या बताएं कि आप कितना लोन चाहते हैं।"
    else:
        return "Could you tell me more about what you need? How much loan are you looking for?"


def _extract_information(user_input: str, state: Dict[str, Any], 
                         conversation_history: list, language: str) -> Dict[str, Any]:
    """
    Extract structured information from user input.
    Updates state with phone, amount, and other details.
    """
    import re
    extracted = {}
    
    # Extract phone number (10 digits) - but only if we haven't already got it
    if not state.get('phone'):
        # Look for 10 consecutive digits (phone number pattern)
        phone_matches = re.findall(r'\b\d{10}\b', user_input)
        
        if phone_matches:
            phone = phone_matches[0]  # Take first 10-digit match
            
            # Verify phone in database
            db = load_mock_db()
            ver_status, record = verification_agent(phone, db)
            
            if record:
                extracted['phone'] = phone
                extracted['customer_name'] = record.get('name', '')
                extracted['credit_score'] = record.get('credit_score', 700)
                extracted['pre_approved_limit'] = record.get('approved_amount', 500000)
                extracted['income'] = record.get('income', 50000)
                extracted['verified'] = True
    
    # Extract loan amount - only if we haven't already got it
    if not state.get('requested_amount'):
        user_lower = user_input.lower()
        
        # Pattern 1: "X lakhs" or "X lakh" or "X lac"
        lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lac)', user_lower)
        if lakh_match:
            amount_value = float(lakh_match.group(1))
            extracted['requested_amount'] = int(amount_value * 100000)
        
        # Pattern 2: "X crore" (if mentioned)
        crore_match = re.search(r'(\d+(?:\.\d+)?)\s*crore', user_lower)
        if crore_match and 'requested_amount' not in extracted:
            amount_value = float(crore_match.group(1))
            extracted['requested_amount'] = int(amount_value * 10000000)
        
        # Pattern 3: Large 6-7 digit numbers (like 500000 for 5 lakhs)
        if 'requested_amount' not in extracted:
            large_numbers = re.findall(r'\b(\d{6,7})\b', user_input)
            
            # Filter to reasonable loan amounts (10K to 1 Crore)
            for num_str in large_numbers:
                num = int(num_str)
                if 100000 <= num <= 100000000:  # Between 1 lakh and 1 crore
                    extracted['requested_amount'] = num
                    break
    
    # Determine eligibility path if we have both phone and amount

    if state.get('phone') and extracted.get('requested_amount'):
        pre_approved = state.get('pre_approved_limit', extracted.get('pre_approved_limit', 0))
        requested = extracted['requested_amount']
        
        if requested <= pre_approved:
            extracted['eligibility_path'] = 'FAST_TRACK'
        else:
            extracted['eligibility_path'] = 'CONDITIONAL_REVIEW'
    
    return extracted


def calculate_emi(principal: float, rate: float, tenure_months: int) -> float:
    """Simple EMI calculation."""
    r = rate / 100.0 / 12.0
    n = tenure_months
    if r == 0:
        return principal / n
    emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return emi


# Deprecated phase functions - kept for backward compatibility
def run_phase_1_sales(user_input: str, first_message: bool = True) -> Dict[str, Any]:
    """Deprecated - use run_unified_agent instead."""
    if first_message:
        return {'message': _get_greeting('english'), 'phase': 1}
    return {'message': "Please tell me about your loan needs.", 'phase': 1}


def run_phase_2_underwriting(user_input: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Deprecated - use run_unified_agent instead."""
    return {'message': "How can I help you?", 'phase': 2}


def run_phase_3_conditional(state: Dict[str, Any], uploaded_file) -> Dict[str, Any]:
    """Deprecated - use run_unified_agent instead."""
    return {'message': "Document received.", 'decision': 'Pending'}


def run_phase_4_sanction(state: Dict[str, Any]) -> Optional[str]:
    """Deprecated - use run_unified_agent instead."""
    return None
