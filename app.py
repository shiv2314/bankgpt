"""
BankGPT - Persuasive, End-to-End Loan Processor with Voice
Split-Screen: Chat (Left) + XAI Panel (Right)
Unified Conversation Flow: Natural LLM-driven dialogue
Voice Input/Output: Speech recognition and text-to-speech
"""

import streamlit as st
import time
from pathlib import Path

from master_agent import run_unified_agent
from session_manager import init_session, get_conversation_state, update_conversation_state, add_message, get_messages
from voice_helper import speak_text, is_voice_available, get_audio_input_widget, recognize_speech_from_streamlit_audio, check_microphone_permission
from document_helper import generate_sanction_letter_pdf
from utils import compute_emi

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BankGPT | Loan Processor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Dark Theme - BankGPT) ---
st.markdown("""
    <style>
    /* Dark background for main app */
    .stApp { 
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d0d0d;
    }
    
    /* All text white */
    .stMarkdown, .stWrite, .stText { 
        color: #ffffff !important;
    }
    
    /* Sidebar text white */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .stWrite,
    [data-testid="stSidebar"] .stText { 
        color: #ffffff !important;
    }
    
    /* Chat message styling */
    .chat-message-user { 
        background-color: #2d3748; 
        border-radius: 15px; 
        padding: 10px; 
        margin: 8px 0;
        color: #ffffff;
    }
    .chat-message-assistant { 
        background-color: #374151; 
        border-radius: 15px; 
        padding: 10px; 
        margin: 8px 0;
        color: #ffffff;
    }
    
    /* XAI Panel */
    .xai-panel { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        color: white; 
        padding: 20px; 
        border-radius: 10px;
    }
    
    /* Metric values */
    div[data-testid="stMetricValue"] { 
        font-size: 20px; 
        font-weight: bold;
        color: #ffffff;
    }
    
    /* Chat input styling */
    .stChatInput input { 
        color: #ffffff !important;
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
    }
    
    /* Chat input placeholder */
    .stChatInput input::placeholder {
        color: #a0aec0 !important;
    }
    
    /* Button styling */
    .stButton button { 
        color: white !important;
        font-weight: bold;
        background-color: #667eea !important;
        border-color: #667eea !important;
    }
    
    .stButton button:hover {
        background-color: #764ba2 !important;
    }
    
    /* Container styling */
    .stContainer { 
        border-color: #4a5568;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Divider */
    hr {
        border-color: #4a5568 !important;
    }
    
    /* Radio and checkbox */
    [role="radio"], [role="checkbox"] {
        color: #ffffff !important;
    }
    
    /* Select options */
    .stSelectbox, .stMultiSelect {
        color: #ffffff !important;
    }
    
    /* Tabs */
    .stTabs {
        background-color: #2d3748;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
init_session()

# --- SIDEBAR: BRANDING & CONTROLS ---
with st.sidebar:
    st.markdown("# 🏦 BankGPT")
    st.markdown("*Personal Loan in 10 Minutes*")
    st.divider()
    
    # Language & Voice Settings
    language = st.radio("🌐 Language", ["English", "हिंदी"], horizontal=False, key="sidebar_language")
    
    # Voice settings - Streamlit handles microphone permissions automatically
    voice_available = is_voice_available()
    
    if voice_available:
        voice_mode = st.toggle("🎙️ Voice Input/Output", value=False, key="sidebar_voice",
                               help="Use voice input with automatic microphone permission handling")
        if voice_mode:
            tts_enabled = st.toggle("🔊 Text-to-Speech", value=False, key="sidebar_tts")
            st.info("🎤 Microphone access will be requested when you use voice input")
        else:
            tts_enabled = False
    else:
        voice_mode = False
        tts_enabled = False
        st.warning("⚠️ Voice features may not be available in this environment")
    
    # Demo mode for testing (optional fallback)
    if not voice_mode:
        demo_voice_mode = st.toggle("🎭 Demo Voice Mode", value=False, key="sidebar_demo_voice", 
                                    help="Text input that simulates voice for testing")
    else:
        demo_voice_mode = False
    
    # Update session state with settings
    state = get_conversation_state()
    update_conversation_state({
        'voice_enabled': voice_mode,
        'tts_enabled': tts_enabled,
        'demo_voice_mode': demo_voice_mode,
        'language': language,
        'voice_available': voice_available
    })
    
    st.divider()
    st.subheader(" Document Upload")
    
    # Document upload section
    st.markdown("**Upload Files (if required)**")
    uploaded_file = st.file_uploader(
        "Choose a PDF or image file",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        key='salary_slip_upload',
        help="Upload your salary slip for income verification"
    )
    
    if uploaded_file:
        # Store uploaded file info in session state
        st.session_state['uploaded_document'] = {
            'name': uploaded_file.name,
            'type': uploaded_file.type,
            'size': uploaded_file.size
        }
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info("Document will be verified during loan processing.")
        # Reflect upload into conversation state so the agent can progress
        update_conversation_state({'document_uploaded': True})

    st.divider()
    st.subheader("📑 Sanction Letter")
    state = get_conversation_state()
    stage = state.get('conversation_stage', 'greeting')
    loan_amount = state.get('requested_amount') or state.get('pre_approved_limit')

    can_generate = stage in ('approved', 'completed') and loan_amount is not None
    if can_generate:
        # Prepare PDF and show download option
        pdf_bytes = generate_sanction_letter_pdf(state, interest_rate=12.0, tenure_months=60)
        estimated_emi = compute_emi(float(loan_amount), 12.0, 60)

        st.download_button(
            "⬇️ Download Sanction Letter (PDF)",
            data=pdf_bytes,
            file_name="sanction_letter.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.caption(f"EMI estimate: Rs. {estimated_emi:,.0f} @ 12% for 60 months")
    else:
        st.info("Sanction letter will be available after approval is confirmed.")


    
    if st.button("🔄 Start New Application", use_container_width=True):
        st.session_state.clear()
        st.rerun()
        init_session()
        st.rerun()
    
    st.divider()
    
    # Contact Us Section
    st.subheader("📞 Contact Us")
    st.markdown("""
    **Our team is ready to help:**
    
    📱 **Phone**  
    1800-BANKGPT (1800-226-5478)
    
    💬 **WhatsApp**  
    +91-XXXXXXXXXX
    
    📧 **Email**  
    support@bankgpt.in
    
    ⏰ **Hours**  
    Monday-Friday, 9 AM - 6 PM IST
    """)
    
    st.divider()
    
    # FAQ Section
    st.subheader("❓ Frequently Asked Questions")
    
    with st.expander("💰 What is the interest rate?"):
        st.write("""
        Our competitive interest rate is **8.5% per annum** for personal loans.
        The exact rate may vary based on your credit profile and income.
        """)
    
    with st.expander("⏱️ How long does approval take?"):
        st.write("""
        For pre-approved customers, approval is **INSTANT** (less than 1 minute)!
        
        For new customers, it typically takes 2-4 hours.
        Once approved, funds are disbursed within 2-4 business hours.
        """)
    
    with st.expander("📄 What documents do I need?"):
        st.write("""
        **Minimum documents required:**
        - Valid ID proof (Aadhar/PAN/Passport)
        - Address proof (Recent utility bill/bank statement)
        - Income proof (Salary slip/IT returns)
        
        For pre-approved customers within their limit:
        No documents needed at approval stage!
        """)
    
    with st.expander("💳 Can I prepay my loan?"):
        st.write("""
        Yes! You can prepay your loan anytime **after 6 months** with:
        - **Zero prepayment penalty** ✅
        - No hidden charges
        - Instant credit of prepaid amount to principal
        """)
    
    with st.expander("🛡️ Is my personal information safe?"):
        st.write("""
        Absolutely! We use:
        - **256-bit SSL encryption** for all data
        - **RBI-compliant** security standards
        - **Zero data sharing** without your consent
        - Your information is PCI-DSS compliant
        """)
    
    with st.expander("🏦 Who is BankGPT?"):
        st.write("""
        BankGPT is an AI-powered loan processing platform by Tata Capital.
        
        We provide:
        - **Instant loan approvals** for pre-approved customers
        - **AI-driven personalization** for better rates
        - **24/7 chatbot support** via voice & text
        - **Transparent process** with clear terms & conditions
        """)
    
    with st.expander("✅ What's the maximum loan amount?"):
        st.write("""
        **Maximum loan amount:** Up to ₹25 lakhs
        
        Your actual pre-approved limit depends on:
        - Your income level
        - Credit score
        - Employment stability
        - Existing obligations
        """)

# --- MAIN LAYOUT: CHAT (Left 2/3) vs XAI PANEL (Right 1/3) ---
col_chat, col_xai = st.columns([2, 1.2], gap="large")

# ================== RIGHT COLUMN: XAI PANEL (GLASS BOX) ==================
with col_xai:
    st.markdown("### 🧠 Conversation Context")
    
    state = get_conversation_state()
    
    # Metrics Row
    with st.container(border=True):
        m1, m2 = st.columns(2)
        
        credit_score = state.get('credit_score', '--')
        verified = "✅" if state.get('verified') else "⏳"
        
        m1.metric("Credit Score", credit_score, delta="Safe" if credit_score != '--' else None)
        m2.metric("Verification", verified)
        
        st.divider()
        
        # Eligibility Status
        eligibility = state.get('eligibility_path', 'Pending')
        st.markdown(f"**Eligibility Path: {eligibility}**")
    
    st.divider()
    
    # Application Status
    st.markdown("**📋 Application Summary**")
    with st.container(border=True):
        # Display name if available from CRM lookup
        if state.get('customer_name'):
            st.write(f"👤 **Name:** {state.get('customer_name')}")
        
        # Phone number is the unique identifier
        if state.get('phone'):
            st.write(f"📱 **ID (Phone):** {state.get('phone')}")
        
        if state.get('requested_amount'):
            st.write(f"💰 **Requested:** ₹{state.get('requested_amount'):,}")
        
        if state.get('pre_approved_limit'):
            st.write(f"✅ **Pre-approved Limit:** ₹{state.get('pre_approved_limit'):,}")
        
        if state.get('income'):
            st.write(f"💼 **Income:** ₹{state.get('income'):,}/month")


# ================== LEFT COLUMN: CHAT INTERFACE ==================
with col_chat:
    st.markdown("### 💬 Chat with BankGPT")
    
    # Chat History Display
    messages = get_messages()
    for msg in messages:
        role = msg['role']
        content = msg['content']
        with st.chat_message(role):
            st.write(content)
    
    st.divider()
    
    # ===== VOICE INPUT SECTION =====
    state = get_conversation_state()
    voice_enabled = state.get('voice_enabled', False)
    tts_enabled = state.get('tts_enabled', False)
    voice_available = state.get('voice_available', False)
    demo_voice_mode = state.get('demo_voice_mode', False)
    messages = get_messages()
    
    # Voice input controls
    if voice_enabled and (voice_available or demo_voice_mode):
        st.markdown("### 🎤 Voice Input Mode Active")
        
        if demo_voice_mode:
            # Demo mode: text input simulates voice
            st.info("🎭 Demo Mode: Type your message (simulates voice input)")
            voice_input_text = st.text_input("Simulated Voice Input:", key="demo_voice_input", 
                                            placeholder="Type what you would say...")
            if voice_input_text:
                st.session_state['voice_input'] = voice_input_text
                st.success(f"✅ Simulated Input: {voice_input_text}")
        else:
            # Real microphone mode using Streamlit's audio input
            st.markdown("#### 🎙️ Voice Recording")
            
            # Show microphone permission info
            check_microphone_permission()
            
            # Audio input widget
            audio_bytes = get_audio_input_widget(key="main_audio_input")
            
            if audio_bytes:
                with st.spinner("🔄 Transcribing audio..."):
                    recognized_text = recognize_speech_from_streamlit_audio(audio_bytes)
                    if recognized_text:
                        st.session_state['voice_input'] = recognized_text
                        st.success(f"✅ Transcribed: {recognized_text}")
                        st.rerun()  # Refresh to process the input
                    else:
                        st.error("❌ Could not transcribe audio. This could be due to:")
                        st.markdown("""
                        - **Audio quality**: Make sure you're in a quiet environment
                        - **Speaking clarity**: Speak clearly and at normal pace
                        - **Internet connection**: Google Speech Recognition requires internet
                        - **Audio format**: The recording might not be in a supported format
                        - **Audio length**: Try recording for 3-10 seconds
                        """)
                        st.info("💡 **Tips for better recognition:**")
                        st.markdown("""
                        - Speak directly into the microphone
                        - Use simple, clear sentences
                        - Avoid background noise
                        - Ensure stable internet connection
                        """)
    
    # Regular text input or voice input
    if voice_enabled and (voice_available or demo_voice_mode) and 'voice_input' in st.session_state:
        user_input = st.session_state.voice_input
        st.session_state.voice_input = None  # Clear for next input
        st.info(f"📝 Processing: {user_input}")
    else:
        user_input = st.chat_input("Type your message..." if not voice_enabled else "Speak or type...")
    
    if user_input:
        # Add user message
        add_message('user', user_input)
        
        with st.spinner("🤖 Processing..."):
            # Call unified agent for entire conversation flow
            result = run_unified_agent(user_input, state, messages)
            bot_response = result['message']
            
            # Add bot response
            add_message('assistant', bot_response)
            
            # Text-to-speech if enabled
            if tts_enabled:
                language_code = 'hindi' if result.get('detected_language') == 'hindi' else 'english'
                speak_text(bot_response, language=language_code, async_mode=True)
            
            # Update state with extracted information
            update_dict = {
                'detected_language': result.get('detected_language', state.get('detected_language', 'english')),
                'conversation_stage': result.get('conversation_stage', state.get('conversation_stage', 'greeting'))
            }
            
            # Add extracted info if present
            if result.get('phone'):
                update_dict['phone'] = result['phone']
            if result.get('customer_name'):
                update_dict['customer_name'] = result['customer_name']
            if result.get('credit_score'):
                update_dict['credit_score'] = result['credit_score']
            if result.get('pre_approved_limit'):
                update_dict['pre_approved_limit'] = result['pre_approved_limit']
            if result.get('income'):
                update_dict['income'] = result['income']
            if result.get('requested_amount'):
                update_dict['requested_amount'] = result['requested_amount']
            if result.get('eligibility_path'):
                update_dict['eligibility_path'] = result['eligibility_path']
            if result.get('verified'):
                update_dict['verified'] = result['verified']
            
            update_conversation_state(update_dict)
        
        st.rerun()
