# voice_helper.py - Speech recognition and text-to-speech utilities
"""
Handles voice input (speech-to-text) and voice output (text-to-speech).
Uses Streamlit's st.audio_input for recording and Google Speech Recognition for transcription.
"""

import speech_recognition as sr
import pyttsx3
import threading
import tempfile
import os
import io
from typing import Optional, Tuple
import streamlit as st

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speaking rate
tts_engine.setProperty('volume', 0.9)  # Volume (0-1)


def speak_text(text: str, language: str = 'english', async_mode: bool = True) -> None:
    """
    Convert text to speech and play it.
    
    Args:
        text: Text to speak
        language: Language for speech output
        async_mode: If True, speak in background thread
    """
    try:
        # Set language-specific voice
        voices = tts_engine.getProperty('voices')
        if language == 'hindi':
            # Try to find Hindi voice, fallback to default
            tts_engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
        else:
            # Use default English voice
            tts_engine.setProperty('voice', voices[0].id if voices else None)
        
        # Queue the text
        tts_engine.say(text)
        
        # Speak
        if async_mode:
            # Run in background thread
            thread = threading.Thread(target=tts_engine.runAndWait)
            thread.daemon = True
            thread.start()
        else:
            # Block until done
            tts_engine.runAndWait()
    
    except Exception as e:
        print(f"TTS Error: {e}")


def transcribe_audio_with_google(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribe audio bytes using Google Speech Recognition.
    
    Args:
        audio_bytes: Audio data as bytes
        
    Returns:
        Transcribed text or None if failed
    """
    recognizer = sr.Recognizer()
    temp_file_path = None
    
    try:
        # Create a temporary file with the original audio format
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        # Debug: Check file size
        file_size = os.path.getsize(temp_file_path)
        print(f"🔍 Audio file size: {file_size} bytes")
        
        if file_size == 0:
            print("❌ Audio file is empty")
            return None
        
        # Try to read the audio file directly first
        try:
            with sr.AudioFile(temp_file_path) as source:
                # Adjust for ambient noise and record the audio
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source)
                
                # Debug: Check if audio was recorded
                if len(audio.frame_data) == 0:
                    print("❌ No audio data found in file")
                    return None
                    
                print(f"🔍 Audio frames: {len(audio.frame_data)} bytes")
        except Exception as e:
            print(f"❌ Error reading audio file: {e}")
            return None
        
        # Try to recognize using Google Speech Recognition API
        print("🔄 Sending to Google Speech Recognition...")
        text = recognizer.recognize_google(audio, language="en-US")
        print(f"✅ Transcribed: {text}")
        return text
    
    except sr.UnknownValueError:
        print("❌ Google Speech Recognition could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"❌ Could not request results from Google Speech Recognition service: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error in transcription: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Ensure temp file is cleaned up
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print("🗑️ Cleaned up temporary audio file")
            except:
                pass


def get_audio_input_widget(key: str = "audio_input") -> Optional[bytes]:
    """
    Display Streamlit audio input widget and return audio bytes.
    
    Args:
        key: Unique key for the widget
        
    Returns:
        Audio bytes or None if no audio recorded
    """
    try:
        audio_file = st.audio_input(
            "🎤 Record your message",
            key=key,
            help="Click the microphone button to start recording. Speak clearly for 3-10 seconds."
        )
        
        if audio_file:
            # Read bytes from UploadedFile object
            audio_bytes = audio_file.read()
            st.write(f"📊 Audio data received: {len(audio_bytes)} bytes")
            return audio_bytes
            
        return None
    except Exception as e:
        st.error(f"❌ Audio recording error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def transcribe_audio_simple(audio_bytes: bytes) -> Optional[str]:
    """
    Simple audio transcription using alternative approach.
    
    Args:
        audio_bytes: Audio data as bytes
        
    Returns:
        Transcribed text or None if failed
    """
    recognizer = sr.Recognizer()
    
    try:
        # Create an audio data object directly from bytes
        audio_data = sr.AudioData(audio_bytes, 16000, 2)  # Assume 16kHz, 16-bit
        
        # Try to recognize
        text = recognizer.recognize_google(audio_data, language="en-US")
        print(f"✅ Simple transcription: {text}")
        return text
        
    except Exception as e:
        print(f"❌ Simple transcription failed: {e}")
        return None


def recognize_speech_from_streamlit_audio(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribe audio from Streamlit's audio_input widget.
    
    Args:
        audio_bytes: Audio data from st.audio_input
        
    Returns:
        Transcribed text or None if failed
    """
    if not audio_bytes:
        return None
    
    # Try the file-based approach first
    result = transcribe_audio_with_google(audio_bytes)
    
    if result:
        return result
    
    # If that fails, try the simple approach
    print("🔄 Trying alternative transcription method...")
    result = transcribe_audio_simple(audio_bytes)
    
    return result


def is_voice_available() -> bool:
    """
    Check if voice input/output is available.
    With Streamlit's audio_input, this is always True since Streamlit handles microphone permissions.
    """
    return True


def check_microphone_permission() -> bool:
    """
    Display information about microphone permissions.
    Streamlit's audio_input will automatically request microphone permission.
    
    Returns:
        Always True since Streamlit handles permissions
    """
    st.info(
        "🎤 **Microphone Setup:**\n"
        "- Make sure your browser allows microphone access\n"
        "- Click the microphone icon below to start recording\n"
        "- Speak clearly and wait for transcription"
    )
    return True
