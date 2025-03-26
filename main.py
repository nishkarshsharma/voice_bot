import streamlit as st
import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import speech_recognition as sr
from google.cloud import storage
from generate_audio import text_to_speech
import tempfile
import base64

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "env.json" 

def generate_uuid(length: int = 8) -> str:
    """Generate a random unique identifier."""
    import random
    import string
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

@st.cache_resource
def load_config():
    """Load configuration from config.json."""
    with open("config.json") as config:
        import json
        return json.load(config)

config = load_config()
MODEL_ID = config["MODEL_ID"]
PROJECT_ID = config["PROJECT_ID"]
REGION = config["REGION"]
BUCKET = config["BUCKET"]

if not os.path.exists("recordings"):
    os.makedirs("recordings")

if not os.path.exists("recordings/output"):
    os.makedirs("output")


vertexai.init(project=PROJECT_ID, location=REGION)
model = GenerativeModel(MODEL_ID)

def get_base64_of_audio(audio_path):
    """Convert audio file to base64 for embedding."""
    with open(audio_path, 'rb') as audio_file:
        return base64.b64encode(audio_file.read()).decode()

def autoplay_audio(audio_path):
    """Create an HTML5 audio autoplay element."""
    base64_audio = get_base64_of_audio(audio_path)
    return f"""
    <audio autoplay>
        <source src="data:audio/wav;base64,{base64_audio}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Upload a file to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def recognize_speech_from_mic(unique_number):
    """Capture speech from microphone, save, and upload to GCS."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        st.toast('Listening... Speak now')
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix=f"recorded_audio-{unique_number}-") as temp_audio:
        st.toast('Processing Audio...')
        temp_audio_path = temp_audio.name
        temp_audio.write(audio.get_wav_data())
    
    upload_location = config["BUCKET"]
    upload_to_gcs(upload_location, temp_audio_path, f"recorded_audio-{unique_number}.wav")
    
    try:
        transcription = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        transcription = "Could not understand audio"
    except sr.RequestError:
        transcription = "Could not request results"
    
    os.unlink(temp_audio_path)
    
    return f"gs://{BUCKET}/recorded_audio-{unique_number}.wav", transcription

def main():
    st.markdown("""
    <style>
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .chat-header {
        text-align: center;
        color: #333;
        margin-bottom: 20px;
    }
    .chat-message {
        margin-bottom: 15px;
        padding: 10px;
        border-radius: 8px;
    }
    .user-message {
        background-color: #e6f3ff;
        border-left: 4px solid #3391ff;
    }
    .assistant-message {
        background-color: #f0f0f0;
        border-left: 4px solid #4CAF50;
    }
    /* Center the button container */
    .stButton {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    /* Align clear chat to the right */
    .clear-chat-container {
        display: flex;
        justify-content: flex-end;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="chat-header">🎙️ HOME LLC</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Voice Bot by Nishkarsh Sharma</p>', unsafe_allow_html=True)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    for chat in st.session_state.chat_history:
        if chat['type'] == 'user':
            st.markdown(f'<div class="chat-message user-message">👤 {chat["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">🤖 {chat["content"]}</div>', unsafe_allow_html=True)
        
        # Autoplay audio if exists
        if chat.get('audio'):
            st.components.v1.html(autoplay_audio(chat['audio']), height=0)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="stButton">', unsafe_allow_html=True)
        start_listening = st.button("🎤 Start Listening", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="clear-chat-container">', unsafe_allow_html=True)
        clear_chat = st.button("🗑️ Clear Chat")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if clear_chat:
        st.session_state.chat_history = []
        st.rerun()
    
    if start_listening:
        unique_number = generate_uuid()
        
        try:
            audio_file_uri, user_transcription = recognize_speech_from_mic(unique_number)
            
            st.session_state.chat_history.append({
                'type': 'user', 
                'content': user_transcription
            })
            
            prompt = """Please provide a helpful and conversational response to the following input. 
            Ensure the response is clear, concise, and engaging."""
            
            audio_file = Part.from_uri(audio_file_uri, mime_type="audio/wav")
            contents = [audio_file, prompt, user_transcription]
            response = model.generate_content(contents)
            
            output_file_name = text_to_speech(unique_number, response.text)
            
            st.session_state.chat_history.append({
                'type': 'assistant', 
                'content': response.text,
                'audio': output_file_name
            })
            
            st.rerun()
            os.remove(output_file_name)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()