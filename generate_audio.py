import torch
from transformers import VitsModel, VitsTokenizer
from scipy.io import wavfile
import streamlit as st

def text_to_speech(unique_number, text):
    st.toast("Generating Output...")
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")

    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Generate speech
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract the waveform
    speech = outputs["waveform"].numpy()[0]

    filename = f"recordings/output/output_{unique_number}.wav"

    # Save file
    wavfile.write(filename, 16000, speech)
    return filename