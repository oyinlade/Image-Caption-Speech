import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import tempfile
import os
import asyncio
import edge_tts

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()
    return processor, model

# Load the model and processor
processor, model = load_model()

# Function to generate captions
def generate_caption(image):
    image = image.convert("RGB")  # Ensure compatibility
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float32)
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    return caption

# Function to handle text-to-speech
def text_to_speech(text):
    output_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    async def tts():
        tts_engine = edge_tts.Communicate(text, "en-US-JennyNeural")
        await tts_engine.save(output_audio)
    
    asyncio.run(tts())
    return output_audio

# Streamlit UI
st.title("🖼️ Image Captioning with Speech Feedback")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

tts_enabled = st.checkbox("Enable Text-to-Speech")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            caption = generate_caption(image)
            st.success(f"Caption: {caption}")
            
            if tts_enabled:
                with st.spinner("Generating speech..."):
                    audio_file = text_to_speech(caption)
                    st.audio(audio_file, format="audio/mp3")
                    os.remove(audio_file)  # Clean up temp file
