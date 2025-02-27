import streamlit as st
from PIL import Image
import io
import requests
from gtts import gTTS
import os
import base64
from transformers import pipeline

# Set page configuration
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def load_captioner():
    """Load the image captioning model"""
    try:
        return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to generate image caption using local Hugging Face model
def get_image_caption(image):
    try:
        # Load the captioner (uses caching so it only loads once)
        captioner = load_captioner()
        
        if captioner is None:
            return "Failed to load the captioning model."
        
        # Generate caption
        with st.spinner("Model is processing the image..."):
            result = captioner(image)
        
        if result and len(result) > 0 and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, list) and len(result) > 0:
            # Some models return a different format
            return result[0].get("text", str(result[0]))
        else:
            return str(result)
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Function to create a link to download the TTS audio
def get_audio_download_link(tts, filename="speech.mp3", text="Download Audio"):
    """Generates a download link for the TTS audio file"""
    with open(filename, "wb") as f:
        tts.write_to_fp(f)
    
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Main app
def main():
    st.title("🖼️ Image Caption Generator with Text-to-Speech")
    st.write("Upload an image, and I'll describe what I see!")
    
    # Display info about first-time loading
    st.info("⚠️ Note: The first time you run this app, it will download the captioning model which might take a few minutes. Subsequent runs will be much faster.")
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("Generate Caption"):
            with st.spinner("Analyzing image..."):
                # Get caption
                caption = get_image_caption(image)
                
                # Display caption
                st.subheader("Caption:")
                st.write(caption)
                
                # Generate TTS audio
                try:
                    tts = gTTS(text=caption, lang='en')
                    
                    # Create a download link for the audio
                    st.markdown("### 🔊 Audio")
                    audio_file = "caption_audio.mp3"
                    tts.save(audio_file)
                    
                    # Audio player
                    audio_bytes = open(audio_file, "rb").read()
                    st.audio(audio_bytes, format="audio/mp3")
                    
                    # Download link
                    st.markdown(get_audio_download_link(tts, audio_file, "Download audio file"), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating audio: {str(e)}")
                
                st.success("Done!")

if __name__ == "__main__":
    main()
