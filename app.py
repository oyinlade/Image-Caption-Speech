import streamlit as st
from PIL import Image
import io
import requests
import pyttsx3
import threading
import time
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

# Function to speak text aloud
def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Error with text-to-speech: {str(e)}")

# Main app
def main():
    st.title("🖼️ Image Caption Generator with Text-to-Speech")
    st.write("Upload an image, and I'll describe what I see and read it aloud!")
    
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
                
                # Speak the caption
                st.write("🔊 Speaking caption...")
                
                # Use threading to avoid blocking the UI
                threading.Thread(target=speak_text, args=(caption,)).start()
                
                st.success("Done!")

if __name__ == "__main__":
    main()