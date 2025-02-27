import streamlit as st
from PIL import Image
import io
import os
import base64
from transformers import pipeline
import edge_tts
import asyncio

# Set page configuration
st.set_page_config(
    page_title="Fast Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def load_captioner():
    """Load the smaller image captioning model (faster startup)"""
    try:
        return pipeline("image-to-text", model="Salesforce/blip-image-captioning-small")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Generate caption using the cached model
def get_image_caption(image):
    try:
        captioner = load_captioner()
        if captioner is None:
            return "Failed to load the captioning model."

        with st.spinner("Generating caption..."):
            result = captioner(image)

        return result[0].get("generated_text", "No caption generated.") if result else "No caption found."

    except Exception as e:
        return f"Error processing image: {str(e)}"

# Generate TTS using EdgeTTS
async def generate_tts(text, filename="caption_audio.mp3"):
    tts = edge_tts.Communicate(text, "en-US-JennyNeural")
    await tts.save(filename)

# Create download link for the TTS audio
def get_audio_download_link(filename="caption_audio.mp3", text="Download Audio"):
    with open(filename, "rb") as f:
        audio_bytes = f.read()

    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Main App
def main():
    st.title("🖼️ Fast Image Caption Generator with TTS")
    st.write("Upload an image, and I'll describe what I see!")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate caption button
        if st.button("Generate Caption"):
            caption = get_image_caption(image)

            # Display caption
            st.subheader("Caption:")
            st.write(caption)

            # Generate TTS audio
            try:
                audio_file = "caption_audio.mp3"
                asyncio.run(generate_tts(caption, audio_file))

                # Audio player
                st.markdown("### 🔊 Audio")
                audio_bytes = open(audio_file, "rb").read()
                st.audio(audio_bytes, format="audio/mp3")

                # Download link
                st.markdown(get_audio_download_link(audio_file, "Download audio file"), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating audio: {str(e)}")

            st.success("Done!")

if __name__ == "__main__":
    main()
