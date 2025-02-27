import streamlit as st
from PIL import Image
import io
import base64
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import edge_tts
import asyncio

# Set page configuration
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the optimized image captioning model."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16)
    return processor, model

# Function to generate image caption
def get_image_caption(image):
    processor, model = load_model()
    
    # Convert image for processing
    inputs = processor(images=image, return_tensors="pt").to(model.device, torch.float16)
    
    # Generate caption
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Function to generate TTS audio and return a download link
async def get_audio_download_link(text, filename="speech.mp3"):
    """Generate audio file using edge-tts and return a download link."""
    tts = edge_tts.Communicate(text, "en-US-AriaNeural")
    await tts.save(filename)
    
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download Audio</a>'
    return href, audio_bytes

# Main app
def main():
    st.title("🖼️ Image Caption Generator with Text-to-Speech")
    st.write("Upload an image, and I'll describe what I see!")
    
    st.info("⚠️ First-time load may take a few seconds while the model is initialized.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Generate Caption"):
            with st.spinner("Analyzing image..."):
                caption = get_image_caption(image)
                st.subheader("Caption:")
                st.write(caption)
                
                # Generate TTS audio
                try:
                    st.markdown("### 🔊 Audio")
                    audio_file = "caption_audio.mp3"
                    download_link, audio_bytes = asyncio.run(get_audio_download_link(caption, audio_file))
                    st.audio(audio_bytes, format="audio/mp3")
                    st.markdown(download_link, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating audio: {str(e)}")
                
                st.success("Done!")

if __name__ == "__main__":
    main()
