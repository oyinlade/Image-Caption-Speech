import streamlit as st
from PIL import Image
import cv2
import numpy as np
import torch
import asyncio
import base64
import time
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import edge_tts

# Set page configuration
st.set_page_config(
    page_title="Real-time Environment Captioner - Oyinlade's Treat 🌹",
    page_icon="📹",
    layout="centered"
)

# Session state initialization
if 'last_caption' not in st.session_state:
    st.session_state.last_caption = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'capture_interval' not in st.session_state:
    st.session_state.capture_interval = 3  # seconds between captures

@st.cache_resource
def load_model():
    """Load the optimized image captioning model."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", 
        torch_dtype=torch.float16
    )
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

# Function to generate TTS audio and return audio bytes
async def generate_audio(text, filename="speech.mp3"):
    """Generate audio file using edge-tts and return the audio bytes."""
    tts = edge_tts.Communicate(text, "en-US-AriaNeural")
    await tts.save(filename)
    
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    
    return audio_bytes

def main():
    st.title("📹 Real-time Environment Captioner")
    st.write("This app uses your webcam to caption what it sees in real-time, perfect for assisting blind users.")
    
    # Display info about first-time loading
    st.info("⚠️ First-time load may take a few seconds, but it gets faster after that 🤗🦾.")
    
    # Camera settings sidebar
    st.sidebar.header("Settings")
    st.session_state.capture_interval = st.sidebar.slider(
        "Seconds between captures", 
        min_value=1, 
        max_value=10, 
        value=st.session_state.capture_interval,
        help="How often to capture and analyze a new frame"
    )
    
    # Audio feedback settings
    enable_audio = st.sidebar.checkbox("Enable audio feedback", value=True)
    
    # Start/Stop camera 
    col1, col2 = st.sidebar.columns(2)
    start_button = col1.button("Start Camera")
    stop_button = col2.button("Stop Camera")
    
    # Create placeholder for the webcam feed
    video_placeholder = st.empty()
    
    # Create placeholder for captions
    caption_container = st.container()
    caption_header = caption_container.empty()
    caption_text = caption_container.empty()
    audio_placeholder = caption_container.empty()
    
    # Status display
    status_display = st.empty()
    
    if start_button:
        st.session_state.processing = True
        
    if stop_button:
        st.session_state.processing = False
        status_display.write("Camera stopped")
    
    # Main camera and processing loop
    if st.session_state.processing:
        # Access webcam
        try:
            status_display.write("Starting camera...")
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not access webcam. Please check your permissions.")
                st.session_state.processing = False
                return
                
            # Timestamp for frame capture interval
            last_capture_time = 0
            
            while st.session_state.processing:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image from webcam")
                    break
                
                # Convert to RGB (from BGR)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True, caption="Live Feed")
                
                current_time = time.time()
                # Process frame at specified intervals
                if current_time - last_capture_time >= st.session_state.capture_interval:
                    last_capture_time = current_time
                    
                    # Show processing status
                    status_display.write("Status: Processing frame...")
                    
                    # Convert to PIL Image for processing
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Get caption
                    caption = get_image_caption(pil_image)
                    
                    # Update last caption
                    st.session_state.last_caption = caption
                    
                    # Display caption
                    caption_header.subheader("Caption:")
                    caption_text.write(caption)
                    
                    # Generate and play audio if enabled
                    if enable_audio:
                        try:
                            audio_file = "caption_audio.mp3"
                            audio_bytes = asyncio.run(generate_audio(caption, audio_file))
                            audio_placeholder.audio(audio_bytes, format="audio/mp3")
                        except Exception as e:
                            audio_placeholder.error(f"Error generating audio: {str(e)}")
                    
                    # Update status
                    status_display.write("Status: Waiting for next capture")
                
                # Small sleep to reduce CPU usage
                time.sleep(0.01)
            
            # Release the webcam when stopped
            cap.release()
        
        except Exception as e:
            st.error(f"Error in webcam processing: {str(e)}")
            st.session_state.processing = False
    
    # Display instructions when not processing
    if not st.session_state.processing:
        st.info("Click 'Start Camera' in the sidebar to begin capturing and captioning your environment.")
        
        # Display sample capabilities
        st.subheader("How it works:")
        st.write("1. The app captures frames from your webcam in real-time")
        st.write("2. At regular intervals, it analyzes a frame and generates a description")
        st.write("3. The description is displayed and read aloud (when audio is enabled)")
        st.write("4. This continues until you click 'Stop Camera'")
        st.write("For blind users, this provides continuous audio descriptions of their surroundings.")

if __name__ == "__main__":
    main()
