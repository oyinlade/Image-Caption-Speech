# 🖼️ Image Caption Generator with TTS

Welcome to the **Image Caption Generator with Text-to-Speech**! 🚀

This app lets you upload an image, generates a caption for it using a Salesforce Blip model, and even reads it out loud using TTS. 😉
https://image-caption-speech-2nfzakrufpbixcn7mtzvz6.streamlit.app/

---

## 🔥 Features
- **Image Captioning**: Uses BLIP (Salesforce/blip-image-captioning-base) to generate descriptions of uploaded images.
- **Text-to-Speech (TTS)**: Converts captions to audio using Edge-TTS and provides a download link.
- **User-Friendly UI**: Built with Streamlit for an intuitive and responsive interface.

## 🛠️ Setup and Installation
### Prerequisites
Ensure you have Python installed (Python 3.8 or later is recommended).

### Installation
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
---

## 🏗️ How It Works

1. You upload an image.
2. The app processes it using **BLIP (Salesforce/blip-image-captioning-base)**.
3. AI generates a caption for the image.
4. The text is converted to speech using **edge-tts**.
5. You can listen to the audio or download it. 🎧

---

## 💡 Why This?

- Works locally, no API needed.
- Uses **FP16 for optimized performance**.
- Simple, clean UI with Streamlit.

---

## 🚀 Future Improvements
- Adding more voice options
- Supporting multiple languages
- Making the model run even faster




## 🔧 Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   or try the deployed version: https://image-caption-speech-2nfzakrufpbixcn7mtzvz6.streamlit.app/
2. Upload an image and click **Generate Caption**.
3. Listen to the generated caption or download the audio file.

## 📦 Dependencies
- `streamlit`
- `torch`
- `transformers`
- `PIL` (Pillow)
- `edge-tts`
- `asyncio`
- `base64`

## 💡 How It Works
1. The user uploads an image via the Streamlit interface.
2. The image is processed using the BLIP model to generate a descriptive caption.
3. The caption is then converted into speech using Edge-TTS.
4. The user can play the generated audio or download it as an MP3 file.

## ⚠️ Notes
- The first-time model load may take a few seconds, but subsequent processing is faster.
- Edge-TTS requires an active internet connection for speech synthesis.

## 📝 License
This project is open-source and available under the MIT License.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests to improve functionality.

## 🖼️ Demo Screenshot


---
Developed with ❤️ by Oyinlade.

