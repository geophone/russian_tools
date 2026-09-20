### whisper2.py requirements
```bash
sudo apt update
sudo apt install -y pulseaudio-utils python3-venv

python3 -m venv .venv
source .venv/bin/activate

pip install --break-system-packages faster-whisper numpy piper-tts
python3 -m piper.download_voices en_US-lessac-medium
```

### translation loop requirements
```bash
pip install --break-system-packages sentencepice
```

### screen translation requirements
```bash
pip install --break-system-packages mss easyocr opencv-python-headless PyQt6 transformers sentencepiece sacremoses pytesseract
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-rus
```
