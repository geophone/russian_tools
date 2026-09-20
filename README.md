# Vibe coded Russian translation utils 
* The whisper2 isn't bad and the translation loops are good
* Screen ocr is still wip have to refine requirements and think about it.  Basically just a loop of tesseract -l rus <screenshot path> stdout

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
this is very wip and doens't work as well as I'd like
```bash
pip install --break-system-packages pipewire-capture mss easyocr opencv-python-headless PyQt6 transformers sentencepiece sacremoses pytesseract
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-rus
```
