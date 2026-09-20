#!/usr/bin/env python3

import os
import re
import time
import queue
import tempfile
import threading
import subprocess
import tkinter as tk

import pytesseract
import torch

from collections import deque
from PIL import Image, ImageEnhance, ImageOps
from transformers import MarianMTModel, MarianTokenizer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"

# Take a new full-screen screenshot every 5 seconds.
CAPTURE_INTERVAL = 5.0

# Tesseract sparse-text mode works well for webpages/chat.
OCR_CONFIG = "--oem 3 --psm 11"

# Save the latest screenshot here so you can inspect it.
LATEST_SCREENSHOT = "/tmp/russian_screen_latest.png"

SEEN_MESSAGE_LIMIT = 500


# ============================================================
# STATE
# ============================================================

stop_event = threading.Event()
ui_queue = queue.Queue()

seen_messages = deque(maxlen=SEEN_MESSAGE_LIMIT)
seen_set = set()


# ============================================================
# CYRILLIC
# ============================================================

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def contains_cyrillic(text):
    return bool(CYRILLIC_RE.search(text))


def cyrillic_count(text):
    return len(CYRILLIC_RE.findall(text))


# ============================================================
# CHECK SCREENSHOT COMMAND
# ============================================================

if subprocess.run(
    ["which", "gnome-screenshot"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
).returncode != 0:

    raise RuntimeError(
        "gnome-screenshot is not installed.\n\n"
        "Run:\n"
        "sudo apt install gnome-screenshot"
    )


# ============================================================
# CHECK TESSERACT
# ============================================================

print("[status] Checking Tesseract...")

languages = pytesseract.get_languages(config="")

print("[status] Tesseract languages:", languages)

if "rus" not in languages:
    raise RuntimeError(
        "Russian OCR data is not installed.\n\n"
        "Run:\n"
        "sudo apt install tesseract-ocr-rus"
    )


# ============================================================
# LOAD TRANSLATOR
# ============================================================

print()
print("[status] Loading translator...")
print("[status]", MODEL_NAME)

tokenizer = MarianTokenizer.from_pretrained(
    MODEL_NAME
)

translator = MarianMTModel.from_pretrained(
    MODEL_NAME
)

translator.eval()

device = torch.device("cpu")
translator.to(device)

print("[status] Translator ready")

print(
    "[status] Session:",
    os.environ.get("XDG_SESSION_TYPE", "unknown"),
)


# ============================================================
# GNOME SCREENSHOT CAPTURE
# ============================================================

def capture_screen():
    """
    Capture the desktop using GNOME itself.

    This avoids direct framebuffer/X11 grabbing, which can
    produce black images under Wayland.
    """

    # Remove the old screenshot so we never accidentally
    # process a stale image.
    try:
        os.remove(LATEST_SCREENSHOT)
    except FileNotFoundError:
        pass

    result = subprocess.run(
        [
            "gnome-screenshot",
            "-f",
            LATEST_SCREENSHOT,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "gnome-screenshot failed:\n"
            + result.stderr
        )

    if not os.path.isfile(
        LATEST_SCREENSHOT
    ):
        raise RuntimeError(
            "gnome-screenshot returned successfully "
            "but no screenshot file was created."
        )

    image = Image.open(
        LATEST_SCREENSHOT
    )

    image.load()

    return image.convert("RGB")


# ============================================================
# TRANSLATION
# ============================================================

def translate_ru_to_en(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        result = translator.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
        )

    return tokenizer.decode(
        result[0],
        skip_special_tokens=True,
    ).strip()


# ============================================================
# TEXT CLEANUP
# ============================================================

def normalize_text(text):
    return " ".join(
        text.split()
    ).strip()


def strip_username(text):
    """
    Example:

        foxyfox_26: белкокрады позорные

    becomes:

        белкокрады позорные
    """

    text = normalize_text(text)

    if ":" not in text:
        return text

    before, after = text.split(
        ":",
        1,
    )

    after = normalize_text(after)

    if contains_cyrillic(after):
        return after

    return text


def extract_russian(text):

    lines = []

    for raw in text.splitlines():

        line = normalize_text(raw)

        if not line:
            continue

        if not contains_cyrillic(line):
            continue

        line = strip_username(line)

        if not contains_cyrillic(line):
            continue

        # Avoid treating one random OCR character
        # as a Russian sentence.
        if cyrillic_count(line) < 2:
            continue

        if line not in lines:
            lines.append(line)

    return lines


# ============================================================
# OCR
# ============================================================

def tesseract_pass(image, description):

    text = pytesseract.image_to_string(
        image,
        lang="rus+eng",
        config=OCR_CONFIG,
    )

    print()
    print(
        f"========== OCR {description} =========="
    )

    print(text[:5000])

    print(
        "========================================"
    )

    return extract_russian(text)


def find_russian(image):

    # --------------------------------------------------------
    # PASS 1
    # Exact screenshot.
    # --------------------------------------------------------

    found = tesseract_pass(
        image,
        "ORIGINAL",
    )

    if found:
        return found


    # --------------------------------------------------------
    # PASS 2
    # Upscaled RGB.
    # --------------------------------------------------------

    large = image.resize(
        (
            image.width * 2,
            image.height * 2,
        ),
        Image.Resampling.LANCZOS,
    )

    found = tesseract_pass(
        large,
        "2X",
    )

    if found:
        return found


    # --------------------------------------------------------
    # PASS 3
    # Grayscale / autocontrast.
    # --------------------------------------------------------

    gray = ImageOps.autocontrast(
        large.convert("L")
    )

    gray = ImageEnhance.Sharpness(
        gray
    ).enhance(
        1.5
    )

    found = tesseract_pass(
        gray,
        "GRAYSCALE",
    )

    if found:
        return found


    # --------------------------------------------------------
    # PASS 4
    # Higher contrast.
    # --------------------------------------------------------

    contrast = ImageEnhance.Contrast(
        gray
    ).enhance(
        1.8
    )

    return tesseract_pass(
        contrast,
        "HIGH CONTRAST",
    )


# ============================================================
# MESSAGE HISTORY
# ============================================================

def is_new_message(text):

    if text in seen_set:
        return False

    if len(seen_messages) >= SEEN_MESSAGE_LIMIT:

        old = seen_messages.popleft()

        seen_set.discard(old)

    seen_messages.append(text)
    seen_set.add(text)

    return True


# ============================================================
# CAPTURE WORKER
# ============================================================

def capture_worker():

    screenshot_number = 0

    while not stop_event.is_set():

        started = time.monotonic()

        screenshot_number += 1

        print()
        print(
            "======================================"
        )

        print(
            f"SCREENSHOT {screenshot_number}"
        )

        print(
            "======================================"
        )

        try:

            ui_queue.put(
                (
                    "status",
                    (
                        f"Taking screenshot "
                        f"{screenshot_number}..."
                    ),
                )
            )


            # ================================================
            # GNOME / WAYLAND SCREENSHOT
            # ================================================

            image = capture_screen()


            print(
                "[capture] Captured:",
                image.size,
            )

            print(
                "[capture] Saved:",
                LATEST_SCREENSHOT,
            )


            # ================================================
            # OCR
            # ================================================

            russian_lines = find_russian(
                image
            )


            if not russian_lines:

                print()
                print(
                    "[OCR] No Russian detected."
                )

                # Do NOT clear old English.
                ui_queue.put(
                    (
                        "status",
                        (
                            "No new Russian detected. "
                            "Keeping previous translation."
                        ),
                    )
                )


            else:

                print()
                print("[OCR] Russian found:")

                for line in russian_lines:

                    print(
                        "  ",
                        repr(line),
                    )


                # ============================================
                # ONLY PROCESS NEW TEXT
                # ============================================

                new_lines = [
                    line
                    for line in russian_lines
                    if is_new_message(line)
                ]


                if not new_lines:

                    ui_queue.put(
                        (
                            "status",
                            (
                                "Russian found, but "
                                "nothing new."
                            ),
                        )
                    )


                else:

                    english_lines = []


                    for russian in new_lines:

                        print()
                        print(
                            "RU:",
                            russian,
                        )


                        try:

                            english = (
                                translate_ru_to_en(
                                    russian
                                )
                            )

                        except Exception as exc:

                            print(
                                "Translation error:",
                                repr(exc),
                            )

                            continue


                        if not english:
                            continue


                        print(
                            "EN:",
                            english,
                        )


                        english_lines.append(
                            english
                        )


                    # ========================================
                    # ONLY CHANGE DISPLAY WHEN WE HAVE
                    # A NEW SUCCESSFUL TRANSLATION.
                    # ========================================

                    if english_lines:

                        english_text = "\n".join(
                            english_lines
                        )


                        ui_queue.put(
                            (
                                "translation",
                                english_text,
                            )
                        )


                        ui_queue.put(
                            (
                                "status",
                                (
                                    f"Translated "
                                    f"{len(english_lines)} "
                                    f"new Russian line(s)."
                                ),
                            )
                        )


        except Exception as exc:

            print(
                "[capture/OCR error]",
                repr(exc),
            )

            # Again, preserve previous translation.
            ui_queue.put(
                (
                    "status",
                    (
                        "Capture/OCR failed; "
                        "keeping previous translation."
                    ),
                )
            )


        # ====================================================
        # WAIT UNTIL NEXT SCREENSHOT
        # ====================================================

        elapsed = (
            time.monotonic()
            - started
        )

        delay = max(
            0,
            CAPTURE_INTERVAL
            - elapsed,
        )


        print(
            f"[status] Next screenshot "
            f"in {delay:.1f}s"
        )


        stop_event.wait(
            delay
        )


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Russian → English"
)

root.geometry(
    "1050x250+100+100"
)

root.configure(
    bg="black"
)

root.attributes(
    "-topmost",
    True,
)


title_label = tk.Label(
    root,

    text="Russian → English",

    bg="black",

    fg="#999999",

    font=(
        "DejaVu Sans",
        14,
        "bold",
    ),
)

title_label.pack(
    pady=(10, 5)
)


# IMPORTANT:
#
# Display English only.
#
# Since we're OCRing the entire screen, displaying Russian
# ourselves would cause OCR to see our own window.
english_label = tk.Label(
    root,

    text="Waiting for Russian text...",

    bg="black",

    fg="white",

    font=(
        "DejaVu Sans",
        23,
        "bold",
    ),

    justify="left",

    anchor="nw",

    wraplength=1000,

    padx=20,

    pady=15,
)

english_label.pack(
    fill="both",
    expand=True,
)


status_label = tk.Label(
    root,

    text="Starting screen capture...",

    bg="black",

    fg="#888888",

    font=(
        "DejaVu Sans",
        10,
    ),
)

status_label.pack(
    pady=(0, 10)
)


# ============================================================
# UI QUEUE
# ============================================================

def process_ui():

    try:

        while True:

            kind, value = (
                ui_queue.get_nowait()
            )


            if kind == "translation":

                # Previous text remains until this happens.
                english_label.config(
                    text=value
                )


            elif kind == "status":

                status_label.config(
                    text=value
                )


    except queue.Empty:

        pass


    if not stop_event.is_set():

        root.after(
            100,
            process_ui,
        )


# ============================================================
# EXIT
# ============================================================

def shutdown(event=None):

    stop_event.set()

    try:
        root.destroy()
    except tk.TclError:
        pass


root.protocol(
    "WM_DELETE_WINDOW",
    shutdown,
)

root.bind(
    "<Escape>",
    shutdown,
)


# ============================================================
# START
# ============================================================

threading.Thread(
    target=capture_worker,
    daemon=True,
).start()

root.after(
    100,
    process_ui,
)

root.mainloop()

stop_event.set()
