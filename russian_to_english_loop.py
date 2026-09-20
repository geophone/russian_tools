#!/usr/bin/env python3

import torch
from transformers import MarianMTModel, MarianTokenizer


MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"


print("Loading Russian -> English translator...")

tokenizer = MarianTokenizer.from_pretrained(
    MODEL_NAME
)

model = MarianMTModel.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Ready.")
print("Paste Russian text and press Enter.")
print("Type 'quit' to exit.\n")


def translate_ru_to_en(text: str) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():
        translated = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4,
        )

    return tokenizer.decode(
        translated[0],
        skip_special_tokens=True,
    )


while True:
    try:
        russian = input("RU: ").strip()

        if not russian:
            continue

        if russian.lower() in {
            "quit",
            "exit",
            "q",
        }:
            break

        english = translate_ru_to_en(
            russian
        )

        print(f"EN: {english}\n")

    except KeyboardInterrupt:
        print("\nExiting.")
        break
