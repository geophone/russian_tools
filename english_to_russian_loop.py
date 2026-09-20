#!/usr/bin/env python3

import torch
from transformers import MarianMTModel, MarianTokenizer


MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"


print("Loading English -> Russian translator...")

tokenizer = MarianTokenizer.from_pretrained(
    MODEL_NAME
)

model = MarianMTModel.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Ready.")
print("Paste English text and press Enter.")
print("Type 'quit' to exit.\n")


def translate_en_to_ru(text: str) -> str:
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
        english = input("EN: ").strip()

        if not english:
            continue

        if english.lower() in {
            "quit",
            "exit",
            "q",
        }:
            break

        russian = translate_en_to_ru(
            english
        )

        print(f"RU: {russian}\n")

    except KeyboardInterrupt:
        print("\nExiting.")
        break
