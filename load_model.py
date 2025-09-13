"""
load_model.py
--------------
Standalone test to load a lightweight, instruction-tuned model for Q&A so the
RAG pipeline can run on CPU or modest GPUs.

Why Flan-T5?
- "google/flan-t5-base" is lightweight, instruction-tuned, and performs well
  for question answering and summarization tasks.
- Works on CPU; uses GPU automatically if available.
- You can swap to larger variants (flan-t5-large/xl) if hardware allows.

This script only validates model loading and a single generation pass.
Later, you can wrap the pipeline with LangChain's HuggingFacePipeline for RAG.
"""

from __future__ import annotations

import sys
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


DEFAULT_MODEL = "google/flan-t5-base"


def detect_device_for_pipeline() -> int:
    """Return pipeline device id: 0 for CUDA, -1 for CPU."""
    return 0 if torch.cuda.is_available() else -1


def load_qna_model(model_name: str = DEFAULT_MODEL):
    """Load tokenizer and model suitable for text2text generation (Q&A)."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def build_pipeline(tokenizer, model):
    device = detect_device_for_pipeline()
    print(f"Using device: {'cuda:0' if device == 0 else 'cpu'}")
    return pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=256,
        temperature=0.2,
    )


def main(custom_prompt: Optional[str] = None) -> None:
    prompt = custom_prompt or (
        "Question: What is the first-line treatment for type 2 diabetes? "
        "Context: Metformin is widely prescribed as a first-line treatment for type 2 diabetes."
    )

    tokenizer, model = load_qna_model(DEFAULT_MODEL)
    generate_pipe = build_pipeline(tokenizer, model)

    result = generate_pipe(prompt)[0]["generated_text"]
    print("\nModel output:\n", result)


if __name__ == "__main__":
    user_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    main(user_prompt)

