# AI Engineering Learning Lab

This repository is a minimal workspace for learning and practicing AI engineering without adding frameworks up front.

## Goal

Use this repo to:

- keep notes while studying AI concepts and tools
- build small projects and experiments incrementally
- store reusable prompts
- run simple scripts against local models such as Ollama

## Structure

- `notes/` for study notes, summaries, and references
- `projects/` for small hands-on builds and experiments
- `prompts/` for prompt templates and prompt iteration
- `scripts/` for utility scripts

## Local Ollama script

`scripts/ask_ollama.py` sends a prompt to a local Ollama server at `http://localhost:11434/api/generate` using the model `qwen3:14b-q4_K_M`.

Run it with:

```bash
python3 scripts/ask_ollama.py "Explain retrieval-augmented generation in two sentences."
```

If you run it without arguments, it uses a small default prompt.
