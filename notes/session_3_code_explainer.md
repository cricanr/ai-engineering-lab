# Session 3 — Code Explainer

## Goal

Build a small local AI tool that reads a source file and asks a local Ollama model to explain it.

## What this demonstrates

- LLMs can be used as local developer tools.
- File content becomes prompt context.
- The prompt acts as an output contract.
- A structured response is easier to review than free-form text.

## Key pattern

```text
source file -> prompt context -> local model -> structured explanation