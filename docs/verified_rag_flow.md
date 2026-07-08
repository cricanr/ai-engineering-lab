# Verified RAG Flow

This note explains the verification flow used in:

```text
scripts/tiny_rag_hybrid_smart_verified.py
```

The main idea:

```text
generate grounded answer
  -> verify whether the answer is supported by cited chunks
```

---

## Big picture

```text
Question
   |
   v
Retrieve relevant chunks
   |
   v
Assign source IDs
S1, S2, S3, ...
   |
   v
Qwen answers using only retrieved chunks
   |
   v
Grounded JSON answer
   |
   v
Verifier prompt is built from:
- original question
- generated answer
- cited source chunks only
   |
   v
Qwen verifies the answer
   |
   v
Verification JSON:
supported / partially_supported / unsupported
```

---

## Full flow diagram

```text
                         ┌────────────────────┐
                         │   User question    │
                         └─────────┬──────────┘
                                   │
                                   v
                         ┌────────────────────┐
                         │ Retrieval pipeline │
                         │ smart hybrid RAG   │
                         └─────────┬──────────┘
                                   │
                                   v
                         ┌────────────────────┐
                         │ Retrieved chunks   │
                         │ top-k context      │
                         └─────────┬──────────┘
                                   │
                                   v
                         ┌────────────────────┐
                         │ Assign source IDs  │
                         │ S1, S2, S3, ...    │
                         └─────────┬──────────┘
                                   │
                                   v
                ┌────────────────────────────────────┐
                │ build_grounded_prompt()            │
                │                                    │
                │ Gives Qwen:                         │
                │ - question                          │
                │ - retrieved chunks                  │
                │ - source IDs                        │
                │ - context-only rules                │
                └──────────────────┬─────────────────┘
                                   │
                                   v
                ┌────────────────────────────────────┐
                │ Qwen call #1                        │
                │ role: answer writer                 │
                │                                    │
                │ Must return grounded answer JSON:   │
                │ - answer                            │
                │ - insufficient_context              │
                │ - confidence                        │
                │ - cited_source_ids                  │
                │ - missing_information               │
                └──────────────────┬─────────────────┘
                                   │
                                   v
                ┌────────────────────────────────────┐
                │ normalize_answer_json()             │
                │                                    │
                │ Python checks/cleans answer shape.  │
                └──────────────────┬─────────────────┘
                                   │
                                   v
                ┌────────────────────────────────────┐
                │ print_grounded_answer()             │
                │                                    │
                │ Python maps S1/S2/S3 back to:       │
                │ - file path                         │
                │ - line range                        │
                │ - chunk kind/title                  │
                └──────────────────┬─────────────────┘
                                   │
                                   v
                ┌────────────────────────────────────┐
                │ build_verification_prompt()         │
                │                                    │
                │ Gives verifier only:                │
                │ - original question                 │
                │ - generated answer                  │
                │ - cited chunks only                 │
                │                                    │
                │ Important:                          │
                │ The verifier does not get all       │
                │ retrieved chunks, only cited ones.  │
                └──────────────────┬─────────────────┘
                                   │
                                   v
                ┌────────────────────────────────────┐
                │ Qwen call #2                        │
                │ role: strict verifier               │
                │                                    │
                │ Must check:                         │
                │ Is the answer supported by the      │
                │ cited context only?                 │
                └──────────────────┬─────────────────┘
                                   │
                                   v
                ┌────────────────────────────────────┐
                │ normalize_verification_json()       │
                │                                    │
                │ Python checks/cleans verifier shape.│
                └──────────────────┬─────────────────┘
                                   │
                                   v
                ┌────────────────────────────────────┐
                │ print_verification_result()         │
                │                                    │
                │ Shows:                              │
                │ - supported                         │
                │ - partially_supported               │
                │ - unsupported                       │
                │ - unsupported claims                │
                │ - recommended fix                   │
                └────────────────────────────────────┘
```

---

## Two Qwen calls

The same local chat model can be used twice, but with different jobs.

```text
Qwen call #1
  role: answer writer
  input: question + retrieved chunks
  output: grounded answer JSON

Qwen call #2
  role: verifier
  input: question + answer + cited chunks only
  output: verification JSON
```

Default model:

```text
qwen3:14b-q4_K_M
```

---

## What the first Qwen call returns

The answer writer returns:

```json
{
  "answer": "direct answer, or explanation that context is insufficient",
  "insufficient_context": false,
  "confidence": "low|medium|high",
  "cited_source_ids": ["S1"],
  "missing_information": []
}
```

This answers the user question and cites source IDs.

---

## What the second Qwen call returns

The verifier returns:

```json
{
  "support_status": "supported|partially_supported|unsupported",
  "verdict": "short human-readable verdict",
  "unsupported_claims": [],
  "source_ids_checked": ["S1"],
  "recommended_fix": "empty string if no fix needed"
}
```

This does not answer the original question again.

It only checks whether the first answer is supported by the cited chunks.

---

## Why verification is useful

Grounding asks:

```text
Which chunks support your answer?
```

Verification asks:

```text
Do those chunks actually support the answer?
```

That catches cases where the model cites a real source but says more than the source supports.

Example:

```text
Source S1 says:
The script caches embeddings in docs/.rag_cache.

Answer says:
The script stores embeddings in PostgreSQL and docs/.rag_cache.

Verifier should say:
partially_supported

Unsupported claim:
The cited source does not support the PostgreSQL claim.
```

---

## Important limitation

Verification is still done by an LLM.

So it is not a mathematical guarantee.

It is a stronger safety pattern:

```text
generation + evidence + second-pass review
```

But the verifier can still make mistakes.

In a production system, you might add:

```text
- different verifier model
- deterministic source-ID validation
- claim extraction
- human review for high-risk answers
- automated tests/evals
```

---

## Mental model

```text
Grounded RAG:
"Answer using these chunks and cite them."

Verified RAG:
"Now check whether the cited chunks really support that answer."
```

---

## One-line summary

```text
Question -> retrieve chunks -> answer with citations -> verify answer against cited chunks -> show support status
```
