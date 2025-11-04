# 📚 Domain-Specific Q&A Model (Mistral 7B + QLoRA + Gradio)

This project demonstrates how to build a **domain-specific question-answering model** by:
1. Extracting text from multiple PDFs
2. Automatically generating a **Q&A training dataset**
3. Fine-tuning **Mistral 7B** with **QLoRA** on a **Google Colab T4 GPU**
4. Saving and resuming model checkpoints in Google Drive
5. Deploying a **Gradio chat interface** that displays both answers and supporting context
6. (Optional) Converting the final model to **GGUF** for offline use with llama.cpp / LM Studio / Ollama

---

## ✅ Why This Approach Works Better Than RAG
Traditional RAG pipelines depend heavily on:
- Embeddings relevance
- Chunk sizes
- Vector search retrieval quality

This can lead to hallucination if the retrieval step fails.

By contrast, **fine-tuning the model** on curated Q&A pairs:
- Reduces hallucination
- Enables *true domain recall*
- Makes responses **faster**
- Removes dependence on external APIs

Result: **The model actually *knows* the domain**, instead of just being assisted by a database.

---

## 🧱 Project Structure

