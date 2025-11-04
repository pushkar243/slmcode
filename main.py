# -------------------------
# SANITY: Generate 50 educational Q/A from PDFs (Colab cell)
# -------------------------
# Paste this into one Colab cell and run.
# Make sure: Runtime -> GPU (T4). Upload PDFs to /content/pdfs/ or change PDF_FOLDER.

# Install minimal required libs (skip if already installed)
!pip install -q transformers pypdf sentence-transformers accelerate

# ---------- Config ----------
PDF_FOLDER = "/content/pdfs"            # change if your PDFs are on Drive
DRIVE_SAVE_DIR = "/content/drive/MyDrive/mistral_pipeline"  # optional
QA_COUNT_TARGET = 50                   # you requested 50
PAIRS_PER_CHUNK = 1                    # produce 1 Q/A per chunk to control quality
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
GEN_MODEL = "google/gemma-2b-it"       # educational style prompt tuned below
DEVICE = "cuda"                        # ensure GPU runtime

# ---------- Mount Drive (optional) ----------
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, re, json, csv, time
from pathlib import Path
pdf_dir = Path(PDF_FOLDER)
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

# ---------- 1) Extract pages from PDFs ----------
from pypdf import PdfReader
pages = []   # each element: {file,page,text}
if not pdf_dir.exists():
    raise SystemExit(f"PDF folder {PDF_FOLDER} not found. Upload PDFs there or change PDF_FOLDER.")

for fp in sorted(pdf_dir.glob("*.pdf")):
    try:
        r = PdfReader(str(fp))
        for i, p in enumerate(r.pages):
            txt = p.extract_text() or ""
            txt = re.sub(r"\s+", " ", txt).strip()
            if len(txt) > 80:
                pages.append({"file": fp.name, "page": i+1, "text": txt})
    except Exception as e:
        print("Error reading", fp.name, ":", e)

print("Pages extracted:", len(pages))
if len(pages) == 0:
    raise SystemExit("No PDF text found. Are PDFs scanned images? Use OCR first.")

# ---------- 2) Chunk pages into manageable pieces ----------
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = []
for p in pages:
    parts = splitter.split_text(p["text"])
    for idx, part in enumerate(parts):
        chunks.append({"chunk_id": f"{p['file']}_p{p['page']}_c{idx}",
                       "file": p["file"], "page": p["page"], "text": part})
print("Chunks created:", len(chunks))

# ---------- 3) Load generator (Gemma 2B) ----------
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

generator = None
tokenizer_gen = None

def try_load_gen(model_name):
    try:
        t = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False, use_fast=True)
        m = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        return t, m
    except Exception as e:
        print("Load failed for", model_name, ":", e)
        return None, None

print("Attempting to load generator:", GEN_MODEL)
tokenizer_gen, model_gen = try_load_gen(GEN_MODEL)

if tokenizer_gen is None or model_gen is None:
    # fallback: use a very small HF model if gemma is not available
    fallback = "microsoft/phi-3-mini-4k-instruct"
    print("Falling back to", fallback)
    tokenizer_gen, model_gen = try_load_gen(fallback)
    if tokenizer_gen is None:
        raise SystemExit("No generator available. Upload a local GGUF or ensure HF access.")

gen_device = next(model_gen.parameters()).device
print("Generator loaded on", gen_device)

# ---------- 4) Educational-style prompt template ----------
PROMPT_TEMPLATE = """You are a helpful subject-matter expert. Read the text below and generate ONE clear, educational question followed by a concise, explanation-rich answer.
Only use facts present in the text. Do NOT add outside information. Use complete sentences and an explanatory tone (suitable for teaching a colleague). 

Text:
\"\"\"
{chunk}
\"\"\"

Return exactly in this format:

Q: <question>
A: <answer>
"""

# ---------- 5) Generate Q/A pairs (stop when we reach QA_COUNT_TARGET) ----------
raw_qas = []
processed = 0
# We'll iterate through chunks in order — stop after collecting QA_COUNT_TARGET pairs
for ch in chunks:
    if processed >= QA_COUNT_TARGET:
        break
    prompt = PROMPT_TEMPLATE.format(chunk=ch["text"])
    try:
        inputs = tokenizer_gen(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model_gen.device)
        gen = model_gen.generate(**inputs, max_new_tokens=256, do_sample=False)
        out_text = tokenizer_gen.decode(gen[0], skip_special_tokens=True)
    except Exception as e:
        print("Generation error for chunk", ch["chunk_id"], ":", e)
        continue

    # parse Q/A
    if "Q:" in out_text and "A:" in out_text:
        try:
            q = out_text.split("Q:", 1)[1].split("A:",1)[0].strip()
            a = out_text.split("A:",1)[1].strip()
            if len(q) > 3 and len(a) > 10:   # simple filters
                raw_qas.append({"question": q, "answer": a, "chunk_id": ch["chunk_id"], "file": ch["file"], "page": ch["page"]})
                processed += 1
        except Exception as e:
            # fallback naive parse
            lines = [ln.strip() for ln in out_text.splitlines() if ln.strip()]
            q=None; a=None
            for ln in lines:
                if ln.lower().startswith("q:"):
                    q = ln.split(":",1)[1].strip()
                elif ln.lower().startswith("a:"):
                    a = ln.split(":",1)[1].strip()
                    if q and a:
                        raw_qas.append({"question": q, "answer": a, "chunk_id": ch["chunk_id"], "file": ch["file"], "page": ch["page"]})
                        processed += 1
                        q=None; a=None
    else:
        # If output doesn't have Q/A markers, try heuristics
        text = out_text.strip()
        # find first question-like sentence (ends with '?') and following sentence(s)
        import re
        m = re.search(r'([A-Z][^?!.]{10,}\?)', text)
        if m:
            q = m.group(1).strip()
            rest = text[m.end():].strip()
            a = rest.split("\n",1)[0].strip() if rest else ""
            if q and a:
                raw_qas.append({"question": q, "answer": a, "chunk_id": ch["chunk_id"], "file": ch["file"], "page": ch["page"]})
                processed += 1

    # small throttle
    time.sleep(0.12)

print(f"Generated {len(raw_qas)} Q/A pairs (target was {QA_COUNT_TARGET})")

# ---------- 6) Persist qna_dataset.csv and copy to Drive ----------
OUT_CSV = "qna_dataset.csv"
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question","answer","chunk_id","file","page"])
    for r in raw_qas:
        writer.writerow([r["question"], r["answer"], r["chunk_id"], r["file"], r["page"]])
print("Saved:", OUT_CSV)
# copy to drive
shutil.copy(OUT_CSV, os.path.join(DRIVE_SAVE_DIR, OUT_CSV))
print("Copied to Drive:", os.path.join(DRIVE_SAVE_DIR, OUT_CSV))

# ---------- 7) Token-length sanity checks (using the generator tokenizer) ----------
from statistics import mean
q_lens = [len(tokenizer_gen.encode(r["question"])) for r in raw_qas]
a_lens = [len(tokenizer_gen.encode(r["answer"])) for r in raw_qas]
print("Questions: count", len(q_lens), "avg tokens", mean(q_lens) if q_lens else 0, "max", max(q_lens) if q_lens else 0)
print("Answers:   count", len(a_lens), "avg tokens", mean(a_lens) if a_lens else 0, "max", max(a_lens) if a_lens else 0)

# ---------- 8) Show 10 samples for manual QA ----------
import pandas as pd
df = pd.read_csv(OUT_CSV)
print("\n=== 10 sample Q/A (manual inspection) ===")
display(df.sample(min(10, len(df))).reset_index(drop=True))

# ---------- 9) Short automated checks ----------
# - ensure at least 10 Q/A were generated
# - basic heuristic: answer must contain at least one noun from chunk (quick check)
def quick_quality_check(df, chunks_map, min_pairs=10):
    issues=[]
    if len(df) < min_pairs:
        issues.append(f"Too few Q/A pairs: {len(df)} < {min_pairs}")
    # check small % of answers contain words from their source chunk
    ok_count=0; total=0
    for i, row in df.sample(min(20,len(df))).iterrows():
        total += 1
        src = chunks_map.get(row['chunk_id'],"")
        # look for overlap of 3+ characters words
        overlap = 0
        for tok in re.findall(r'\\w{4,}', row['answer'].lower()):
            if tok in src.lower():
                overlap += 1
            if overlap >= 1:
                ok_count += 1
                break
    if total>0:
        pct = ok_count/total
        if pct < 0.3:
            issues.append(f"Low contextual overlap in sample ({ok_count}/{total} ≈ {pct:.2f})")
    return issues

# build chunk map
chunk_map = {c['chunk_id']: c['text'] for c in chunks}
issues = quick_quality_check(df, chunk_map, min_pairs=10)
print("Quick quality check issues:", issues if issues else "None detected")

print("\nSanity step done. If samples look good, reply 'Dataset Ready ✅' and I will give the optimized training pipeline next.")
