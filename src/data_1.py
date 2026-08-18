import os
import json
import hashlib
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from tqdm import tqdm
from openai import OpenAI
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

# CONFIG


INPUT_DIR = "C:\\Users\\rudsi\\Desktop\\Expo-2026\\cache\\pdfs"
OUTPUT_DIR = "C:\\Users\\rudsi\\Desktop\\Expo-2026\\data\\raw_literature"  

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen3:4b"

# THREAD SAFETY
llm_lock = Lock()


# RESUME SUPPORT
stream_path = os.path.join(OUTPUT_DIR, "stream.csv")
citation_path = os.path.join(OUTPUT_DIR, "citations.csv")
failed_path = os.path.join(OUTPUT_DIR, "failed.csv")

processed_ids = set()

if os.path.exists(stream_path):
    try:
        df_prev = pd.read_csv(stream_path)
        if "source_file" in df_prev.columns:
            processed_ids = set(df_prev["source_file"].dropna().unique())
    except:
        pass

# TEXT EXTRACTION
def extract_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:6]:  # speed cap
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


def extract_html(path):
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text("\n")


# CHUNKING
def chunk_text(text, size=8000):
    chunks = []
    cur = ""

    for line in text.split("\n"):
        if len(cur) + len(line) > size:
            chunks.append(cur)
            cur = line
        else:
            cur += "\n" + line

    if cur:
        chunks.append(cur)

    return chunks

# LLM
SYSTEM_PROMPT = """
Return ONLY valid JSON.

Schema:
{
 "records":[
  {
   "plant_species":"",
   "plant_part":"",
   "extraction_solvent":"",
   "extraction_method":"",
   "concentration_mg_per_ml":null,
   "crop_species":"",
   "crop_variety":"",
   "germination_percent":null,
   "root_length_mm":null,
   "shoot_length_mm":null,
   "total_phenolic_content_mg_gae_per_g":null,
   "compound_data":{},
   "incubation_temp_c":null,
   "incubation_days":null,
   "tds_mg_per_ml":null,
   "notes":"",
   "confidence":0.0
  }
 ]
}
"""

def llm_call(text):
    with llm_lock:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ]
        )
    return resp.choices[0].message.content

# SAFETY CLEANING 
def clean_record(r):
    # ensure compound_data is ALWAYS dict
    if not isinstance(r.get("compound_data"), dict):
        r["compound_data"] = {}

    return r
    
# FILE PROCESSOR
def process_file(file):
    path = os.path.join(INPUT_DIR, file)
    ext = file.split(".")[-1]

    file_id = hashlib.md5(file.encode()).hexdigest()[:8]

    try:
        if file in processed_ids:
            return file, [], file_id

    
        # TEXT EXTRACTION
        if ext == "pdf":
            text = extract_pdf_text(path)
        else:
            text = extract_html(path)

        if len(text.strip()) < 500:
            text += extract_pdf_text(path)

        chunks = chunk_text(text)

        results = []

        for c in chunks:
            raw = llm_call(c)

            try:
                data = json.loads(raw)
                records = data.get("records", [])

                for r in records:
                    r = clean_record(r)
                    r["compound_data"] = json.dumps(r["compound_data"]) 
                    results.append(r)

            except:
                try:
                    fixed = llm_call("Fix JSON ONLY:\n\n" + raw)
                    data = json.loads(fixed)
                    records = data.get("records", [])

                    for r in records:
                        r = clean_record(r)
                        r["compound_data"] = json.dumps(r["compound_data"])
                        results.append(r)

                except:
                    continue

        return file, results, file_id

    except Exception:
        return file, [], file_id

# MAIN ENGINE
def main():

    files = os.listdir(INPUT_DIR)

    results = []
    citations = []
    failed = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_file, f): f for f in files}

        for future in tqdm(as_completed(futures), total=len(futures)):

            file, recs, file_id = future.result()

            citations.append({
                "study_id": file_id,
                "source_file": file
            })

            if not recs:
                failed.append({"file": file})
                continue

            for r in recs:
                r["source_file"] = file
                r["study_id"] = file_id

                results.append(r)
                
                pd.DataFrame([r]).to_csv(
                    stream_path,
                    mode="a",
                    header=not os.path.exists(stream_path),
                    index=False
                )

    # FINAL OUTPUTS

    df = pd.DataFrame(results)

    df.to_csv(os.path.join(OUTPUT_DIR, "dataset.csv"), index=False)

    df.to_parquet(os.path.join(OUTPUT_DIR, "dataset.parquet"), index=False)

    pd.DataFrame(citations).to_csv(citation_path, index=False)
    pd.DataFrame(failed).to_csv(failed_path, index=False)

    print("DONE")

##############################################################################

if __name__ == "__main__":
    main()
