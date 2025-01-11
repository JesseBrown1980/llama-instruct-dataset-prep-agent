import os
import json
import time
import requests
from urllib.parse import urlparse
from rdflib import Graph, RDF, RDFS, SKOS, OWL

# Optional PDF/HTML parsing. If missing, skip relevant steps.
try:
    import fitz  # PyMuPDF
    from bs4 import BeautifulSoup
    CAN_PARSE_PDF = True
    CAN_PARSE_HTML = True
except ImportError:
    CAN_PARSE_PDF = False
    CAN_PARSE_HTML = False

############################################
# Agents invoking local Ollama with llama3.1
############################################

def ollama_invoke(prompt, model="llama3.1", temperature=0.0):
    url = "http://localhost:11434/complete"  
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "model": model, "temperature": temperature}
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()["completion"].strip()

def cleaner_agent(raw_text, context=""):
    """
    Splits raw_text into ~4000-word chunks if needed.
    Each chunk is posted to Ollama for summarization/cleaning.
    Only relevant info is returned. No pleasantries.
    """
    words = raw_text.split()
    cleaned_result = []
    chunk_size = 4000 - len(context.split())  # ensure total under 4000
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        prompt = f"""
You are a cleaning agent. Keep only essential information relevant to this context:
{context}
Raw text:
{chunk}
Return cleaned text. No pleasantries.
"""
        partial_cleaned = ollama_invoke(prompt, model="llama3.1", temperature=0.0)
        cleaned_result.append(partial_cleaned)
        start = end
    return "\n".join(cleaned_result).strip()

def maker_agent(question, knowledge_context):
    """
    Produces a concise answer based on knowledge_context.
    """
    prompt = f"""
You create a concise answer. 
Context:
{knowledge_context}
Question:
{question}
No pleasantries.
"""
    return ollama_invoke(prompt, model="llama3.1", temperature=0.2)

def formatter_agent(raw_answer, question, context):
    """
    Converts raw_answer into Dolly/LLAMA style JSON (instruction/context/response).
    No extraneous text. Only a JSON array with one object.
    """
    prompt = f"""
Create Dolly JSON. Array of one object with "instruction", "context", "response". 
Instruction: "{question}"
Context: "{context}"
Response: "{raw_answer}"
No extraneous keys. No pleasantries. Only JSON.
"""
    return ollama_invoke(prompt, model="llama3.1", temperature=0.0)

def checker_agent(draft_json):
    """
    Checks logical consistency and returns correct JSON if needed.
    """
    prompt = f"""
You are a logical checker. Input is JSON. Check correctness. If errors, fix them.
Return JSON. No pleasantries.
JSON:
{draft_json}
"""
    return ollama_invoke(prompt, model="llama3.1", temperature=0.0)

############################################
# Downloading and parsing references
############################################

def parse_pdf_to_text(pdf_path):
    if not CAN_PARSE_PDF:
        return ""
    text = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)
    except:
        return ""

def parse_html_to_text(html_bytes):
    if not CAN_PARSE_HTML:
        return ""
    try:
        soup = BeautifulSoup(html_bytes, "html.parser")
        return soup.get_text(separator="\n")
    except:
        return ""

def download_and_clean_resource(url, parent_context="", cache_dir="cache_refs"):
    """
    Downloads resource if not cached. If RDF/OWL/TTL, parse later as RDF.
    If PDF/HTML, parse to text, pass to cleaner_agent with parent_context.
    Returns a dict with { "url":..., "text":..., "time":..., ... }
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    parsed = urlparse(url)
    safe_fname = parsed.netloc + parsed.path.replace("/", "_")
    local_path = os.path.join(cache_dir, safe_fname)

    if os.path.exists(local_path):
        # Already downloaded
        pass
    else:
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            time.sleep(1)
        except:
            return {"url": url, "text": "", "downloaded": False}

    # Check content type or extension
    extension = os.path.splitext(local_path)[1].lower()
    meta_time = time.ctime(os.path.getmtime(local_path))

    # If RDF or OWL or TTL, handle in main parse later. Here just store empty text.
    if extension in [".owl", ".rdf", ".ttl"]:
        return {"url": url, "text": "", "time": meta_time, "downloaded": True}

    # If PDF
    if extension == ".pdf":
        raw_text = parse_pdf_to_text(local_path)
        clean_text = cleaner_agent(raw_text, parent_context)
        return {"url": url, "text": clean_text, "time": meta_time, "downloaded": True}

    # If HTML
    content_type = ""
    try:
        # Attempt to get from requests head. If missing, guess
        content_type = requests.head(url).headers.get("Content-Type", "")
    except:
        pass

    if "html" in content_type or extension in [".htm", ".html"]:
        with open(local_path, "rb") as f:
            html_bytes = f.read()
        raw_text = parse_html_to_text(html_bytes)
        clean_text = cleaner_agent(raw_text, parent_context)
        return {"url": url, "text": clean_text, "time": meta_time, "downloaded": True}

    # Fallback: treat as plain text
    with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()
    clean_text = cleaner_agent(raw_text, parent_context)
    return {"url": url, "text": clean_text, "time": meta_time, "downloaded": True}

############################################
# OWL parsing and knowledge graph building
############################################

def build_knowledge_graph(owl_file):
    g = Graph()
    g.parse(owl_file)
    graph_data = {
        "classes": {},
        "properties": {},
        "relationships": [],
        "refs": []
    }

    # Collect references from isDefinedBy or imports
    refs = set()
    for s, p, o in g:
        if p in [RDFS.isDefinedBy, OWL.imports] and isinstance(o, str):
            refs.add(o)

    # Basic class extraction
    for c in g.subjects(RDF.type, OWL.Class):
        label = list(g.objects(c, RDFS.label))
        definition = list(g.objects(c, SKOS.definition))
        label_str = label[0] if label else str(c)
        def_str = definition[0] if definition else ""
        graph_data["classes"][str(c)] = {
            "uri": str(c),
            "label": str(label_str),
            "definition": str(def_str),
        }

    # Basic object property extraction
    for op in g.subjects(RDF.type, OWL.ObjectProperty):
        label = list(g.objects(op, RDFS.label))
        definition = list(g.objects(op, SKOS.definition))
        label_str = label[0] if label else str(op)
        def_str = definition[0] if definition else ""
        graph_data["properties"][str(op)] = {
            "uri": str(op),
            "label": str(label_str),
            "definition": str(def_str),
        }

    # Relationship extraction
    for s, p, o in g:
        if isinstance(s, str) or isinstance(o, str):
            continue
        if isinstance(s, (type(RDF.type))) or isinstance(o, (type(RDF.type))):
            continue
        graph_data["relationships"].append({
            "subject": str(s),
            "predicate": str(p),
            "object": str(o)
        })

    graph_data["refs"] = list(refs)
    return graph_data

############################################
# Question Generation
############################################

def generate_questions(label, definition):
    # Minimal set
    questions = [
        f"What is {label}?",
        f"How is {label} used?",
    ]
    if definition:
        questions.append(f"Explain the definition of {label}")
    return questions

############################################
# Main Pipeline
############################################

def process_owl_file_full(owl_path, limit=5, output_kg="knowledge_graph.json", output_ds="dolly_dataset.json"):
    # 1) Build local knowledge graph
    kg = build_knowledge_graph(owl_path)

    # 2) Scrape and download references
    clean_refs = []
    parent_context = "Relevant to the ontology references"
    for ref in kg["refs"]:
        info = download_and_clean_resource(ref, parent_context)
        clean_refs.append(info)
    kg["downloadedRefs"] = clean_refs

    # 3) Store knowledge graph to JSON
    with open(output_kg, "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)

    # 4) Create Dolly dataset
    dataset = []
    # We'll combine classes + properties and limit total
    items = list(kg["classes"].values()) + list(kg["properties"].values())
    items = items[:limit]

    for item in items:
        label = item["label"]
        definition = item["definition"]
        q_list = generate_questions(label, definition)

        # Summarize knowledge context: label + definition + possible downloadedRefs
        # For brevity, just use definition. Could add ref text if relevant.
        knowledge_context = definition

        for q in q_list:
            # Maker agent
            raw_answer = maker_agent(q, knowledge_context)

            # Format to Dolly JSON
            draft_json = formatter_agent(raw_answer, q, knowledge_context)

            # Checker agent
            final_json = checker_agent(draft_json)
            # Attempt parse
            try:
                final_data = json.loads(final_json)
                # If it's a single object, wrap in list, or if it's already array, just extend
                if isinstance(final_data, dict):
                    dataset.append(final_data)
                elif isinstance(final_data, list):
                    dataset.extend(final_data)
            except:
                # If it fails, skip
                pass

    # 5) Write dataset
    with open(output_ds, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

############################################
# Entry
############################################

if __name__ == "__main__":
    owl_input = "SOLI/SOLI.owl"  # adjust
    process_owl_file_full(owl_input, limit=5)
