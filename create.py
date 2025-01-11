import os
import json
import time
import logging
import requests
from urllib.parse import urlparse
from rdflib import Graph, RDF, RDFS, SKOS, OWL

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('owl_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Optional PDF/HTML parsing. If missing, skip relevant steps.
try:
    import fitz  # PyMuPDF
    from bs4 import BeautifulSoup
    CAN_PARSE_PDF = True
    CAN_PARSE_HTML = True
    logger.info("PDF and HTML parsing capabilities loaded successfully")
except ImportError:
    CAN_PARSE_PDF = False
    CAN_PARSE_HTML = False
    logger.warning("PDF and/or HTML parsing libraries not available. Some features will be disabled")


############################################
# Agents invoking local Ollama with llama3.1
############################################

def ollama_invoke(prompt, model="llama3.1", temperature=0.0):
    """
    Invoke local Ollama endpoint with prompt. Adjust URL and model name as needed.
    """
    logger.debug(f"Invoking Ollama with model={model}, temperature={temperature}")
    url = "http://localhost:11434/complete"  # Adjust port/URL for your Ollama instance
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "model": model,
        "temperature": temperature
    }
    try:
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()["completion"].strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error invoking Ollama: {str(e)}")
        raise


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


def questioner_agent_invoke(knowledge_context):
    """
    Generates as many interesting legal questions as possible (no redundancy).
    Must return a JSON array of strings (the questions).
    """
    prompt = f"""
You are a question generator agent. Given this knowledge context, generate as many interesting legal questions as you can. 
Return a JSON array of question strings. No pleasantries, no repetition.

Context:
{knowledge_context}
Output only JSON.
"""
    return ollama_invoke(prompt, model="llama3.1", temperature=0.2)


def question_checker_agent_invoke(question_list_json):
    """
    Checks a JSON array of questions for redundancy or errors.
    Returns corrected JSON array if needed. No pleasantries.
    """
    prompt = f"""
You are a logical checker. The input is a JSON array of questions. Check for redundancy or errors. Return a corrected JSON array. 
No pleasantries. Only valid JSON.

Questions:
{question_list_json}
"""
    return ollama_invoke(prompt, model="llama3.1", temperature=0.0)


def maker_agent(question, knowledge_context):
    """
    Produces a concise answer based on knowledge_context for a single question.
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
    Converts raw_answer into Dolly/LLAMA style JSON:
    [
      {
        "instruction": ...,
        "context": ...,
        "response": ...
      }
    ]
    No extraneous text. Only JSON.
    """
    prompt = f"""
Create Dolly JSON. Array with one object. 
Keys: "instruction", "context", "response". 
Instruction: "{question}"
Context: "{context}"
Response: "{raw_answer}"
No extraneous keys. No pleasantries. Only JSON.
"""
    return ollama_invoke(prompt, model="llama3.1", temperature=0.0)


def checker_agent(draft_json):
    """
    Checks logical consistency of the final Dolly JSON and returns valid JSON if needed.
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
    If PDF/HTML, parse to text, then pass to cleaner_agent with parent_context.
    Returns dict: { "url":..., "text":..., "time":..., "downloaded":... }
    """
    logger.info(f"Processing resource: {url}")
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, urlparse(url).netloc + "_" + str(hash(url)) + ".json")
    
    if os.path.exists(cache_file):
        logger.debug(f"Using cached version of {url}")
        with open(cache_file) as f:
            return json.load(f)
            
    logger.info(f"Downloading resource from {url}")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    parsed = urlparse(url)
    safe_fname = parsed.netloc + parsed.path.replace("/", "_")
    local_path = os.path.join(cache_dir, safe_fname)

    if not os.path.exists(local_path):
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            time.sleep(1)
        except:
            return {"url": url, "text": "", "downloaded": False, "time": ""}

    extension = os.path.splitext(local_path)[1].lower()
    meta_time = time.ctime(os.path.getmtime(local_path))

    # If RDF or OWL or TTL, we don't parse text here
    if extension in [".owl", ".rdf", ".ttl"]:
        return {"url": url, "text": "", "time": meta_time, "downloaded": True}

    # PDF
    if extension == ".pdf":
        raw_text = parse_pdf_to_text(local_path)
        clean_text = cleaner_agent(raw_text, parent_context)
        return {"url": url, "text": clean_text, "time": meta_time, "downloaded": True}

    # Attempt to see if HTML
    content_type = ""
    try:
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
    """
    Reads the OWL file into an RDFLib Graph, extracts basic info:
      - classes
      - properties
      - relationships
      - references (OWL imports, RDFS.isDefinedBy)
    Returns a dict.
    """
    logger.info(f"Building knowledge graph from {owl_file}")
    g = Graph()
    try:
        g.parse(owl_file)
        logger.info(f"Successfully parsed OWL file containing {len(g)} triples")
    except Exception as e:
        logger.error(f"Failed to parse OWL file: {str(e)}")
        raise

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
        # skip if s or o are strings (sometimes weird data)
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
# Main Pipeline
############################################

def process_owl_file_full(owl_path, limit=5,
                         output_kg="knowledge_graph.json",
                         output_ds="dolly_dataset.json"):
    """
    Full pipeline:
      1) Build local knowledge graph from owl_path
      2) Download references
      3) Save knowledge graph to JSON
      4) For each (class or property) up to limit:
         - Combine label+definition (and possibly ref texts) as knowledge context
         - Use questioner_agent + question_checker_agent to generate final Q list
         - For each Q: call maker_agent -> formatter_agent -> checker_agent -> store
      5) Save final Dolly-format Q&A dataset
    """
    logger.info(f"Starting full OWL processing pipeline for {owl_path}")
    logger.info(f"Parameters: limit={limit}, output_kg={output_kg}, output_ds={output_ds}")
    
    try:
        # Step 1: Build knowledge graph
        logger.info("Step 1: Building knowledge graph")
        kg = build_knowledge_graph(owl_path)
        
        # Step 2: Download references
        logger.info("Step 2: Downloading references")
        clean_refs = []
        for ref in kg["refs"]:
            # Convert URIRef to string
            ref_url = str(ref)
            try:
                ref_data = download_and_clean_resource(ref_url)
                clean_refs.append(ref_data)
                logger.info(f"Successfully downloaded reference: {ref_url}")
            except Exception as e:
                logger.error(f"Failed to download reference {ref_url}: {str(e)}")
        
        kg["downloadedRefs"] = clean_refs
        
        # Step 3: Save knowledge graph
        logger.info(f"Step 3: Saving knowledge graph to {output_kg}")
        with open(output_kg, "w", encoding="utf-8") as f:
            json.dump(kg, f, indent=2, ensure_ascii=False)
            
        # Step 4: Generate Q&A pairs
        logger.info("Step 4: Generating Q&A pairs")
        dataset = []
        processed = 0
        
        # Combine classes and properties
        items = list(kg["classes"].values()) + list(kg["properties"].values())
        items = items[:limit]
        
        for item in items:
            logger.info(f"Processing item {processed + 1}/{limit}: {item.get('label', 'Unknown')}")
            label = item["label"]
            definition = item["definition"]

            # Build knowledge context (label + definition)
            knowledge_context = f"{label}. {definition}"

            # a) Use questioner agent
            questioner_prompt_result = questioner_agent_invoke(knowledge_context)
            # b) Check with question_checker
            question_checker_result = question_checker_agent_invoke(questioner_prompt_result)

            # Parse final question list
            try:
                final_questions = json.loads(question_checker_result)
                if not isinstance(final_questions, list):
                    final_questions = []
            except:
                logger.warning(f"Failed to parse questions for {label}")
                final_questions = []

            # c) For each question, call maker -> formatter -> checker
            for q in final_questions:
                raw_answer = maker_agent(q, knowledge_context)
                draft_json = formatter_agent(raw_answer, q, knowledge_context)
                final_json = checker_agent(draft_json)
                # parse and add to dataset
                try:
                    final_data = json.loads(final_json)
                    if isinstance(final_data, dict):
                        dataset.append(final_data)
                    elif isinstance(final_data, list):
                        dataset.extend(final_data)
                except:
                    logger.warning(f"Failed to parse final JSON for question: {q}")
                
            processed += 1
            
        # Step 5: Save dataset
        logger.info(f"Step 5: Saving dataset to {output_ds}")
        with open(output_ds, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
            
        logger.info("Pipeline completed successfully")
        return dataset
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


############################################
# Entry Point
############################################

if __name__ == "__main__":
    owl_input = "SOLI/SOLI.owl"  # Adjust as needed
    process_owl_file_full(owl_input, limit=5)
