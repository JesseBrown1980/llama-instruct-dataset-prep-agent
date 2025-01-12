# downloader.py

import os
import time
import json
import logging
import requests
from urllib.parse import urlparse
import re
from agents import cleaner_agent
from datetime import datetime

try:
    import fitz  # PyMuPDF
    from bs4 import BeautifulSoup
    CAN_PARSE_PDF = True
    CAN_PARSE_HTML = True
except ImportError:
    CAN_PARSE_PDF = False
    CAN_PARSE_HTML = False

logger = logging.getLogger(__name__)

def parse_pdf_to_text(pdf_path):
    if not CAN_PARSE_PDF:
        return ""
    import fitz
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
    from bs4 import BeautifulSoup
    try:
        soup = BeautifulSoup(html_bytes, "html.parser")
        return soup.get_text(separator="\n")
    except:
        return ""

def write_failed_url(url: str, error: str):
    """Write failed URL and error message to a file."""
    failed_file = "failed_downloads.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(failed_file, "a") as f:
        f.write(f"{timestamp} | {url} | Error: {error}\n")

def download_and_clean_resource(url, parent_context="", cache_dir="cache_refs"):
    """
    Downloads resource if not cached. If RDF/OWL/TTL, parse later as RDF.
    If PDF/HTML/JSON, parse to text, then pass to cleaner_agent with parent_context.
    Returns dict: { "url":..., "text":..., "time":..., "downloaded":... }
    """
    logger.info(f"Processing resource: {url}")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Use hash of full URL as filename to avoid any path issues
    url_hash = str(hash(url))
    local_path = os.path.join(cache_dir, f"resource_{url_hash}")
    cache_file = os.path.join(cache_dir, f"cache_{url_hash}.json")
    
    # Check if cached result exists
    if os.path.exists(cache_file):
        logger.debug(f"Using cached version of {url}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    logger.info(f"Downloading resource from {url}")

    # First try to get content type
    content_type = ""
    try:
        content_type = requests.head(url).headers.get("Content-Type", "").lower()
    except:
        pass

    # Download if not exists
    if not os.path.exists(local_path):
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            time.sleep(1)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to download {url}: {error_msg}")
            write_failed_url(url, error_msg)
            return {"url": url, "text": "", "downloaded": False, "time": datetime.now().isoformat()}

    meta_time = time.ctime(os.path.getmtime(local_path))
    extension = os.path.splitext(local_path)[1].lower()

    # If RDF or OWL or TTL, we don't parse text here
    if extension in [".owl", ".rdf", ".ttl"] or "rdf" in content_type:
        result = {"url": url, "text": "", "time": meta_time, "downloaded": True}
        _save_json(result, cache_file)
        return result

    # Handle JSON content
    if "json" in content_type or extension == ".json":
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                raw_text = json.dumps(json.load(f), indent=2)
            clean_text = cleaner_agent(raw_text, parent_context)
            result = {"url": url, "text": clean_text, "time": meta_time, "downloaded": True}
            _save_json(result, cache_file)
            return result
        except Exception as e:
            logger.error(f"Failed to process JSON from {url}: {str(e)}")

    # PDF
    if "pdf" in content_type or extension == ".pdf":
        raw_text = parse_pdf_to_text(local_path)
        try:
            clean_text = cleaner_agent(raw_text, parent_context)
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {str(e)}")
            clean_text = raw_text  # Use raw text if cleaning fails
        result = {"url": url, "text": clean_text, "time": meta_time, "downloaded": True}
        _save_json(result, cache_file)
        return result

    # HTML
    if "html" in content_type or extension in [".htm", ".html"]:
        with open(local_path, "rb") as f:
            html_bytes = f.read()
        raw_text = parse_html_to_text(html_bytes)
        try:
            clean_text = cleaner_agent(raw_text, parent_context)
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {str(e)}")
            clean_text = raw_text  # Use raw text if cleaning fails
        result = {"url": url, "text": clean_text, "time": meta_time, "downloaded": True}
        _save_json(result, cache_file)
        return result

    # Fallback: treat as plain text
    with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()
    try:
        clean_text = cleaner_agent(raw_text, parent_context)
    except Exception as e:
        logger.error(f"Failed to process URL {url}: {str(e)}")
        clean_text = raw_text  # Use raw text if cleaning fails
    result = {"url": url, "text": clean_text, "time": meta_time, "downloaded": True}
    _save_json(result, cache_file)
    return result

def _save_json(data, path):
    """Helper to save JSON data with indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
