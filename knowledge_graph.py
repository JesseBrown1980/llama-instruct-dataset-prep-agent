import logging
import json
import os
from typing import Optional, Dict, List, Any, Tuple
from rdflib import Graph, URIRef, Literal, OWL, RDF, RDFS
import requests
from datetime import datetime
from downloader import download_and_clean_resource

logger = logging.getLogger(__name__)

def format_time(time_str: str) -> str:
    """Convert ISO time to format YYYYMMDDHHMMSS."""
    # Parse the ISO time string
    dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S%z")
    return dt.strftime("%Y%m%d%H%M%S")

def fetch_soli_metadata(url: str) -> Dict[str, Any]:
    """Fetch metadata from SOLI URL."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Successfully fetched SOLI metadata from {url}")
            return data
        logger.warning(f"Failed to fetch SOLI metadata from {url}: {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching SOLI metadata from {url}: {str(e)}")
    return {"class_not_found": True}

def extract_soli_key(uri: str) -> str:
    """Extract key from SOLI URI."""
    if not uri:
        return None
    parts = uri.split('/')
    if len(parts) > 0:
        key = parts[-1]
        if key.startswith('R') and len(key) == 23:  # SOLI keys are 23 chars long
            return key
    return None

def is_soli_url(url: str) -> bool:
    """Check if URL is a SOLI URL."""
    return url and "soli.openlegalstandard.org" in url

def is_owl_file(url: str) -> bool:
    """Check if URL points to an OWL file."""
    return url and (url.lower().endswith('.owl') or url.lower().endswith('.rdf'))

def get_about_uri(g: Graph, node: URIRef) -> str:
    """Get the node URI if it's a SOLI URL."""
    node_uri = str(node)
    if is_soli_url(node_uri):
        return node_uri
    return None

def process_urls_for_context(urls: List[str], cache_dir: str, current_time: str) -> Tuple[List[Dict], List[str]]:
    """Process URLs to download and clean content."""
    context = []
    failed_urls = []
    
    for url in urls:
        if not url:
            continue
        try:
            result = download_and_clean_resource(url, cache_dir=cache_dir)
            if result["downloaded"]:
                relative_path = os.path.relpath(result["local_path"], os.getcwd())
                context.append({
                    "url": url,
                    "local_path": relative_path,
                    "text": result["text"],
                    "time": format_time(current_time)
                })
            else:
                failed_urls.append(url)
        except Exception as e:
            logger.error(f"Failed to process URL {url}: {str(e)}")
            failed_urls.append(url)
    
    return context, failed_urls

def process_node(g: Graph, node: URIRef, cache_dir: str, current_time: str) -> Dict[str, Any]:
    """Process a node in the graph and return its data."""
    node_uri = str(node)
    if not is_soli_url(node_uri):
        logger.info(f"Skipping node {node_uri} - not a SOLI URL")
        return None
    
    # Extract SOLI key
    node_key = extract_soli_key(node_uri)
    if not node_key:
        logger.info(f"Skipping node {node_uri} - not a valid SOLI key")
        return None
    
    # Get English label
    english_label = None
    for s, p, o in g.triples((node, RDFS.label, None)):
        if isinstance(o, Literal) and (o.language is None or o.language == 'en'):
            english_label = str(o)
            break
    
    # Initialize node data
    node_data = {
        "key": node_key,
        "uri": node_uri,
        "timestamp": format_time(current_time),
        "context": [],
        "failed_urls": [],
        "parents": [],
        "children": [],
        "has_parent_in_kb": False,
        "languages": ["en"] if english_label else [],
        "deprecated": False,
        "country": None,
        "source": None,
        "labels": {"en": english_label} if english_label else {},
        "definitions": {},
        "class_not_found": False
    }
    
    # Fetch SOLI metadata
    soli_data = fetch_soli_metadata(node_uri)
    if soli_data:
        if "class_not_found" in soli_data:
            node_data["class_not_found"] = True
        else:
            # Update basic metadata
            node_data.update({
                "preferred_label": soli_data.get("preferred_label"),
                "alternative_labels": soli_data.get("alternative_labels", []),
                "definition": soli_data.get("definition"),
                "translations": soli_data.get("translations", {}),
                "deprecated": soli_data.get("deprecated", False),
                "country": soli_data.get("country"),
                "source": soli_data.get("source"),
                "notes": soli_data.get("notes", []),
                "examples": soli_data.get("examples", []),
                "history_note": soli_data.get("history_note"),
                "editorial_note": soli_data.get("editorial_note"),
                "in_scheme": soli_data.get("in_scheme"),
                "identifier": soli_data.get("identifier"),
                "description": soli_data.get("description")
            })
            
            # Add languages from translations
            if soli_data.get("translations"):
                for lang in soli_data["translations"].keys():
                    if lang not in node_data["languages"]:
                        node_data["languages"].append(lang)
            
            # Process parents from sub_class_of
            for parent in soli_data.get("sub_class_of", []):
                if is_soli_url(parent):
                    parent_key = extract_soli_key(parent)
                    if parent_key and parent_key not in node_data["parents"]:
                        node_data["parents"].append(parent_key)
            
            # Process children
            for child in soli_data.get("parent_class_of", []):
                if is_soli_url(child):
                    child_key = extract_soli_key(child)
                    if child_key and child_key not in node_data["children"]:
                        node_data["children"].append(child_key)
            
            # Process URLs for context
            urls_to_process = []
            if soli_data.get("is_defined_by"):
                urls_to_process.append(soli_data["is_defined_by"])
            urls_to_process.extend(soli_data.get("see_also", []))
            
            if urls_to_process:
                context, failed = process_urls_for_context(urls_to_process, cache_dir, current_time)
                node_data["context"].extend(context)
                node_data["failed_urls"].extend(failed)
    
    return node_data

def build_knowledge_graph(owl_file: str, output_file: str = "test-kb.json", limit: Optional[int] = None) -> None:
    """
    Builds and saves a knowledge graph from OWL file with rich metadata and context.
    If the output file exists, it will update the existing knowledge base.
    
    Args:
        owl_file: Path to OWL file
        output_file: Path to output JSON file
        limit: Optional int to limit number of nodes (for testing)
    """
    current_time = "2025-01-11T14:05:04-08:00"  # Use provided time
    logger.info(f"Building knowledge graph from {owl_file}")
    
    g = Graph()
    try:
        g.parse(owl_file)
        logger.info(f"Successfully parsed OWL file containing {len(g)} triples")
    except Exception as e:
        logger.error(f"Failed to parse OWL file: {str(e)}")
        raise

    # Create cache directory if needed
    cache_dir = "cache_refs"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Load or create knowledge base
    try:
        with open(output_file, 'r') as f:
            kb = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        kb = {}
    
    # Ensure basic structure exists
    kb.setdefault("nodes", {})
    kb.setdefault("refs", [])
    kb.setdefault("failed_urls", [])
    
    # Process AnnotationProperty nodes
    nodes = []
    for node in g.subjects(RDF.type, OWL.AnnotationProperty):
        # Check if node has a label
        has_label = False
        for _ in g.triples((node, RDFS.label, None)):
            has_label = True
            break
        if has_label and is_soli_url(str(node)):
            nodes.append((node, "annotation"))
    
    if limit:
        nodes = nodes[:limit]
    
    # Process each node
    for node, node_type in nodes:
        logger.info(f"Processing {node_type}: {node}")
        node_data = process_node(g, node, cache_dir, current_time)
        
        if node_data:
            node_data["type"] = node_type
            node_key = node_data["key"]
            
            # Check if node already exists and merge if needed
            existing_node = kb["nodes"].get(node_key, {})
            if existing_node:
                # Preserve existing context and failed_urls
                node_data["context"] = (
                    existing_node.get("context", []) + 
                    node_data.get("context", [])
                )
                node_data["failed_urls"] = list(set(
                    existing_node.get("failed_urls", []) + 
                    node_data.get("failed_urls", [])
                ))
            
            # Check if parents exist in KB
            for parent_key in node_data["parents"]:
                if parent_key in kb["nodes"]:
                    node_data["has_parent_in_kb"] = True
                    # Add this node as child to parent if not already there
                    parent_node = kb["nodes"][parent_key]
                    if node_key not in parent_node.get("children", []):
                        parent_node.setdefault("children", []).append(node_key)
            
            kb["nodes"][node_key] = node_data
            
            # Update global failed_urls list
            kb["failed_urls"].extend([
                url for url in node_data.get("failed_urls", [])
                if url not in kb["failed_urls"]
            ])
            
            # Update global refs list
            for ctx in node_data.get("context", []):
                url = ctx.get("url")
                if url and not any(r.get("url") == url for r in kb["refs"]):
                    kb["refs"].append({
                        "url": url,
                        "local_path": ctx.get("local_path"),
                        "time": ctx.get("time")
                    })
            
            # Write KB after each node is processed
            with open(output_file, 'w') as f:
                json.dump(kb, f, indent=2)
            
    logger.info(f"Knowledge base creation complete. Output saved to {output_file}")
