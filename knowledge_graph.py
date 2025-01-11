import logging
import json
import os
from typing import Optional, Dict, List, Any, Tuple
from rdflib import Graph, URIRef, Literal, OWL, RDF, RDFS
import requests
from datetime import datetime
from downloader import download_and_clean_resource
from class_counter import increment_count

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
        "class_not_found": False,
        "downloadedRefs": []
    }
    
    # First try to download and process the about URL itself
    try:
        about_context, about_failed = process_urls_for_context([node_uri], cache_dir, current_time)
        if about_context:
            node_data["context"].extend(about_context)
            for ctx in about_context:
                if ctx["local_path"] not in node_data["downloadedRefs"]:
                    node_data["downloadedRefs"].append(ctx["local_path"])
        if about_failed:
            for url in about_failed:
                if url not in node_data["failed_urls"]:
                    node_data["failed_urls"].append(url)
    except Exception as e:
        logger.error(f"Failed to process about URL {node_uri}: {str(e)}")
        if node_uri not in node_data["failed_urls"]:
            node_data["failed_urls"].append(node_uri)
    
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
            
            # Add definition to context if available
            if soli_data.get("definition"):
                node_data["context"].append({
                    "url": node_uri,
                    "text": soli_data["definition"],
                    "type": "definition",
                    "time": format_time(current_time)
                })
            
            # Add languages from translations
            if soli_data.get("translations"):
                for lang in soli_data["translations"].keys():
                    if lang not in node_data["languages"]:
                        node_data["languages"].append(lang)
            
            # Process parents from sub_class_of
            if soli_data.get("sub_class_of"):
                for parent in soli_data["sub_class_of"]:
                    if is_soli_url(parent) and parent not in node_data["parents"]:
                        node_data["parents"].append(parent)
            
            # Process children
            if soli_data.get("parent_class_of"):
                for child in soli_data["parent_class_of"]:
                    if is_soli_url(child) and child not in node_data["children"]:
                        node_data["children"].append(child)
            
            # Process additional URLs for context
            urls_to_process = []
            if soli_data.get("is_defined_by"):
                urls_to_process.append(soli_data["is_defined_by"])
            urls_to_process.extend(soli_data.get("see_also", []))
            
            if urls_to_process:
                context, failed = process_urls_for_context(urls_to_process, cache_dir, current_time)
                if context:
                    node_data["context"].extend(context)
                    for ctx in context:
                        if ctx["local_path"] not in node_data["downloadedRefs"]:
                            node_data["downloadedRefs"].append(ctx["local_path"])
                if failed:
                    for url in failed:
                        if url not in node_data["failed_urls"]:
                            node_data["failed_urls"].append(url)
    
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
    current_time = "2025-01-11T14:42:19-08:00"  # Use provided time
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
        kb = {
            "nodes": {},
            "failed_urls": [],
            "downloadedRefs": [],
            "refs": []
        }
    
    # Find all subjects that have any predicate
    nodes = []
    seen = set()  # Just for deduping
    for s, p, o in g:
        node_uri = str(s)
        if is_soli_url(node_uri) and node_uri not in seen:
            nodes.append(s)
            seen.add(node_uri)
    
    logger.info(f"Found {len(nodes)} unique SOLI URLs")
    
    if limit:
        nodes = nodes[:limit]
        logger.info(f"Limited to {limit} nodes")
    
    # Process each node
    for node in nodes:
        node_uri = str(node)
        node_key = extract_soli_key(node_uri)
        
        if not node_key:
            continue
            
        # Skip if we've already processed this node
        if node_key in kb["nodes"]:
            continue
        
        # Get the type if available
        node_type = "unknown"
        for _, _, type_uri in g.triples((node, RDF.type, None)):
            type_str = str(type_uri)
            if '#' in type_str:
                node_type = type_str.split('#')[-1].lower()
                break
        
        # Get English label if available
        english_label = None
        for _, _, label in g.triples((node, RDFS.label, None)):
            if isinstance(label, Literal) and (label.language is None or label.language == 'en'):
                english_label = str(label)
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
            "content_urls": [],
            "has_parent_in_kb": False,
            "languages": ["en"] if english_label else [],
            "deprecated": False,
            "country": None,
            "source": None,
            "labels": {"en": english_label} if english_label else {},
            "definitions": {},
            "class_not_found": False,
            "downloadedRefs": [],
            "type": node_type
        }
        
        # First try to download and process the about URL itself
        try:
            about_context, about_failed = process_urls_for_context([node_uri], cache_dir, current_time)
            if about_context:
                node_data["context"].extend(about_context)
                for ctx in about_context:
                    if ctx["local_path"] not in node_data["downloadedRefs"]:
                        node_data["downloadedRefs"].append(ctx["local_path"])
            if about_failed:
                for url in about_failed:
                    if url not in node_data["failed_urls"]:
                        node_data["failed_urls"].append(url)
        except Exception as e:
            logger.error(f"Failed to process about URL {node_uri}: {str(e)}")
            if node_uri not in node_data["failed_urls"]:
                node_data["failed_urls"].append(node_uri)
        
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
                })
                
                # Add definition to context if available
                if soli_data.get("definition"):
                    node_data["context"].append({
                        "url": node_uri,
                        "text": soli_data["definition"],
                        "type": "definition",
                        "time": format_time(current_time)
                    })
                
                # Add languages from translations
                if soli_data.get("translations"):
                    for lang in soli_data["translations"].keys():
                        if lang not in node_data["languages"]:
                            node_data["languages"].append(lang)
                
                # Add parent URLs
                if soli_data.get("sub_class_of"):
                    for parent in soli_data["sub_class_of"]:
                        if is_soli_url(parent) and parent not in node_data["parents"]:
                            node_data["parents"].append(parent)
                
                # Add child URLs
                if soli_data.get("parent_class_of"):
                    for child in soli_data["parent_class_of"]:
                        if is_soli_url(child) and child not in node_data["children"]:
                            node_data["children"].append(child)
                
                # Add content URLs
                if soli_data.get("is_defined_by"):
                    url = soli_data["is_defined_by"]
                    if url not in node_data["content_urls"]:
                        node_data["content_urls"].append(url)
                if soli_data.get("see_also"):
                    for url in soli_data["see_also"]:
                        if url not in node_data["content_urls"]:
                            node_data["content_urls"].append(url)
        
        # Increment counter for new node
        count = increment_count()
        logger.info(f"Found new class {node_key} of type {node_type}. Total count: {count}")
        
        # Add to KB
        kb["nodes"][node_key] = node_data
        
        # Update global lists
        for url in node_data["failed_urls"]:
            if url not in kb["failed_urls"]:
                kb["failed_urls"].append(url)
        for ref in node_data["downloadedRefs"]:
            if ref not in kb["downloadedRefs"]:
                kb["downloadedRefs"].append(ref)
        
        # Write KB after each node is processed
        with open(output_file, 'w') as f:
            json.dump(kb, f, indent=2)
    
    logger.info(f"Knowledge base creation complete. Output saved to {output_file}")
