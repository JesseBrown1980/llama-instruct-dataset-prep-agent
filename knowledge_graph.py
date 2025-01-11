# knowledge_graph.py

import logging
from rdflib import Graph, RDF, RDFS, SKOS, OWL
import re
from typing import Dict, Set, List, Optional
from downloader import download_and_clean_resource
import json
import os
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

def extract_soli_key(uri: str) -> Optional[str]:
    """Extract SOLI key from URI."""
    match = re.search(r'R[A-Za-z0-9]{22}', uri)
    return match.group(0) if match else None

def fetch_soli_metadata(uri: str) -> Optional[Dict]:
    """Fetch metadata from SOLI API for a given URI."""
    try:
        response = requests.get(uri)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch SOLI metadata for {uri}: {str(e)}")
        return None

def process_node(g: Graph, node_uri: str, cache_dir: str = "cache_refs") -> Dict:
    """Process a single node and gather all its metadata and relationships."""
    key = extract_soli_key(str(node_uri))
    if not key:
        return None
        
    # Initialize node data
    node_data = {
        "key": key,
        "uri": str(node_uri),
        "timestamp": datetime.now().isoformat(),
        "labels": {},
        "definitions": {},
        "alt_labels": {},
        "parents": [],
        "children": [],
        "context": [],
        "failed_urls": [],
        "has_parent_in_kb": False,
        "languages": set()
    }
    
    # Get labels and definitions
    for label in g.objects(node_uri, RDFS.label):
        lang = label.language or 'en'
        node_data["labels"][lang] = str(label)
        node_data["languages"].add(lang)
    
    for definition in g.objects(node_uri, SKOS.definition):
        lang = definition.language or 'en'
        node_data["definitions"][lang] = str(definition)
        node_data["languages"].add(lang)
    
    # Get alternative labels
    for alt_label in g.objects(node_uri, SKOS.altLabel):
        lang = alt_label.language or 'en'
        if lang not in node_data["alt_labels"]:
            node_data["alt_labels"][lang] = []
        node_data["alt_labels"][lang].append(str(alt_label))
        node_data["languages"].add(lang)
    
    # Fetch SOLI metadata
    soli_metadata = fetch_soli_metadata(str(node_uri))
    if soli_metadata:
        node_data.update({
            "preferred_label": soli_metadata.get("preferred_label"),
            "definition": soli_metadata.get("definition"),
            "alternative_labels": soli_metadata.get("alternative_labels", []),
            "translations": soli_metadata.get("translations", {}),
            "deprecated": soli_metadata.get("deprecated", False),
            "country": soli_metadata.get("country"),
            "source": soli_metadata.get("source")
        })
        
        # Add parent classes from SOLI metadata
        for parent_uri in soli_metadata.get("sub_class_of", []):
            parent_key = extract_soli_key(parent_uri)
            if parent_key:
                node_data["parents"].append(parent_key)
        
        # Add child classes from SOLI metadata
        for child_uri in soli_metadata.get("parent_class_of", []):
            child_key = extract_soli_key(child_uri)
            if child_key:
                node_data["children"].append(child_key)
    
    # Process relationships in the graph
    for parent in g.objects(node_uri, RDFS.subClassOf):
        parent_key = extract_soli_key(str(parent))
        if parent_key and parent_key not in node_data["parents"]:
            node_data["parents"].append(parent_key)
    
    for parent in g.objects(node_uri, RDFS.subPropertyOf):
        parent_key = extract_soli_key(str(parent))
        if parent_key and parent_key not in node_data["parents"]:
            node_data["parents"].append(parent_key)
    
    # Download and process context
    for context_uri in g.objects(node_uri, RDFS.isDefinedBy):
        try:
            context_data = download_and_clean_resource(str(context_uri), cache_dir=cache_dir)
            if context_data["downloaded"]:
                relative_path = os.path.relpath(context_data.get("local_path", ""), os.getcwd())
                node_data["context"].append({
                    "url": str(context_uri),
                    "local_path": relative_path,
                    "text": context_data["text"],
                    "time": context_data["time"]
                })
            else:
                node_data["failed_urls"].append(str(context_uri))
        except Exception as e:
            logger.error(f"Failed to process context {context_uri}: {str(e)}")
            node_data["failed_urls"].append(str(context_uri))
    
    # Convert sets to lists for JSON serialization
    node_data["languages"] = list(node_data["languages"])
    
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
    
    # Process classes and properties
    nodes = []
    for node_type in [(OWL.Class, "class"), (OWL.ObjectProperty, "property")]:
        for node in g.subjects(RDF.type, node_type[0]):
            nodes.append((node, node_type[1]))
    
    if limit:
        nodes = nodes[:limit]
    
    # Process each node
    for node, node_type in nodes:
        logger.info(f"Processing {node_type}: {node}")
        node_data = process_node(g, node, cache_dir)
        
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
