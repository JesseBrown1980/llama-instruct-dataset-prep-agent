# main.py

import logging
from knowledge_graph import build_knowledge_graph
from downloader import download_and_clean_resource
from agents import cleaner_agent
import json
import argparse
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('owl_processing.log')
    ]
)
logger = logging.getLogger(__name__)

def ensure_file_exists(file_path: str, default_content: dict = None):
    """Ensure file exists, create with default content if it doesn't."""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            content = default_content or {
                "nodes": {},
                "refs": [],
                "failed_urls": []
            }
            json.dump(content, f, indent=2)

def parse_limit(limit_str: str) -> int:
    """Parse the limit argument which can be a number or 'all'"""
    if limit_str.lower() == 'all':
        return None
    try:
        limit = int(limit_str)
        if limit <= 0:
            raise ValueError("Limit must be positive")
        return limit
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{limit_str}' is not a valid limit. Use a positive number or 'all'")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build knowledge base from OWL file')
    parser.add_argument('--owl-input', type=str, required=True, help='Path to input OWL file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--limit', type=str, default="all", help='Number of classes and properties to process or \'all\' for no limit')
    
    args = parser.parse_args()
    
    # Parse the limit
    limit = parse_limit(args.limit)

    # Ensure output file exists
    ensure_file_exists(args.output)

    # Build knowledge graph
    build_knowledge_graph(
        owl_file=args.owl_input,
        output_file=args.output,
        limit=limit
    )
    
    # Download and clean references
    knowledge_base = json.load(open(args.output))
    clean_refs = []
    for ref in knowledge_base["refs"]:
        ref_url = str(ref)
        try:
            # Download and clean reference in one step
            ref_data = download_and_clean_resource(ref_url, parent_context=f"Legal ontology reference from {ref_url}")
            if ref_data["text"]:  # Only add if we got text back
                clean_refs.append(ref_data["text"])
                logger.info(f"Successfully processed reference: {ref_url}")
            else:
                logger.warning(f"No text content found for reference: {ref_url}")
        except Exception as e:
            logger.error(f"Failed to process reference {ref_url}: {str(e)}")

    knowledge_base["downloadedRefs"] = clean_refs
    
    # Save knowledge base
    with open(args.output, "w") as f:
        json.dump(knowledge_base, f, indent=2)
    
    logger.info(f"Knowledge base creation complete. Output saved to {args.output}")
    if limit:
        logger.info(f"Note: Output was limited to {limit} classes and properties for testing")

if __name__ == "__main__":
    main()
