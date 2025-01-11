# main.py

import logging
from knowledge_graph import build_knowledge_graph
from downloader import download_and_clean_resource
from agents import cleaner_agent
import json
import argparse

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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build knowledge base from OWL file')
    parser.add_argument('--limit', type=int, help='Limit number of classes and properties (for testing)', default=None)
    parser.add_argument('--owl-input', type=str, default="SOLI/SOLI.owl", help='Path to input OWL file')
    parser.add_argument('--output', type=str, default="knowledge_base.json", help='Path to output JSON file')
    args = parser.parse_args()

    # Build knowledge graph
    build_knowledge_graph(
        owl_file=args.owl_input,
        output_file=args.output,
        limit=args.limit
    )
    
    # Download and clean references
    knowledge_base = json.load(open(args.output))
    clean_refs = []
    for ref in knowledge_base["refs"]:
        ref_url = str(ref)
        try:
            # Download reference
            ref_data = download_and_clean_resource(ref_url)
            # Clean using agent
            cleaned_data = cleaner_agent(ref_data, context=f"Legal ontology reference from {ref_url}")
            clean_refs.append(cleaned_data)
            logger.info(f"Successfully processed reference: {ref_url}")
        except Exception as e:
            logger.error(f"Failed to download reference {ref_url}: {str(e)}")

    knowledge_base["downloadedRefs"] = clean_refs
    
    # Save knowledge base
    with open(args.output, "w") as f:
        json.dump(knowledge_base, f, indent=2)
    
    logger.info(f"Knowledge base creation complete. Output saved to {args.output}")
    if args.limit:
        logger.info(f"Note: Output was limited to {args.limit} classes and properties for testing")

if __name__ == "__main__":
    main()
