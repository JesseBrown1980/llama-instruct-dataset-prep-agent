# main.py

import logging
from knowledge_graph import build_knowledge_graph
from downloader import download_and_clean_resource
from agents import cleaner_agent
import json

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
    owl_input = "SOLI/SOLI.owl"  # Adjust path as needed

    # Create knowledge base
    knowledge_base = build_knowledge_graph(owl_input)
    
    # Download and clean references
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
    with open("knowledge_base.json", "w") as f:
        json.dump(knowledge_base, f, indent=2)
    
    logger.info("Knowledge base creation complete.")

if __name__ == "__main__":
    main()
