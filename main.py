# main.py

import logging
from pipeline import process_owl_file_full

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
    dataset = process_owl_file_full(
        owl_path=owl_input,
        limit=5,
        output_kg="knowledge_graph.json",
        output_ds="dolly_dataset.json"
    )
    logger.info("Process complete.")

if __name__ == "__main__":
    main()
