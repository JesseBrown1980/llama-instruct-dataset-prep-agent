import threading
import logging
import argparse
import time
from build_knowledge import build_knowledge_base
from create_dataset import create_dataset
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class PipelineThread(threading.Thread):
    def __init__(self, target, name, **kwargs):
        super().__init__()
        self.target = target
        self.name = name
        self.kwargs = kwargs
        self.result = None
        self.error = None
        
    def run(self):
        try:
            logger.info(f"Starting {self.name}")
            self.result = self.target(**self.kwargs)
            logger.info(f"Completed {self.name}")
        except Exception as e:
            self.error = e
            logger.error(f"Error in {self.name}: {str(e)}")

def wait_for_file(file_path, timeout=300, check_interval=5):
    """Wait for a file to exist and be non-empty."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return True
        time.sleep(check_interval)
    return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the complete pipeline')
    parser.add_argument('--owl-input', type=str, required=True, help='Path to input OWL file')
    parser.add_argument('--kb-output', type=str, required=True, help='Path to knowledge base output JSON file')
    parser.add_argument('--dataset-output', type=str, required=True, help='Path to final dataset output file')
    parser.add_argument('--kb-limit', type=str, default="all", 
                      help='Number of classes/properties to process for knowledge base or "all"')
    parser.add_argument('--dataset-limit', type=str, default="all",
                      help='Number of nodes to process for dataset creation or "all"')
    args = parser.parse_args()

    # Step 1: Build Knowledge Base
    logger.info("Starting knowledge base creation pipeline")
    build_thread = PipelineThread(
        target=build_knowledge_base,
        name="Knowledge Base Creation",
        owl_input=args.owl_input,
        output=args.kb_output,
        limit=args.kb_limit
    )
    build_thread.start()
    build_thread.join()  # Wait for knowledge base creation to complete

    if build_thread.error:
        logger.error("Knowledge base creation failed. Stopping pipeline.")
        return

    # Wait for the knowledge base file to be ready
    if not wait_for_file(args.kb_output):
        logger.error(f"Timeout waiting for knowledge base file: {args.kb_output}")
        return

    # Step 2: Create Dataset
    logger.info("Starting dataset creation pipeline")
    dataset_thread = PipelineThread(
        target=create_dataset,
        name="Dataset Creation",
        kb_path=args.kb_output,
        output=args.dataset_output,
        limit=args.dataset_limit
    )
    dataset_thread.start()
    dataset_thread.join()  # Wait for dataset creation to complete

    if dataset_thread.error:
        logger.error("Dataset creation failed.")
        return

    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
