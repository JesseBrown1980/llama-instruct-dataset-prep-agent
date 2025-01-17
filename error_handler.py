import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('data', 'error_recovery.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_last_processed_node() -> Optional[str]:
    """Read the nodetraversal.jsonl file and get the last processed node ID."""
    try:
        nodetraversal_path = os.path.join('data', 'nodetraversal.jsonl')
        
        if not os.path.exists(nodetraversal_path):
            logger.error(f"Node traversal file not found: {nodetraversal_path}")
            return None
            
        # Read all lines and get the last valid entry
        with open(nodetraversal_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Process lines in reverse to get last valid entry
        for line in reversed(lines):
            try:
                entry = json.loads(line.strip())
                if 'node_key' in entry and entry.get('status') == 'processed':
                    return entry['node_key']
            except json.JSONDecodeError:
                continue
        
        logger.error("No valid processed nodes found in nodetraversal.jsonl")
        return None
        
    except Exception as e:
        logger.error(f"Error reading last processed node: {str(e)}")
        return None

def handle_error(error: Exception, context: str = "") -> None:
    """
    Handle any error by:
    1. Logging the error
    2. Waiting 10 minutes
    3. Restarting from the last processed node
    
    Args:
        error: The exception that was caught
        context: Additional context about where the error occurred
    """
    try:
        # Log the error
        logger.error(f"Error occurred in {context}: {str(error)}")
        
        # Log error details to error log
        error_log_path = os.path.join('data', 'errors.jsonl')
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context
        }
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(error_entry) + '\n')
            f.flush()
        
        # Get the last processed node
        last_node = get_last_processed_node()
        if not last_node:
            logger.error("Could not determine last processed node. Cannot continue.")
            return
            
        # Wait 10 minutes
        logger.info(f"Waiting 10 minutes before resuming from node {last_node}")
        time.sleep(600)  # 10 minutes in seconds
        
        # Restart the process
        logger.info(f"Restarting process from node {last_node}")
        cmd = [
            sys.executable,
            'create_dataset_from_node.py',
            'full-kb.json',
            'dataset.jsonl',
            last_node,
            '--limit',
            'all'
        ]
        
        subprocess.run(cmd, check=True)
        
    except Exception as recovery_error:
        logger.error(f"Error during error recovery: {str(recovery_error)}")
        # If recovery fails, we don't want to create an infinite loop
        # Just log it and exit
        sys.exit(1)
