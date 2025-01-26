import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Optional

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

def get_last_processed_nodes() -> List[str]:
    """Read the nodetraversal.jsonl file and get the last two processed node IDs."""
    try:
        nodetraversal_path = os.path.join('data', 'nodetraversal.jsonl')
        
        if not os.path.exists(nodetraversal_path):
            logger.error(f"Node traversal file not found: {nodetraversal_path}")
            return []
            
        # Read all lines and get the last two entries
        with open(nodetraversal_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Process lines in reverse to get last two valid entries
        last_nodes = []
        for line in reversed(lines):
            try:
                entry = json.loads(line.strip())
                if 'node_key' in entry:
                    last_nodes.append(entry['node_key'])
                    if len(last_nodes) == 2:
                        break
            except json.JSONDecodeError:
                continue
        
        return last_nodes
        
    except Exception as e:
        logger.error(f"Error reading last processed nodes: {str(e)}")
        return []

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

def get_next_node(current_node: str) -> Optional[str]:
    """Get the next node ID from the knowledge base."""
    try:
        # Load the knowledge base
        with open('full-kb.json', 'r', encoding='utf-8') as f:
            kb = json.load(f)
            
        if not isinstance(kb, dict) or 'nodes' not in kb:
            logger.error("Invalid knowledge base format - missing 'nodes' dictionary")
            return None
            
        # Get list of node keys
        node_keys = list(kb['nodes'].keys())
        
        # Find current node index
        try:
            current_idx = node_keys.index(current_node)
            # Return next node if available
            if current_idx + 1 < len(node_keys):
                return node_keys[current_idx + 1]
        except ValueError:
            logger.error(f"Current node {current_node} not found in knowledge base")
            return None
            
        return None
        
    except Exception as e:
        logger.error(f"Error getting next node: {str(e)}")
        return None

def handle_error(error: Exception, context: str = "") -> None:
    """
    Handle any error by:
    1. Logging the error
    2. If same node failed twice, skip to next node
    3. Otherwise wait 10 minutes and retry current node
    
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
        
        # Get the last two processed nodes
        last_nodes = get_last_processed_nodes()
        if not last_nodes:
            logger.error("Could not determine last processed node. Cannot continue.")
            return
            
        current_node = last_nodes[0]
        
        # Check if we've had two consecutive errors on the same node
        if len(last_nodes) >= 2 and last_nodes[0] == last_nodes[1]:
            logger.warning(f"Node {current_node} has failed twice consecutively. Skipping to next node...")
            next_node = get_next_node(current_node)
            if not next_node:
                logger.error("Could not find next node to process. Cannot continue.")
                return
                
            # Log skip information
            skip_log_path = os.path.join('data', 'skip.jsonl')
            skip_entry = {
                'timestamp': datetime.now().isoformat(),
                'skipped_node': current_node,
                'next_node': next_node,
                'reason': 'consecutive_errors',
                'error': {
                    'type': type(error).__name__,
                    'message': str(error),
                    'context': context
                }
            }
            with open(skip_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(skip_entry) + '\n')
                f.flush()
                
            # Restart from next node
            logger.info(f"Restarting process from node {next_node}")
            cmd = [
                sys.executable,
                'create_dataset_from_node.py',
                'full-kb.json',
                'dataset.jsonl',
                next_node,
                '--limit',
                'all'
            ]
            
            subprocess.run(cmd, check=True)
            return
            
        # Otherwise, proceed with normal error recovery
        # Log restart information
        restart_log_path = os.path.join('data', 'restart.jsonl')
        restart_entry = {
            'timestamp': datetime.now().isoformat(),
            'node_key': current_node,
            'error': {
                'type': type(error).__name__,
                'message': str(error),
                'context': context
            },
            'restart_metadata': {
                'wait_time_seconds': 600,
                'recovery_script': 'create_dataset_from_node.py'
            }
        }
        with open(restart_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(restart_entry) + '\n')
            f.flush()
            
        # Wait 10 minutes
        logger.info(f"Waiting 2 minutes before resuming from node {current_node}")
        time.sleep(120)  # 10 minutes in seconds
        
        # Restart the process
        logger.info(f"Restarting process from node {current_node}")
        cmd = [
            sys.executable,
            'create_dataset_from_node.py',
            'full-kb.json',
            'dataset.jsonl',
            current_node,
            '--limit',
            'all'
        ]
        
        subprocess.run(cmd, check=True)
        
    except Exception as recovery_error:
        logger.error(f"Error during error recovery: {str(recovery_error)}")
        # If recovery fails, we don't want to create an infinite loop
        # Just log it and exit
        sys.exit(1)
