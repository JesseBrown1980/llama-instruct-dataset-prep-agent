import json
import logging
import argparse
from typing import Dict, List, Optional
import os
from datetime import datetime
from agents import (
    questioner_agent_invoke,
    question_checker_agent_invoke,
    create_dataset_entry
)
from error_handler import handle_error
from create_dataset_from_node import (
    get_node_context,
    chunk_text_with_overlap,
    process_node,
    DatasetCreator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_creation.log')
    ]
)
logger = logging.getLogger(__name__)

def create_dataset(kb_path: str, output: str, limit: Optional[str] = None) -> None:
    """Create dataset from knowledge base."""
    # Load knowledge base
    logger.info(f"Loading knowledge base from {kb_path}")
    with open(kb_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)
        
    # Create dataset
    creator = DatasetCreator(output)
    
    # Convert limit to int if specified
    limit_num = None
    if limit and limit.lower() != 'all':
        try:
            limit_num = int(limit)
        except ValueError:
            logger.error(f"Invalid limit value: {limit}. Use 'all' or a number.")
            return
    
    # Process each root node until limit is reached
    processed_count = 0
    nodes = kb.get('nodes', {})
    
    # Find root nodes (nodes without parents)
    root_nodes = []
    for node_key, node in nodes.items():
        # A node is a root if no other node has it as a child
        if not any(node_key in other_node.get('children', []) for other_node in nodes.values()):
            root_nodes.append(node_key)
    
    # Process each root node
    for root_key in root_nodes:
        if root_key in nodes:
            processed_count = process_node(
                nodes[root_key],
                processed_count,
                limit_num,
                nodes,
                creator
            )
            
            if limit_num is not None and processed_count >= limit_num:
                break

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb-path', required=True, help='Path to knowledge base JSON file')
    parser.add_argument('--output', required=True, help='Path to output directory')
    parser.add_argument('--limit', default='all', help='Number of nodes to process (default: all)')
    return parser.parse_args()

def main():
    try:
        args = parse_args()
        create_dataset(args.kb_path, args.output, args.limit)
    except Exception as e:
        handle_error(e, context="create_dataset main")

if __name__ == "__main__":
    main()
