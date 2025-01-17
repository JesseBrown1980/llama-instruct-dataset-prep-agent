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

def traverse_kb(kb: Dict) -> List[str]:
    """Traverse the knowledge base and return a list of all node keys."""
    node_keys = []
    nodes = kb.get('nodes', {})
    
    def traverse_node(node_key: str, node: Dict):
        node_keys.append(node_key)
        if 'children' in node:
            for child_key in node['children']:
                # Only traverse child if it's a key in nodes dict
                if child_key in nodes:
                    traverse_node(child_key, nodes[child_key])
    
    # Start from root nodes (nodes without parents or with parents not in KB)
    for node_key, node in nodes.items():
        if not node.get('parents') or not any(parent in nodes for parent in node['parents']):
            traverse_node(node_key, node)
    
    return node_keys

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
    
    # Get list of all nodes to process
    node_keys = traverse_kb(kb)
    logger.info(f"Found {len(node_keys)} nodes to process")
    
    # Process each node until limit is reached
    processed_count = 0
    for node_key in node_keys:
        try:
            # Skip if already processed
            if creator.is_node_processed(node_key):
                logger.info(f"Node {node_key} already processed, skipping")
                continue
            
            # Get node context
            node = kb['nodes'][node_key]
            context = get_node_context(node, kb)
            if not context:
                logger.warning(f"No context found for node {node_key}, skipping")
                continue
            
            # Split context into chunks
            chunks = chunk_text_with_overlap(context)
            for i, chunk in enumerate(chunks):
                # Create dataset entries
                questions = questioner_agent_invoke(chunk)
                if not questions:
                    continue
                    
                filtered_questions = question_checker_agent_invoke(questions)
                if not filtered_questions:
                    continue
                
                # Create dataset entries for each question
                for question in json.loads(filtered_questions):
                    try:
                        entry = create_dataset_entry(question, chunk, node_key)
                        if entry:
                            with open(output, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(entry) + '\n')
                    except Exception as e:
                        logger.error(f"Error creating dataset entry: {str(e)}")
                        continue
            
            # Mark node as processed
            creator.mark_node_processed(node_key)
            processed_count += 1
            
            # Check if limit reached
            if limit_num and processed_count >= limit_num:
                logger.info(f"Reached limit of {limit_num} nodes")
                break
                
        except Exception as e:
            handle_error(e, node_key)
            continue
    
    logger.info(f"Finished processing {processed_count} nodes")

def parse_args():
    parser = argparse.ArgumentParser(description='Create dataset from knowledge base')
    parser.add_argument('kb_path', help='Path to knowledge base JSON file')
    parser.add_argument('output', help='Path to output JSONL file')
    parser.add_argument('--limit', help='Number of nodes to process (default: all)', default='all')
    return parser.parse_args()

def main():
    args = parse_args()
    create_dataset(args.kb_path, args.output, args.limit)

if __name__ == "__main__":
    main()
