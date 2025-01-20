import json
import logging
import argparse
from typing import Dict, List, Optional
import os
from datetime import datetime
from agents import (
    questioner_agent_invoke,
    question_checker_agent_invoke,
    maker_agent,
    formatter_agent,
    checker_agent,
    cleaner_agent,
    make_instruction_agent,
    create_dataset_entry
)
from error_handler import handle_error

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('data', 'dataset_creation.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_node_text(node: Dict) -> Optional[str]:
    """Extract text content from a node, checking both context and definition fields."""
    # Try to get text from context first
    if node.get('context'):
        for ctx in node['context']:
            if ctx.get('text'):
                return ctx['text']
    
    # If no context text, try definition field
    if node.get('definition'):
        return node['definition']
    
    return None

def get_node_context(node: Dict, nodes: Dict) -> Optional[str]:
    """Get the combined context from a node and its parents/children."""
    # Get main node text
    main_text = get_node_text(node)
    if not main_text:
        return None
        
    context_parts = [main_text]
    
    # Add parent context if available
    if node.get('parents'):
        for parent_url in node['parents']:
            parent_id = parent_url.split('/')[-1]
            if parent_id in nodes:
                parent_text = get_node_text(nodes[parent_id])
                if parent_text:
                    context_parts.append(f"Parent context: {parent_text}")
    
    # Add children context if available
    if node.get('children'):
        for child_url in node['children']:
            child_id = child_url.split('/')[-1]
            if child_id in nodes:
                child_text = get_node_text(nodes[child_id])
                if child_text:
                    context_parts.append(f"Child context: {child_text}")
    
    return ' '.join(context_parts)

def chunk_text_with_overlap(text: str, max_words: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        # Calculate end of current chunk
        end = start + max_words
        
        # If this is not the last chunk, include overlap
        if end < len(words):
            # Look for a period in the overlap region to make a cleaner break
            overlap_end = min(end + overlap, len(words))
            period_pos = -1
            for i in range(end, overlap_end):
                if words[i].endswith('.'):
                    period_pos = i + 1
                    break
            
            if period_pos != -1:
                end = period_pos
            else:
                end = overlap_end
        else:
            end = len(words)
        
        # Create chunk
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start = end - overlap if end < len(words) else end
    
    return chunks

class DatasetCreator:
    def __init__(self, output_path: str):
        self.output_path = output_path
        
        # Initialize data files
        self.data_dir = os.path.join(os.path.dirname(output_path), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.chunks_file = os.path.join(self.data_dir, 'chunks.jsonl')
        self.questions_file = os.path.join(self.data_dir, 'questions.jsonl')
        self.dataset_file = os.path.join(self.data_dir, 'dataset.jsonl')
        self.metadata_file = os.path.join(self.data_dir, 'nodetraversal.jsonl')
        
        # Create files only if they don't exist
        for file_path in [self.chunks_file, self.questions_file, self.dataset_file, self.metadata_file]:
            if not os.path.exists(file_path):
                open(file_path, 'w').close()
                logger.info(f"Created new file: {os.path.basename(file_path)}")
            else:
                logger.info(f"Using existing file: {os.path.basename(file_path)}")

    def is_node_processed(self, node_key: str) -> bool:
        """Check if a node has already been processed by looking in nodetraversal.jsonl"""
        if not os.path.exists(self.metadata_file):
            return False
            
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    metadata = json.loads(line)
                    if metadata.get('node_key') == node_key:
                        return True
                except json.JSONDecodeError:
                    continue
        return False

    def mark_node_processed(self, node_key: str) -> None:
        """Mark a node as processed in nodetraversal.jsonl"""
        metadata = {
            'node_key': node_key,
            'processed_at': datetime.now().isoformat(),
            'status': 'processed'
        }
        with open(self.metadata_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metadata) + '\n')
            f.flush()

def create_dataset_from_node(kb_path: str, output: str, start_node_id: str, limit: Optional[int] = None) -> None:
    """Create dataset starting from a specific node in the knowledge base.
    
    This will process either:
    1. All nodes in the KB if limit is None
    2. The specified number of nodes if limit is an integer
    """
    logger.info(f"Loading knowledge base from {kb_path}")
    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {str(e)}")
        return
    
    if not isinstance(kb, dict) or 'nodes' not in kb:
        logger.error("Invalid knowledge base format - missing 'nodes' dictionary")
        return
    
    # Initialize tracking sets
    processed_nodes = set()  # All processed nodes can be retrieved from the nodetraversal.jsonl file
    nodes_to_process = []    # Queue of nodes to process
    
    # Get all nodes
    nodes = kb['nodes']
    total_nodes = len(nodes)
    logger.info(f"Found {total_nodes} total nodes in knowledge base")
    
    # Find the start node and add all subsequent nodes to the queue
    if start_node_id not in nodes:
        logger.error(f"Start node {start_node_id} not found in knowledge base")
        return
        
    # Convert nodes dictionary keys to a list to maintain order
    node_keys = list(nodes.keys())
    # Find the index of the start node
    start_idx = node_keys.index(start_node_id)
    # Add all nodes that appear after the start node
    nodes_to_process = node_keys[start_idx:]
    
    # Calculate total nodes to process
    total_to_process = len(nodes_to_process)
    if limit is not None and limit != 'all':
        # If there's a numeric limit, only keep that many nodes
        nodes_to_process = nodes_to_process[:int(limit)]
        total_to_process = len(nodes_to_process)
    
    logger.info(f"Starting from node at index {start_idx + 1}, processing {total_to_process} nodes")
    creator = DatasetCreator(output)
    nodes_processed = 0
    
    # Process nodes until queue is empty or limit reached
    while nodes_to_process and (limit is None or nodes_processed < limit):
        current_id = nodes_to_process.pop(0)
        
        # Skip if already processed
        if current_id in processed_nodes:
            continue
            
        # Get current node
        current_node = nodes.get(current_id)
        if not current_node:
            continue
            
        # Process current node
        try:
            # Get node text
            node_text = get_node_text(current_node)
            if not node_text:
                continue
                
            # Get context including parents/children
            context = get_node_context(current_node, nodes)
            if not context:
                continue
                
            # Process the node
            logger.info(f"Processing node {nodes_processed + 1}/{total_to_process}: {current_id} (KB index: {start_idx + nodes_processed + 1})")
            
            # Generate and process chunks
            chunks = chunk_text_with_overlap(context) if len(context.split()) > 2000 else [context]
            for i, chunk in enumerate(chunks):
                # Log chunk
                with open(creator.chunks_file, 'a', encoding='utf-8') as f:
                    chunk_info = {
                        'node_key': current_id,
                        'chunk_index': i,
                        'chunk_text': chunk,
                        'timestamp': datetime.now().isoformat()
                    }
                    f.write(json.dumps(chunk_info) + '\n')
                    f.flush()
                
                # Generate questions
                questions_json = questioner_agent_invoke(chunk)
                questions = json.loads(questions_json)
                
                if questions:
                    # Log raw questions
                    with open(creator.questions_file, 'a', encoding='utf-8') as f:
                        for q in questions:
                            question_info = {
                                'node_key': current_id,
                                'chunk_index': i,
                                'chunk_text': chunk,
                                'question': q,
                                'status': 'raw',
                                'timestamp': datetime.now().isoformat()
                            }
                            f.write(json.dumps(question_info) + '\n')
                            f.flush()
                    
                    # Process questions
                    for question in questions:
                        try:
                            entry = create_dataset_entry(question, context, current_id)
                            if entry:
                                with open(creator.dataset_file, 'a', encoding='utf-8') as f:
                                    f.write(entry + '\n')
                                    f.flush()
                        except Exception as e:
                            handle_error(e, context=f"Processing question for node {current_id}")
            
            # Mark node as processed
            processed_nodes.add(current_id)
            creator.mark_node_processed(current_id)
            nodes_processed += 1
            
            # If processing all nodes, we don't need to queue parents/children
            if limit is not None:
                # Queue parent nodes
                if 'parents' in current_node:
                    for parent_url in current_node['parents']:
                        parent_id = parent_url.split('/')[-1]
                        if parent_id in nodes and parent_id not in processed_nodes:
                            nodes_to_process.append(parent_id)
                
                # Queue child nodes
                if 'children' in current_node:
                    for child_url in current_node['children']:
                        child_id = child_url.split('/')[-1]
                        if child_id in nodes and child_id not in processed_nodes:
                            nodes_to_process.append(child_id)
            
        except Exception as e:
            handle_error(e, context=f"Processing node {current_id}")
            continue
    
    logger.info(f"Processed {nodes_processed} nodes out of {total_nodes} total nodes")

def find_node_and_descendants(kb: Dict, start_node_id: str) -> Optional[Dict]:
    """Find a node by key and return it along with all its descendants."""
    nodes = kb.get('nodes', {})
    if not nodes:
        logger.error("No nodes found in knowledge base")
        return None
        
    if start_node_id not in nodes:
        logger.error(f"Node {start_node_id} not found in knowledge base nodes")
        return None
    
    node = nodes[start_node_id]
    node_text = get_node_text(node)
    
    if not node_text:
        logger.error(f"Node {start_node_id} has no text content in either context or definition")
        return None
        
    # Store the text in a consistent location for processing
    node['text'] = node_text
    logger.info(f"Found node {start_node_id} with text length: {len(node_text)}")
    return node

def parse_args():
    parser = argparse.ArgumentParser(description='Create dataset from knowledge base starting from a specific node')
    parser.add_argument('kb_path', help='Path to knowledge base JSON file')
    parser.add_argument('output', help='Path to output dataset file')
    parser.add_argument('start_node_id', help='ID of node to start from')
    parser.add_argument('--limit', help='Maximum number of nodes to process (use "all" for no limit)', default='all')
    return parser.parse_args()

def main():
    try:
        args = parse_args()
        limit = None if args.limit.lower() == 'all' else int(args.limit)
        create_dataset_from_node(args.kb_path, args.output, args.start_node_id, limit)
    except Exception as e:
        handle_error(e, context="create_dataset_from_node main")

if __name__ == "__main__":
    main()
