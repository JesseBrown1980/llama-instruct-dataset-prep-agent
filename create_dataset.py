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

def chunk_text(text: str, max_words: int = 2000) -> List[str]:
    """Split text into chunks of approximately max_words."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_count = 0
    
    for word in words:
        current_chunk.append(word)
        current_count += 1
        
        if current_count >= max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_count = 0
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
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
        
        # Clear files
        open(self.chunks_file, 'w').close()
        open(self.questions_file, 'w').close()
        open(self.dataset_file, 'w').close()

    def process_chunk(self, context_chunk: str, node_key: str, chunk_index: int) -> None:
        """Process a single context chunk to generate questions and answers."""
        # Generate questions - wait for completion
        logger.info(f"Generating questions")
        questions_json = questioner_agent_invoke(context_chunk)
        questions = json.loads(questions_json)
        
        if not questions:
            logger.warning(f"No questions generated")
            return
            
        # Write raw questions
        questions_jsonl = ""
        with open(self.questions_file, 'a', encoding='utf-8') as f:
            for q in questions:
                question = {"question": q}
                question_json = json.dumps(question)
                f.write(question_json + '\n')
                f.flush()
                questions_jsonl += question_json + '\n'
        
        # Validate questions - wait for completion
        logger.info(f"Validating questions")
        cleaned_questions_json = question_checker_agent_invoke(questions_jsonl)
        cleaned_questions = json.loads(cleaned_questions_json)
        
        if not cleaned_questions:
            logger.warning(f"No questions passed validation")
            return
            
        # Process each validated question and write immediately
        logger.info(f"Processing {len(cleaned_questions)} validated questions")
        successful_entries = 0
        
        for question in cleaned_questions:
            # Generate instruction with higher temperature for creativity
            logger.info(f"Generating instruction for question: {question}")
            instruction = make_instruction_agent(question, context_chunk)
                
            # Generate response with lower temperature for accuracy
            logger.info(f"Generating response")
            response = maker_agent(question, context_chunk)
            
            # Create entry - both instruction and response are guaranteed by retry logic
            entry = {
                "instruction": instruction.strip(),
                "context": context_chunk.strip(),
                "response": response.strip()
            }
            
            # Write immediately
            with open(self.dataset_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
                f.flush()
            successful_entries += 1
            logger.info(f"Successfully wrote entry {successful_entries} for question: {question}")
                
        logger.info(f"Completed processing - wrote {successful_entries}/{len(cleaned_questions)} entries")

    def process_node(self, node: Dict) -> None:
        """Process a single node by generating questions from each chunk."""
        node_key = node['key']
        logger.info(f"Processing node: {node_key}")
        
        # Build context
        context = ""
        if node.get('context'):
            for ctx in node['context']:
                if ctx.get('type') == 'definition' and ctx.get('text'):
                    context = ctx['text']
                    break
        
        if not context:
            return
            
        # Clean and chunk
        cleaned_context = cleaner_agent(context)
        chunks = chunk_text(cleaned_context)
        
        # Process chunks
        for i, chunk in enumerate(chunks, 1):
            self.process_chunk(chunk, node_key, i)

def create_dataset(kb_path: str, output: str, limit: str = None):
    """Create dataset from knowledge base."""
    # Load knowledge base
    with open(kb_path, 'r', encoding='utf-8') as f:
        kb = json.load(f)
        
    # Create dataset
    creator = DatasetCreator(output)
    
    # Process nodes
    nodes = list(kb['nodes'].items())
    if limit and limit.lower() != 'all':
        try:
            limit_num = int(limit)
            nodes = nodes[:limit_num]
        except ValueError:
            logger.error(f"Invalid limit value: {limit}. Use 'all' or a number.")
            return
            
    for node_key, node in nodes:
        creator.process_node(node)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb-path', required=True, help='Path to knowledge base JSON file')
    parser.add_argument('--output', required=True, help='Path to output dataset file')
    parser.add_argument('--limit', type=str, default=None, 
                      help='Limit number of nodes to process. Use "all" for no limit or a number')
    return parser.parse_args()

def main():
    args = parse_args()
    create_dataset(args.kb_path, args.output, args.limit)

if __name__ == "__main__":
    main()
