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
    make_instruction_agent
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
    def __init__(self, knowledge_base_path: str, output_path: str):
        self.knowledge_base_path = knowledge_base_path
        self.output_path = output_path
        self.knowledge_base = self._load_knowledge_base()
        
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

    def _load_knowledge_base(self) -> Dict:
        """Load and validate the knowledge base file."""
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
        if not isinstance(kb, dict) or 'nodes' not in kb:
            raise ValueError("Invalid knowledge base format")
        return kb

    def process_chunk(self, context_chunk: str, node_key: str, chunk_index: int) -> None:
        """Process a single context chunk to generate questions and answers."""
        chunk_id = f"{node_key}_chunk_{chunk_index}"
        
        # Write chunk
        with open(self.chunks_file, 'a', encoding='utf-8') as f:
            chunk = {"chunk_id": chunk_id, "node_key": node_key, "content": context_chunk}
            f.write(json.dumps(chunk) + '\n')
            f.flush()
        
        # Generate and write questions
        questions_json = questioner_agent_invoke(context_chunk)
        questions = json.loads(questions_json)
        
        # Write raw questions first
        with open(self.questions_file, 'a', encoding='utf-8') as f:
            for q in questions:
                question = {"chunk_id": chunk_id, "node_key": node_key, "question": q}
                f.write(json.dumps(question) + '\n')
                f.flush()
        
        # Then validate and process for dataset
        cleaned_questions_json = question_checker_agent_invoke(json.dumps(questions))
        cleaned_questions = json.loads(cleaned_questions_json)
        
        # Process each validated question
        with open(self.dataset_file, 'a', encoding='utf-8') as f:
            for question in cleaned_questions:
                try:
                    instruction = make_instruction_agent(question, context_chunk)
                    answer = maker_agent(question, context_chunk)
                    
                    entry = {
                        "instruction": instruction.strip(),
                        "context": context_chunk.strip(),
                        "response": answer.strip()
                    }
                    
                    validated = json.loads(checker_agent(json.dumps(entry)))
                    if all(k in validated for k in ['instruction', 'context', 'response']):
                        f.write(json.dumps(validated) + '\n')
                        f.flush()
                
                except Exception as e:
                    logger.error(f"Failed to process question: {str(e)}")
                    continue

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

    def create_dataset(self, limit: Optional[int] = None) -> None:
        """Create the dataset by processing nodes from the knowledge base."""
        nodes = list(self.knowledge_base['nodes'].values())
        if limit:
            nodes = nodes[:limit]
        
        for node in nodes:
            self.process_node(node)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kb-path', required=True, help='Knowledge base JSON file')
    parser.add_argument('--output', required=True, help='Output dataset path')
    parser.add_argument('--limit', type=int, help='Limit nodes to process')
    args = parser.parse_args()
    
    creator = DatasetCreator(args.kb_path, args.output)
    creator.create_dataset(limit=args.limit)

if __name__ == "__main__":
    main()
