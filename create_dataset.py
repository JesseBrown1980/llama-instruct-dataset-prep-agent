import json
import logging
from typing import Dict, List, Set
import re
from agents import (
    questioner_agent_invoke,
    question_checker_agent_invoke,
    maker_agent,
    formatter_agent,
    checker_agent
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_creation.log')
    ]
)
logger = logging.getLogger(__name__)

def extract_key_from_uri(uri: str) -> str:
    """Extract the key from a URI that matches the SOLI format."""
    match = re.search(r'R[A-Za-z0-9]{22}', uri)
    return match.group(0) if match else None

def get_referenced_keys(entry: Dict, knowledge_base: Dict) -> Set[str]:
    """Find all referenced keys in an entry's relationships."""
    referenced_keys = set()
    
    # Check relationships
    for rel in knowledge_base["relationships"]:
        if rel["subject"] == entry["uri"]:
            key = extract_key_from_uri(rel["object"])
            if key:
                referenced_keys.add(key)
        elif rel["object"] == entry["uri"]:
            key = extract_key_from_uri(rel["subject"])
            if key:
                referenced_keys.add(key)
    
    return referenced_keys

def build_context(key: str, knowledge_base: Dict) -> str:
    """Build context string for a given key using referenced entries."""
    # Find the entry for this key
    entry = None
    for collection in [knowledge_base["classes"], knowledge_base["properties"]]:
        for uri, data in collection.items():
            if extract_key_from_uri(uri) == key:
                entry = data
                break
        if entry:
            break
    
    if not entry:
        return ""
    
    # Get referenced keys
    referenced_keys = get_referenced_keys(entry, knowledge_base)
    
    # Build context string
    context = f"Main Entry:\n{entry['label']}: {entry['definition']}\n\n"
    
    if referenced_keys:
        context += "Related Entries:\n"
        for ref_key in referenced_keys:
            # Find referenced entry
            for collection in [knowledge_base["classes"], knowledge_base["properties"]]:
                for uri, data in collection.items():
                    if extract_key_from_uri(uri) == ref_key:
                        context += f"- {data['label']}: {data['definition']}\n"
                        break
    
    return context.strip()

def process_entry(entry: Dict, context: str) -> List[Dict]:
    """Process a single entry using agents to generate questions and answers."""
    dataset_items = []
    
    # Generate questions using questioner agent
    questions_json = questioner_agent_invoke(context)
    
    # Check and validate questions
    checked_questions_json = question_checker_agent_invoke(questions_json)
    try:
        questions = json.loads(checked_questions_json)
        if not isinstance(questions, list):
            questions = []
    except:
        questions = []
        
    # Generate answers for each question
    for question in questions:
        # Generate raw answer
        raw_answer = maker_agent(question, context)
        
        # Format answer into proper JSON structure
        draft_json = formatter_agent(raw_answer, question, context)
        
        # Final check of the JSON
        checked_json = checker_agent(draft_json)
        
        try:
            final_data = json.loads(checked_json)
            if isinstance(final_data, dict):
                dataset_items.append(final_data)
            elif isinstance(final_data, list):
                dataset_items.extend(final_data)
        except:
            logger.warning(f"Failed to parse final JSON for question: {question}")
            
    return dataset_items

def create_dataset(knowledge_base_path: str, output_path: str, limit: int = None):
    """Create a dataset from the knowledge base using agents."""
    # Load knowledge base
    with open(knowledge_base_path, 'r') as f:
        knowledge_base = json.load(f)
    
    dataset = []
    
    # Process classes and properties
    items = []
    for collection in [knowledge_base["classes"], knowledge_base["properties"]]:
        for uri, entry in collection.items():
            key = extract_key_from_uri(uri)
            if not key:
                continue
            items.append((key, entry))
    
    # Apply limit if specified
    if limit:
        items = items[:limit]
    
    # Process each item
    for idx, (key, entry) in enumerate(items, 1):
        logger.info(f"Processing item {idx}/{len(items)}: {entry['label']}")
        
        # Build context
        context = build_context(key, knowledge_base)
        if not context:
            continue
            
        # Process entry
        entry_items = process_entry(entry, context)
        dataset.extend(entry_items)
    
    # Save dataset
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"Created dataset with {len(dataset)} entries")

def main():
    knowledge_base_path = "knowledge_base.json"
    output_path = "legal_instruction_dataset.json"
    limit = 5  # Process only first 5 items for testing
    create_dataset(knowledge_base_path, output_path, limit)

if __name__ == "__main__":
    main()
