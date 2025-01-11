#!/usr/bin/env python3
"""
Analyze the knowledge base file to get statistics about classes and their properties.
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, Any

def analyze_kb(kb_file: str) -> Dict[str, Any]:
    """Analyze the knowledge base file and return statistics."""
    with open(kb_file, 'r') as f:
        kb = json.load(f)
    
    stats = {
        "total_classes": len(kb["nodes"]),
        "classes_with_definition": 0,
        "classes_with_context": 0,
        "classes_with_parents": 0,
        "classes_with_children": 0,
        "classes_not_found": 0,
        "total_failed_urls": len(kb.get("failed_urls", [])),
        "total_downloaded_refs": len(kb.get("downloadedRefs", [])),
        "languages": defaultdict(int),
        "types": defaultdict(int)
    }
    
    for node_key, node in kb["nodes"].items():
        # Count classes by type
        node_type = node.get("type", "unknown")
        stats["types"][node_type] += 1
        
        # Count classes with definition
        if node.get("definition"):
            stats["classes_with_definition"] += 1
        
        # Count classes with context
        if node.get("context"):
            stats["classes_with_context"] += 1
        
        # Count classes with relationships
        if node.get("parents"):
            stats["classes_with_parents"] += 1
        if node.get("children"):
            stats["classes_with_children"] += 1
        
        # Count classes not found
        if node.get("class_not_found"):
            stats["classes_not_found"] += 1
        
        # Count languages
        for lang in node.get("languages", []):
            stats["languages"][lang] += 1
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Analyze knowledge base file")
    parser.add_argument("--input", default="test-kb.json", help="Input KB file")
    args = parser.parse_args()
    
    stats = analyze_kb(args.input)
    
    print("\nKnowledge Base Statistics:")
    print("=========================")
    print(f"Total Classes: {stats['total_classes']}")
    print(f"Classes with Definition: {stats['classes_with_definition']}")
    print(f"Classes with Context: {stats['classes_with_context']}")
    print(f"Classes with Parents: {stats['classes_with_parents']}")
    print(f"Classes with Children: {stats['classes_with_children']}")
    print(f"Classes Not Found: {stats['classes_not_found']}")
    print(f"Total Failed URLs: {stats['total_failed_urls']}")
    print(f"Total Downloaded References: {stats['total_downloaded_refs']}")
    
    print("\nClass Types:")
    for type_name, count in stats["types"].items():
        print(f"  {type_name}: {count}")
    
    print("\nLanguages:")
    for lang, count in stats["languages"].items():
        print(f"  {lang}: {count}")

if __name__ == "__main__":
    main()
