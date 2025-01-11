import json
import os
import requests
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import RDFS, OWL, SKOS
from urllib.parse import urlparse
import time

def download_referenced_file(url, cache_dir="referenced_files"):
    """Download and cache referenced RDF files."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    # Create a safe filename from the URL
    parsed_url = urlparse(url)
    filename = os.path.join(cache_dir, parsed_url.netloc + parsed_url.path.replace('/', '_'))
    
    if not os.path.exists(filename):
        print(f"Downloading: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(response.content)
            time.sleep(1)  # Be nice to servers
        except Exception as e:
            print(f"Failed to download {url}: {str(e)}")
            return None
    
    return filename

def extract_graph_data(graph):
    """Extract all relationships and metadata from the graph into a JSON-serializable format."""
    graph_data = {
        "classes": {},
        "relationships": []
    }
    
    # Extract classes and their metadata
    for subject in graph.subjects(RDF.type, OWL.Class):
        class_info = {
            "uri": str(subject),
            "labels": [str(label) for label in graph.objects(subject, RDFS.label)],
            "definitions": [str(defn) for defn in graph.objects(subject, SKOS.definition)],
            "subClassOf": [str(parent) for parent in graph.objects(subject, RDFS.subClassOf)],
            "properties": {}
        }
        
        # Get all properties for this class
        for p, o in graph.predicate_objects(subject):
            pred_str = str(p)
            if pred_str not in [str(RDF.type), str(RDFS.label), str(SKOS.definition), str(RDFS.subClassOf)]:
                if pred_str not in class_info["properties"]:
                    class_info["properties"][pred_str] = []
                class_info["properties"][pred_str].append(str(o))
        
        graph_data["classes"][str(subject)] = class_info
    
    # Extract relationships
    for s, p, o in graph:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            graph_data["relationships"].append({
                "subject": str(s),
                "predicate": str(p),
                "object": str(o)
            })
    
    return graph_data

def process_owl_file(file_path):
    # Create main RDF graph
    main_graph = Graph()
    referenced_graphs = {}
    
    print(f"Parsing main file: {file_path}")
    main_graph.parse(file_path)
    
    # First pass: collect all referenced resources
    print("Collecting referenced resources...")
    for s, p, o in main_graph:
        if isinstance(o, URIRef) and any(x in str(p) for x in [RDFS.isDefinedBy, OWL.imports]):
            ref_file = download_referenced_file(str(o))
            if ref_file:
                try:
                    g = Graph()
                    g.parse(ref_file)
                    referenced_graphs[str(o)] = g
                    print(f"Parsed referenced file: {str(o)}")
                except Exception as e:
                    print(f"Failed to parse {str(o)}: {str(e)}")

    # Combine all graphs
    combined_graph = main_graph
    for g in referenced_graphs.values():
        combined_graph += g

    # Extract complete graph data
    graph_data = extract_graph_data(combined_graph)
    
    # Save graph data
    with open('graph_database.json', 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print("Graph database saved to graph_database.json")

    # Generate instruction dataset
    dataset = []
    processed_pairs = set()

    # Process relationships
    print("Processing relationships...")
    for s, p, o in combined_graph:
        if any(x in str(p) for x in [RDF.type, RDFS.subClassOf]):
            subj_label = None
            obj_label = None
            
            # Get labels and definitions
            for graph in [combined_graph] + list(referenced_graphs.values()):
                if not subj_label:
                    subj_labels = list(graph.objects(s, RDFS.label))
                    if subj_labels:
                        subj_label = str(subj_labels[0])
                if not obj_label:
                    obj_labels = list(graph.objects(o, RDFS.label))
                    if obj_labels:
                        obj_label = str(obj_labels[0])

            if not subj_label:
                subj_label = str(s).split('#')[-1].replace('_', ' ')
            if not obj_label:
                obj_label = str(o).split('#')[-1].replace('_', ' ')

            pair_key = f"{subj_label}|{obj_label}"
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            # Get definitions
            definition = None
            for graph in [combined_graph] + list(referenced_graphs.values()):
                definitions = list(graph.objects(s, SKOS.definition))
                if definitions:
                    definition = str(definitions[0])
                    break

            relation_type = "is a type of" if RDF.type in p else "is a subclass of"
            
            instruction = f"What is the relationship between {subj_label} and {obj_label}?"
            context = f"In the context of ontological relationships and classifications{', ' + definition if definition else ''}"
            response = f"{subj_label} {relation_type} {obj_label}"

            dataset.append({
                "instruction": instruction,
                "context": context,
                "response": response
            })

    return dataset

def save_dataset(dataset, output_file):
    """Save the dataset to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_file = "SOLI/SOLI.owl"
    output_file = "llama_instruct_dataset.json"
    
    print("Processing OWL file...")
    try:
        dataset = process_owl_file(input_file)
        print(f"Generated {len(dataset)} instruction-response pairs")
        
        print("Saving dataset...")
        save_dataset(dataset, output_file)
        print(f"Dataset saved to {output_file}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")