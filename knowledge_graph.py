# knowledge_graph.py

import logging
from rdflib import Graph, RDF, RDFS, SKOS, OWL

logger = logging.getLogger(__name__)

def build_knowledge_graph(owl_file):
    """
    Reads the OWL file into an RDFLib Graph, extracts basic info:
      - classes
      - properties
      - relationships
      - references (OWL imports, RDFS.isDefinedBy)
    Returns a dict.
    """
    logger.info(f"Building knowledge graph from {owl_file}")
    g = Graph()
    try:
        g.parse(owl_file)
        logger.info(f"Successfully parsed OWL file containing {len(g)} triples")
    except Exception as e:
        logger.error(f"Failed to parse OWL file: {str(e)}")
        raise

    graph_data = {
        "classes": {},
        "properties": {},
        "relationships": [],
        "refs": []
    }

    # Collect references
    refs = set()
    for s, p, o in g:
        # If p is an import or definition property
        if p in [RDFS.isDefinedBy, OWL.imports] and isinstance(o, str):
            refs.add(o)

    # Classes
    for c in g.subjects(RDF.type, OWL.Class):
        label = list(g.objects(c, RDFS.label))
        definition = list(g.objects(c, SKOS.definition))
        label_str = label[0] if label else str(c)
        def_str = definition[0] if definition else ""
        graph_data["classes"][str(c)] = {
            "uri": str(c),
            "label": str(label_str),
            "definition": str(def_str),
        }

    # Object Properties
    for op in g.subjects(RDF.type, OWL.ObjectProperty):
        label = list(g.objects(op, RDFS.label))
        definition = list(g.objects(op, SKOS.definition))
        label_str = label[0] if label else str(op)
        def_str = definition[0] if definition else ""
        graph_data["properties"][str(op)] = {
            "uri": str(op),
            "label": str(label_str),
            "definition": str(def_str),
        }

    # Relationships
    for s, p, o in g:
        # skip if s or o are strings
        if isinstance(s, str) or isinstance(o, str):
            continue
        graph_data["relationships"].append({
            "subject": str(s),
            "predicate": str(p),
            "object": str(o)
        })

    graph_data["refs"] = list(refs)
    return graph_data
