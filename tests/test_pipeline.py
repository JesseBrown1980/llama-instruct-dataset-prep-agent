import pytest
import json
import os
from unittest.mock import patch, mock_open
from build_knowledge import build_knowledge_base
from create_dataset import create_dataset, DatasetCreator

# Test data
TEST_OWL = """
<?xml version="1.0"?>
<rdf:RDF xmlns="http://www.semanticweb.org/test#"
     xml:base="http://www.semanticweb.org/test"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:xml="http://www.w3.org/XML/1998/namespace"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Class rdf:about="http://www.semanticweb.org/test#TestClass">
        <rdfs:label>Test Class</rdfs:label>
        <rdfs:comment>A test class for unit testing</rdfs:comment>
    </owl:Class>
</rdf:RDF>
"""

TEST_KB = {
    "nodes": {
        "TestClass": {
            "label": "Test Class",
            "description": "A test class for unit testing",
            "type": "class"
        }
    },
    "refs": [],
    "downloadedRefs": []
}

@pytest.fixture
def mock_files(tmp_path):
    # Create test files
    owl_file = tmp_path / "test.owl"
    kb_file = tmp_path / "kb.json"
    dataset_file = tmp_path / "dataset.jsonl"
    
    owl_file.write_text(TEST_OWL)
    kb_file.write_text(json.dumps(TEST_KB))
    
    return {
        "owl": str(owl_file),
        "kb": str(kb_file),
        "dataset": str(dataset_file)
    }

@patch('build_knowledge.build_knowledge_graph')
@patch('build_knowledge.download_and_clean_resource')
def test_build_knowledge_base(mock_download, mock_build, mock_files):
    # Setup
    mock_build.return_value = None
    mock_download.return_value = {"text": "Test reference"}
    
    # Test
    build_knowledge_base(
        owl_input=mock_files["owl"],
        output=mock_files["kb"],
        limit="all"
    )
    
    # Verify
    mock_build.assert_called_once()
    assert os.path.exists(mock_files["kb"])
    
    with open(mock_files["kb"]) as f:
        kb = json.load(f)
        assert "nodes" in kb
        assert "refs" in kb
        assert "downloadedRefs" in kb

@patch('create_dataset.questioner_agent_invoke')
@patch('create_dataset.question_checker_agent_invoke')
@patch('create_dataset.create_dataset_entry')
def test_dataset_creator(mock_entry, mock_checker, mock_questioner, mock_files):
    # Setup
    mock_questioner.return_value = json.dumps(["Test question?"])
    mock_checker.return_value = json.dumps(["Test question?"])
    mock_entry.return_value = json.dumps({
        "instruction": "Test instruction",
        "context": "Test context",
        "response": "Test response"
    })
    
    # Test
    creator = DatasetCreator(mock_files["dataset"])
    creator.process_node({
        "label": "Test",
        "description": "Test description"
    })
    
    # Verify
    assert os.path.exists(mock_files["dataset"])
    with open(mock_files["dataset"]) as f:
        lines = f.readlines()
        assert len(lines) > 0
        entry = json.loads(lines[0])
        assert "instruction" in entry
        assert "context" in entry
        assert "response" in entry

@patch('create_dataset.DatasetCreator')
def test_create_dataset(mock_creator, mock_files):
    # Setup
    mock_instance = MagicMock()
    mock_creator.return_value = mock_instance
    
    # Test
    create_dataset(
        kb_path=mock_files["kb"],
        output=mock_files["dataset"],
        limit="all"
    )
    
    # Verify
    mock_creator.assert_called_once_with(mock_files["dataset"])
    mock_instance.process_node.assert_called()
