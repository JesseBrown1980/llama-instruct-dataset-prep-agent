import pytest
import json
from unittest.mock import patch, MagicMock
from agents import (
    questioner_agent_invoke,
    question_checker_agent_invoke,
    maker_agent,
    make_instruction_agent,
    cleaner_agent,
    create_dataset_entry
)

# Test data
TEST_CONTEXT = """
The Legal Entity Identifier (LEI) is a 20-character, alpha-numeric code that connects to key reference information 
that enables clear and unique identification of legal entities participating in financial transactions.
"""

TEST_QUESTION = "What is the length of a Legal Entity Identifier (LEI)?"
TEST_INSTRUCTION = "Explain the length and format of a Legal Entity Identifier (LEI)."
TEST_RESPONSE = "A Legal Entity Identifier (LEI) is 20 characters long and uses an alpha-numeric format."

@pytest.fixture
def mock_ollama_response():
    return {
        "response": TEST_RESPONSE,
        "eval_count": 100,
        "eval_duration": 1000000
    }

@patch('agents.ollama_invoke')
def test_questioner_agent(mock_ollama):
    # Setup mock
    mock_ollama.return_value = json.dumps([
        "What is the length of a Legal Entity Identifier (LEI)?",
        "What format does an LEI use?"
    ])
    
    # Test
    result = questioner_agent_invoke(TEST_CONTEXT)
    questions = json.loads(result)
    
    # Verify
    assert isinstance(questions, list)
    assert len(questions) > 0
    assert all(isinstance(q, str) for q in questions)
    assert any("length" in q.lower() for q in questions)

@patch('agents.ollama_invoke')
def test_question_checker(mock_ollama):
    # Setup
    test_questions = json.dumps([
        {"question": "What is the length of an LEI?"},
        {"question": "How long is an LEI code?"}  # Similar question
    ])
    mock_ollama.return_value = json.dumps(["What is the length of an LEI?"])
    
    # Test
    result = question_checker_agent_invoke(test_questions)
    cleaned = json.loads(result)
    
    # Verify
    assert isinstance(cleaned, list)
    assert len(cleaned) == 1  # Should remove duplicate
    assert isinstance(cleaned[0], str)

@patch('agents.ollama_invoke')
def test_maker_agent(mock_ollama, mock_ollama_response):
    # Setup
    mock_ollama.return_value = TEST_RESPONSE
    
    # Test
    result = maker_agent(TEST_QUESTION, TEST_CONTEXT)
    
    # Verify
    assert isinstance(result, str)
    assert "20" in result
    assert "alpha-numeric" in result.lower()

@patch('agents.ollama_invoke')
def test_instruction_agent(mock_ollama, mock_ollama_response):
    # Setup
    mock_ollama.return_value = TEST_INSTRUCTION
    
    # Test
    result = make_instruction_agent(TEST_QUESTION, TEST_CONTEXT)
    
    # Verify
    assert isinstance(result, str)
    assert "explain" in result.lower()
    assert "LEI" in result

@patch('agents.ollama_invoke')
def test_cleaner_agent(mock_ollama):
    # Setup
    mock_ollama.return_value = TEST_CONTEXT.strip()
    
    # Test
    result = cleaner_agent(TEST_CONTEXT)
    
    # Verify
    assert isinstance(result, str)
    assert "LEI" in result
    assert len(result) <= len(TEST_CONTEXT)

@patch('agents.make_instruction_agent')
@patch('agents.maker_agent')
def test_create_dataset_entry(mock_maker, mock_instruction):
    # Setup
    mock_instruction.return_value = TEST_INSTRUCTION
    mock_maker.return_value = TEST_RESPONSE
    
    # Test
    result = create_dataset_entry(TEST_QUESTION, TEST_CONTEXT)
    entry = json.loads(result)
    
    # Verify
    assert isinstance(entry, dict)
    assert "instruction" in entry
    assert "response" in entry
    assert "context" in entry
    assert entry["instruction"] == TEST_INSTRUCTION
    assert entry["response"] == TEST_RESPONSE
    assert entry["context"] == TEST_CONTEXT
