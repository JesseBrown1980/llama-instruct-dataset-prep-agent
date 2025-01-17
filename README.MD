# LLaMA Instruction Dataset Preparation Agent

An agent-based system for preparing instruction datasets from knowledge bases.

## Philosophy

This project is built on several key principles:

1. **Resilience First**: The system is designed to be resilient against failures at every step. Rather than failing fast, it employs intelligent retry mechanisms with exponential backoff to maximize the chances of successful dataset entry creation.

2. **Quality Over Quantity**: Each dataset entry goes through multiple validation stages:
   - Question validation before instruction creation
   - Response quality checking
   - Format validation at multiple stages
   - Final comprehensive validation of the complete entry

3. **Contextual Awareness**: The system maintains context from parent and child nodes when processing knowledge base entries, ensuring that generated questions and responses have sufficient context to be meaningful.

4. **Traceability**: Every step of the process is logged and tracked:
   - Unique IDs for each entry
   - Detailed logging of attempts and retries
   - Metadata preservation throughout the pipeline
   - Complete audit trail in log files

## System Flow

The dataset preparation follows a carefully orchestrated flow:

### 1. Knowledge Base Traversal
```
Root Node
├── Get main context
├── Collect parent context
└── Collect child context
```

### 2. Context Processing
```
Combined Context
├── Check length (>2000 words?)
│   ├── Yes -> Split into chunks with overlap
│   └── No  -> Process as single chunk
└── Log to chunks.jsonl
```

### 3. Dataset Entry Creation
Each chunk goes through a multi-stage pipeline:

```
Question Generation
└── Instruction Creation
    └── Response Generation
        └── Quality Check
            └── Final Formatting
```

Each stage employs retry-until-success with exponential backoff:
- Initial retry delay: 1 second
- Exponential increase up to 15 seconds
- Maximum total retry time: 60 seconds per stage

### 4. Validation Layers

```
Entry Validation
├── Content Validation
│   ├── Minimum length checks
│   ├── Format validation
│   └── Content quality checks
├── Structure Validation
│   ├── Required fields
│   ├── JSON structure
│   └── Nested object checks
└── Metadata Validation
    ├── Entry ID
    ├── Node references
    └── Timestamps
```

### 5. File Management

The system maintains several key files:
- `chunks.jsonl`: Raw context chunks
- `questions.jsonl`: Generated questions
- `dataset.jsonl`: Final dataset entries
- `nodetraversal.jsonl`: Processing metadata
- `token_cost.json`: API usage tracking

## Error Handling Philosophy

The system follows a "never give up" approach to error handling:
1. Every operation that could fail is wrapped in retry logic
2. Retries use exponential backoff to avoid overwhelming the system
3. Each retry attempt is logged for debugging
4. Only after exhausting all retries (60 second timeout) will an operation fail
5. Failures are logged with detailed context for post-mortem analysis

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llama-instruct-dataset-prep-agent.git
cd llama-instruct-dataset-prep-agent
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Ollama server and ensure model is available:
```bash
ollama run llama3:8b
```

## Running the Pipeline

### Full Pipeline
Run both knowledge base creation and dataset generation:
```bash
python main.py --owl-input input.owl --kb-output kb.json --dataset-output dataset.json --kb-limit [number|all] --dataset-limit [number|all]
```
Arguments:
- `--owl-input`: Path to input OWL file
- `--kb-output`: Path to save knowledge base JSON
- `--dataset-output`: Path to save final dataset
- `--kb-limit`: Number of classes/properties to process for knowledge base or 'all'
- `--dataset-limit`: Number of nodes to process for dataset creation or 'all'

Examples:
```bash
# Process everything
python main.py --owl-input input.owl --kb-output kb.json --dataset-output dataset.json --kb-limit all --dataset-limit all

# Process 100 classes but only generate dataset from 10 nodes
python main.py --owl-input input.owl --kb-output kb.json --dataset-output dataset.json --kb-limit 100 --dataset-limit 10

# Process all classes but limit dataset generation
python main.py --owl-input input.owl --kb-output kb.json --dataset-output dataset.json --kb-limit all --dataset-limit 50
```

This will:
1. Create knowledge base from OWL file (using kb-limit)
2. Generate instruction dataset from knowledge base (using dataset-limit)
3. Write outputs to specified files

### Individual Components

1. Build Knowledge Base:
```bash
python build_knowledge.py --input path/to/input/files --output knowledge_base.json
```
Arguments:
- `--input`: Directory containing input files
- `--output`: Path to save knowledge base JSON
- `--limit`: (Optional) Limit number of files to process

2. Analyze Knowledge Base:
```bash
python analyze_kb.py --kb-path knowledge_base.json --output analysis.json
```
Arguments:
- `--kb-path`: Path to knowledge base JSON
- `--output`: Path to save analysis results
- `--visualize`: (Optional) Generate visualizations

3. Create Dataset:
```bash
python create_dataset.py --kb-path knowledge_base.json --output dataset.json --limit [number|all]
```
Arguments:
- `--kb-path`: Path to knowledge base JSON
- `--output`: Path to save dataset
- `--limit`: Number of nodes to process or 'all' for entire knowledge base

4. Create Dataset from Specific Node:
```bash
python create_dataset_from_node.py full-kb.json dataset.jsonl RCBP9N5ER7Y6inNQdjggrKt --limit all
```
Arguments:
- First argument: Path to knowledge base JSON file
- Second argument: Path to output dataset file
- Third argument: Node ID to start traversal from
- `--limit`: Number of nodes to process or 'all' for entire knowledge base

The dataset creation process:
1. Generates multiple questions for each chunk of text
2. Validates questions for quality and uniqueness
3. Creates instruction-response pairs for each valid question
4. Writes entries immediately to dataset.jsonl

Each dataset entry contains:
```json
{
    "instruction": "Clear instruction based on question",
    "context": "Original text chunk",
    "response": "Direct answer based on context"
}
```

## Output Files

The pipeline generates several files:
- `data/chunks.jsonl`: Individual text chunks with their node references
- `data/questions.jsonl`: Generated questions for each chunk
- `data/dataset.jsonl`: Final instruction dataset entries
- `class_counter.txt`: Statistics about processed classes
- `token_cost.json`: Token usage tracking

## Testing

### Setup Test Environment
```bash
# Create virtual environment if not already done
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows

# Install test dependencies
pip install -r tests/requirements.txt
```

### Run Tests
```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=./ --cov-report=term-missing

# Run specific test file
pytest tests/test_agents.py
pytest tests/test_pipeline.py

# Run specific test function
pytest tests/test_agents.py::test_questioner_agent
```

### Test Coverage
The test suite covers:
1. Agent Functions
   - Question generation and validation
   - Instruction and response creation
   - Text cleaning and processing
   
2. Pipeline Components
   - Knowledge base creation from OWL
   - Dataset generation from knowledge base
   - End-to-end pipeline integration

### Writing New Tests
Add new tests to:
- `tests/test_agents.py` for agent functionality
- `tests/test_pipeline.py` for pipeline components
- `tests/conftest.py` for shared fixtures and setup

## Monitoring

- Check `dataset_creation.log` for detailed processing logs
- Monitor the `data/` directory for real-time output
- Use `analyze_kb.py` to generate statistics about the knowledge base