# LLAMA Instruct Dataset Preparation Agent

A Python-based tool for converting OWL/RDF ontologies into instruction-following datasets suitable for training Large Language Models (LLMs).

## Features

- Converts OWL/RDF ontologies into structured Q&A pairs
- Downloads and processes referenced documents (PDF, HTML, RDF)
- Uses local Ollama endpoints for generating questions and answers
- Supports caching of downloaded resources
- Comprehensive logging system

## Requirements

- Python 3.7+
- Local Ollama instance running with llama3.1 model
- Required Python packages (see [requirements.txt](cci:7://file:///c:/Users/Fairweather/Documents/GitHub/llama-instruct-dataset-prep-agent/requirements.txt:0:0-0:0))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llama-instruct-dataset-prep-agent.git
cd llama-instruct-dataset-prep-agent

### 2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install dependencies:
pip install -r requirements.txt

### 4. Usage 
The dataset creation is split into two steps:

1. Build the Knowledge Base:
```bash
python build_knowledge.py
```
This will:
- Build a knowledge graph from the OWL file
- Download and clean referenced documents using AI agents
- Save the knowledge base to `knowledge_base.json`

2. Create the Instruction Dataset:
```bash
python create_dataset.py
```
This will:
- Read the knowledge base
- Use AI agents to:
  - Generate relevant questions
  - Create detailed answers
  - Format and validate the output
- Save the results in LLAMA instruction format

### Output Files

- `knowledge_base.json`: Contains the processed ontology structure with cleaned references
- `legal_instruction_dataset.json`: Final instruction dataset in LLAMA format
- `owl_processing.log`: Knowledge base creation log
- `dataset_creation.log`: Dataset creation log

By default, the dataset creation process is limited to 5 items for testing. To process more items, modify the `limit` parameter in `create_dataset.py`.