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
Ensure your Ollama instance is running with the llama3.1 model
Run the main script with your OWL file:

The script will:

Build a knowledge graph from the OWL file
Download and process referenced documents
Generate questions and answers
Save the results in Dolly/LLAMA instruction format
Output Files
knowledge_graph.json: Contains the processed ontology structure
dolly_dataset.json: Final instruction dataset in LLAMA format
owl_processing.log: Detailed processing log