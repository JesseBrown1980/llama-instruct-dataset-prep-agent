# LLaMA Instruction Dataset Preparation Agent

An agent-based system for preparing instruction datasets from knowledge bases.

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

## Monitoring

- Check `dataset_creation.log` for detailed processing logs
- Monitor the `data/` directory for real-time output
- Use `analyze_kb.py` to generate statistics about the knowledge base