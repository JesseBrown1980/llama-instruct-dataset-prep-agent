# Next Steps for Development

## 1. Multi-Format Input Support
- Add parsers for different data formats:
  - JSON/JSONL structured data
  - CSV/Excel spreadsheets
  - Markdown/Text documents
  - PDF documents with structured data
  - XML/YAML configurations
- Create a unified parser interface:
  ```python
  class DataParser:
      def parse(self, input_path: str) -> KnowledgeBase:
          pass
  ```
- Implement format detection and automatic parser selection

## 2. Flexible Knowledge Base Structure
- Make KB structure configurable via JSON schema:
  ```json
  {
    "structure": {
      "nodes": {
        "type": "object",
        "properties": {
          "id": "string",
          "content": "string",
          "metadata": "object"
        }
      },
      "relationships": {
        "type": "array",
        "items": {
          "source": "string",
          "target": "string",
          "type": "string"
        }
      }
    }
  }
  ```
- Support different KB strategies:
  - Graph-based (current)
  - Document-based (for text chunks)
  - Key-value pairs
  - Hierarchical structures
  - Vector embeddings
- Allow custom indexing and querying methods

## 3. Multiple Dataset Formats
### Dolly Format (Current)
```json
{
  "instruction": "...",
  "context": "...",
  "response": "..."
}
```

### Raw Text Format
```json
{
  "text": "...",
  "metadata": {
    "source": "...",
    "category": "...",
    "tags": []
  }
}
```

- Add format converters
- Support custom output templates
- Enable batch processing modes

## 4. API and Authentication
- Implement FastAPI backend:
  ```python
  from fastapi import FastAPI, Security
  from fastapi.security import OAuth2PasswordBearer
  
  app = FastAPI()
  
  @app.post("/api/v1/dataset")
  async def create_dataset(
      input_file: UploadFile,
      config: DatasetConfig,
      current_user: User = Security(get_current_user)
  ):
      pass
  ```
- Add authentication:
  - OAuth2 with JWT
  - API key support
  - Role-based access control
- Implement rate limiting and quotas

## 5. Configurable Question Generation
- Make prompts configurable:
  ```yaml
  prompts:
    question_generation:
      template: |
        Given the following context, generate {num_questions} questions that:
        {criteria}
        
        Context: {context}
      parameters:
        num_questions: 5
        criteria:
          - "Are specific and focused"
          - "Cover key concepts"
          - "Vary in complexity"
  ```
- Support different question types:
  - Multiple choice
  - Open-ended
  - True/False
  - Fill in the blanks
- Allow custom validation rules

## 6. Containerization and Deployment
### Docker Setup
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy application
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Setup volume for data persistence
VOLUME /app/data

# Start services
CMD ["./start.sh"]
```

### Deployment Requirements
- GPU-enabled instance (e.g., AWS g4dn.xlarge)
- Persistent storage for:
  - Knowledge bases
  - Generated datasets
  - Model caches
- Environment variables for configuration
- Health monitoring and logging

### Docker Compose for Development
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MAX_MEMORY=16G
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Implementation Priority
1. Multi-Format Input Support
2. API and Authentication
3. Containerization
4. Configurable Question Generation
5. Flexible KB Structure
6. Multiple Dataset Formats

Each feature should include:
- Unit tests
- Documentation
- Example configurations
- Performance benchmarks
- Migration guides
