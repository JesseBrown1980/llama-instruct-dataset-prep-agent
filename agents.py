import platform
import os
import socket
import subprocess
import requests
import json
import logging
import time
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

# Initialize token cost tracking
_token_cost = {
    "datetime": datetime.now().isoformat(),
    "total_eval_count": 0,
    "total_eval_duration_ns": 0,
    "tasks": {}
}

# Semaphore to limit concurrent Ollama requests
OLLAMA_SEMAPHORE = threading.Semaphore(1)  # Only allow 1 request at a time

def get_machine_ip():
    """Get the current machine's IP address"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable, just used to get local IP
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.1'
    finally:
        s.close()
    return ip

def get_wsl_ip():
    """Get the WSL IP address"""
    try:
        result = subprocess.run(
            ['wsl.exe', 'hostname', '-I'], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split()[0]
    except:
        pass
    return None

def start_ollama(ip):
    """Start Ollama in Ubuntu-22.04 with the specified IP and ensure model is available"""
    logger.info(f"Starting Ollama in Ubuntu-22.04 on {ip}...")
    try:
        # Kill any existing Ollama processes in WSL
        subprocess.run(
            ['wsl.exe', '-d', 'Ubuntu-22.04', '--', 'sudo', 'killall', '-9', 'ollama'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for process cleanup
        time.sleep(1)
        
        # Run ollama serve in Ubuntu-22.04 with public IP
        subprocess.Popen(
            ['wsl.exe', '-d', 'Ubuntu-22.04', '--', f'OLLAMA_HOST={ip}:11434', 'ollama', 'serve'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start (max 30 seconds)
        url = f"http://{ip}:11434/api/version"
        for _ in range(30):
            try:
                resp = requests.get(url)
                if resp.status_code == 200:
                    logger.info("Successfully started Ollama")
                    return True
            except:
                time.sleep(1)
                continue
        
        logger.error("Timed out waiting for Ollama to start")
        return False
        
    except Exception as e:
        logger.error(f"Failed to start Ollama: {str(e)}")
        return False

def ensure_ollama_running():
    """Check if Ollama is running and return the correct IP to use"""
    # First try localhost
    try:
        resp = requests.get("http://localhost:11434/api/version")
        if resp.status_code == 200:
            logger.info("Successfully connected to Ollama on localhost")
            return True, "localhost"
    except:
        pass

    # If localhost fails, try WSL
    wsl_ip = get_wsl_ip()
    if wsl_ip:
        try:
            resp = requests.get(f"http://{wsl_ip}:11434/api/version")
            if resp.status_code == 200:
                logger.info(f"Successfully connected to Ollama on WSL ({wsl_ip})")
                return True, wsl_ip
        except:
            pass

    logger.error("Could not connect to Ollama. Please ensure:")
    logger.error("1. Ollama is installed and running (ollama serve)")
    logger.error("2. Port 11434 is available")
    return False, "localhost"

def ollama_invoke(prompt, model="llama2", temperature=0.8, format_spec="text", agent_name="unknown", max_retries=3):
    """
    Synchronously invoke Ollama API and wait for response.
    Returns only after request is complete.
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # Ensure we wait for complete response
        "temperature": temperature
    }

    retry_count = 0
    while retry_count < max_retries:
        try:
            logger.info(f"Making Ollama request for {agent_name}")
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 500:
                retry_count += 1
                logger.warning(f"Ollama server error (attempt {retry_count}/{max_retries})")
                if retry_count == max_retries:
                    raise Exception(f"Ollama server failed after {max_retries} attempts")
                continue
                
            response.raise_for_status()
            result = response.json()
            
            # Ensure we have a complete response
            if 'response' not in result:
                raise Exception("Incomplete response from Ollama")
            
            if format_spec == "json":
                try:
                    text = result['response']
                    # Find JSON-like content
                    start = text.find('[')
                    end = text.rfind(']') + 1
                    if start >= 0 and end > start:
                        json_str = text[start:end]
                        # Validate JSON before returning
                        json.loads(json_str)  # This will raise if invalid
                        logger.info(f"Successfully completed {agent_name} request")
                        return json_str
                    return text
                except Exception as e:
                    logger.error(f"JSON extraction failed: {str(e)}")
                    raise
            
            logger.info(f"Successfully completed {agent_name} request")
            return result['response']
            
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                logger.error(f"Failed to complete Ollama request: {str(e)}")
                raise
            logger.warning(f"Retrying request (attempt {retry_count}/{max_retries})")
            continue

def log_token_cost(agent_name, eval_count, eval_duration_ns):
    """Log token cost metrics to token_cost.json"""
    global _token_cost
    
    # Update task-specific stats
    if agent_name not in _token_cost["tasks"]:
        _token_cost["tasks"][agent_name] = {
            "eval_count": 0,
            "eval_duration_ns": 0
        }
    
    _token_cost["tasks"][agent_name]["eval_count"] += eval_count
    _token_cost["tasks"][agent_name]["eval_duration_ns"] += eval_duration_ns
    
    # Update totals
    _token_cost["total_eval_count"] += eval_count
    _token_cost["total_eval_duration_ns"] += eval_duration_ns
    
    # Write accumulated stats to file
    with open('token_cost.json', 'w') as f:
        json.dump(_token_cost, f, indent=4)

def chunk_text_with_overlap(text: str, chunk_size: int = 2000, overlap: int = 50) -> List[str]:
    """Split text into chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        # If we're not at the start, include overlap from previous chunk
        if start > 0:
            chunk_start = max(0, start - overlap)
        else:
            chunk_start = 0
            
        # Calculate end of this chunk
        chunk_end = min(start + chunk_size, len(words))
        
        # Create chunk
        chunk = ' '.join(words[chunk_start:chunk_end])
        chunks.append(chunk)
        
        # Move to next chunk
        start = chunk_end
        
        # If we've processed all words, break
        if chunk_end >= len(words):
            break
            
    return chunks

def cleaner_agent(raw_text, context=""):
    """
    Splits raw_text into ~4000-word chunks if needed.
    Each chunk is posted to Ollama for summarization/cleaning.
    Only relevant info is returned. No pleasantries.
    """
    words = raw_text.split()
    cleaned_result = []
    chunk_size = 4000 - len(context.split())
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        prompt = f"""
You are a cleaning agent. Keep only essential information relevant to this context:
{context}
Raw text:
{chunk}
Return cleaned text. No pleasantries.
"""
        partial_cleaned = ollama_invoke(prompt, model="llama3:8b", temperature=0.0, agent_name="cleaner_agent")
        cleaned_result.append(partial_cleaned)
        start = end
    return "\n".join(cleaned_result).strip()

def format_questions_to_json(questions_text):
    """Convert free-form questions text into JSON array."""
    prompt = f"""Convert these questions into a clean JSON array of strings. Remove any explanations, numbering, or extra text. Just return a JSON array of the actual questions:

{questions_text}

Example output format:
["What is X?", "How does Y work?", "When was Z implemented?"]
"""
    return ollama_invoke(prompt, model="llama3:8b", temperature=0.1,
                        format_spec="json", agent_name="question_formatter")

def questioner_agent_invoke(knowledge_context):
    """
    Generate questions from knowledge context.
    Returns a JSON array of strings (the questions).
    """
    # If context is too large, chunk it first
    MAX_CONTEXT_LENGTH = 4000  # characters
    if len(knowledge_context) > MAX_CONTEXT_LENGTH:
        chunks = chunk_text_with_overlap(knowledge_context, chunk_size=MAX_CONTEXT_LENGTH//4)  # words
        all_questions = set()
        
        for chunk in chunks:
            chunk_prompt = f"""You are an expert questioner. Generate 2-3 specific, detailed questions about this part of the knowledge:

{chunk}

Questions must:
- Be answerable from just this context
- Be specific and detailed
- Focus on key information

Generate questions in numbered format:
1. [Question]
2. [Question]
etc."""
            
            try:
                questions_text = ollama_invoke(chunk_prompt, model="llama3:8b", temperature=0.7,
                                            format_spec="text", agent_name="questioner_agent")
                
                # Format this chunk's questions
                chunk_questions = process_chunk_with_retry(questions_text)
                all_questions.update(chunk_questions)
                
            except Exception as e:
                logger.error(f"Failed to process knowledge chunk: {str(e)}")
                continue
        
        return json.dumps(list(all_questions))
    
    # For smaller contexts, process normally
    prompt = f"""You are an expert questioner. Your task is to generate high-quality, diverse questions from the given knowledge.

Generate 5-7 questions that:
- Focus ONLY on information present in the context
- Are specific and detailed
- Cover different aspects of the knowledge
- Mix factual, conceptual, and analytical questions
- Are clear and well-formed
- Can be answered using only the given context

Here is the knowledge:
{knowledge_context}

Generate questions in this format:
1. [Your first question]
2. [Your second question]
...and so on

Remember: Only ask questions that can be answered using the context provided."""

    # Get free-form questions
    questions_text = ollama_invoke(prompt, model="llama3:8b", temperature=0.7,
                                 format_spec="text", agent_name="questioner_agent")
    
    # Process and return questions
    questions = process_chunk_with_retry(questions_text)
    return json.dumps(list(questions))

def process_chunk_with_retry(chunk, max_retries=3):
    """Process a chunk with retries on failure."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            chunk_questions_json = format_questions_to_json(chunk)
            return json.loads(chunk_questions_json)
        except (json.JSONDecodeError, Exception) as e:
            retry_count += 1
            if retry_count == max_retries:
                logger.error(f"Failed to process chunk after {max_retries} retries: {str(e)}")
                raise
            logger.warning(f"Retry {retry_count}/{max_retries} for chunk processing")
            # Immediate retry, no wait

def question_checker_agent_invoke(question_list_json):
    """
    Checks a JSON array of questions for redundancy or errors.
    Returns corrected JSON array if needed. No pleasantries.
    """
    prompt = f"""
You are a question checker agent. Check these questions for redundancy or errors.
Questions:
{question_list_json}

"""
    return ollama_invoke(prompt, model="llama3:8b", temperature=0.0, 
                        format_spec="json", agent_name="question_checker_agent")

def maker_agent(question, knowledge_context):
    """
    Produces a concise answer based on knowledge_context for a single question.
    """
    prompt = f"""
You are a maker agent. Answer this question using only this knowledge:
{knowledge_context}
Question:
{question}
No pleasantries.
"""
    return ollama_invoke(prompt, model="llama3:8b", temperature=0.2, agent_name="maker_agent")

def formatter_agent(raw_answer, question, context):
    """
    Converts raw_answer into Dolly/LLAMA style JSON with format specification
    """
    prompt = f"""
You are a formatter agent. Convert this into Dolly/LLAMA JSON format:
Question: {question}
Answer: {raw_answer}
Context: {context}
"""
    format_spec = {
        "type": "object",
        "properties": {
            "instruction": {"type": "string"},
            "context": {"type": "string"},
            "response": {"type": "string"}
        },
        "required": ["instruction", "context", "response"]
    }
    
    return ollama_invoke(prompt, model="llama3:8b", temperature=0.0, 
                        format_spec=format_spec, agent_name="formatter_agent")

def json_validator_agent(json_str: str) -> str:
    """
    Validates and ensures the input is proper JSON format.
    Returns cleaned JSON string or raises error.
    """
    prompt = f"""
You are a JSON validator. Ensure this is valid JSON with instruction/context/response fields.
If not valid, fix it. Return ONLY the valid JSON, no other text:
{json_str}

Requirements:
1. Must be valid JSON
2. Must have instruction, context, response fields
3. No other fields allowed
4. No metadata or extra info
5. Return ONLY the JSON
"""
    return ollama_invoke(prompt, model="llama3:8b", temperature=0.0,
                        format_spec="json", agent_name="json_validator_agent")

def checker_agent(draft_json: str) -> str:
    """
    Checks logical consistency of the final Dolly JSON and returns valid JSON if needed.
    """
    prompt = f"""
You are a checker agent. Check this JSON for logical consistency such that the data is in strict json, the responses have no pleasantries and are completely concise and the response is valid for the given instruction and context. If anything does not match up, fix it and return strict json only:
{draft_json}
Return corrected JSON if needed. No other text.
"""
    response = ollama_invoke(prompt, model="llama3:8b", temperature=0.0, 
                         format_spec="json", agent_name="checker_agent")
    return json_validator_agent(response)

def check_windows_ollama():
    """Check if Ollama is running on Windows and error out if it is"""
    try:
        resp = requests.get("http://localhost:11434/api/version")
        if resp.status_code == 200:
            logger.error("ERROR: Ollama is running on Windows!")
            logger.error("This application requires Ollama to run in WSL/Ubuntu.")
            logger.error("Please:")
            logger.error("1. Stop Ollama on Windows")
            logger.error("2. Install and run Ollama in WSL instead")
            logger.error("3. Run 'ollama serve' in WSL terminal")
            raise RuntimeError("Ollama must run in WSL/Ubuntu, not Windows")
    except requests.exceptions.ConnectionError:
        # This is good - means Ollama is not running on Windows
        pass

def make_instruction_agent(question: str, context: str) -> str:
    """
    Converts a question into a clear instruction for the model.
    Returns only the instruction text, no JSON formatting.
    """
    prompt = f"""
You are an instruction creator. Convert this question into a clear instruction. A sample instruction is: Define what X means in the context of Y , What should you do for X in the context of Y:
Context: {context}
Question: {question}

Requirements:
1. Return ONLY the instruction text
2. Make it clear and actionable
3. No JSON formatting
4. No prefixes like 'Task:' or 'Instruction:'
5. No quotes or special characters
6. Must be relevant to the context
"""
    return ollama_invoke(prompt, model="llama3:8b", temperature=0.1, 
                        agent_name="make_instruction_agent")
