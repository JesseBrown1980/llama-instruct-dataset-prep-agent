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
            
            text = result['response'].strip()
            
            if format_spec == "json":
                try:
                    # Find the first [ and last ]
                    start = text.find('[')
                    end = text.rfind(']') + 1
                    
                    if start >= 0 and end > start:
                        json_str = text[start:end]
                        # Validate JSON before returning
                        json.loads(json_str)  # This will raise if invalid
                        logger.info(f"Successfully completed {agent_name} request")
                        return json_str
                        
                    logger.error(f"No JSON array found in response: {text}")
                    return "[]"  # Return empty array as fallback
                    
                except Exception as e:
                    logger.error(f"JSON extraction failed: {str(e)}")
                    logger.error(f"Raw response: {text}")
                    return "[]"  # Return empty array as fallback
            
            logger.info(f"Successfully completed {agent_name} request")
            return text
            
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                logger.error(f"Failed to complete Ollama request: {str(e)}")
                if format_spec == "json":
                    return "[]"  # Return empty array as fallback
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

def question_checker_agent_invoke(questions_jsonl: str) -> str:
    """
    Check a list of questions for quality and redundancy.
    Input is JSONL format with questions. Returns a cleaned JSON array of questions.
    """
    # Extract just the questions from JSONL
    questions = []
    try:
        for line in questions_jsonl.split('\n'):
            if line.strip():
                q_obj = json.loads(line)
                if 'question' in q_obj:
                    questions.append(q_obj['question'])
    except Exception as e:
        logger.error(f"Failed to parse questions JSONL: {str(e)}")
        logger.error(f"Input JSONL: {questions_jsonl}")
        return "[]"

    if not questions:
        logger.error("No questions found in JSONL input")
        return "[]"

    logger.info(f"Checking {len(questions)} questions")
    prompt = f"""You are a question checker agent. Return ALL questions that are not completely invalid.
Keep ALL questions unless they are:
1. Not actually questions (statements, fragments, etc.)
2. Completely nonsensical or unrelated to context
3. Exact duplicates of other questions

Even if a question is:
- Similar to another (but not exact duplicate)
- Basic or simple
- Could be worded better
STILL KEEP IT - we will improve it later.

Here are the questions to check:
{json.dumps(questions, indent=2)}

Return ONLY a JSON array of the valid questions, like this:
["Question 1", "Question 2"]

DO NOT explain or add any other text. ONLY return the JSON array."""

    result = ollama_invoke(prompt, model="llama3:8b", temperature=0.0,
                        format_spec="json", agent_name="question_checker_agent")
    
    # Ensure we have valid JSON array
    try:
        cleaned_questions = json.loads(result)
        if not isinstance(cleaned_questions, list):
            logger.error(f"Question checker returned non-array JSON: {result}")
            return json.dumps(questions)  # Return original questions
        
        if not cleaned_questions:
            logger.warning("Question checker returned empty array")
            return json.dumps(questions)  # Return original questions
            
        logger.info(f"Kept {len(cleaned_questions)} questions after checking")
        return json.dumps(cleaned_questions)
    except Exception as e:
        logger.error(f"Failed to parse question checker output: {str(e)}")
        logger.error(f"Raw output: {result}")
        return json.dumps(questions)  # Return original questions

def maker_agent(question: str, context: str, max_retries: int = 3) -> str:
    """
    Produces a concise answer based on knowledge_context for a single question.
    Lower temperature (0.2) for more accurate, context-focused responses.
    Will retry up to max_retries times if generation fails.
    """
    for attempt in range(max_retries):
        try:
            prompt = f"""Answer this question based ONLY on the provided context.
Be direct, specific, and comprehensive.

Context: {context}

Question: {question}

Return ONLY your answer, no other text."""

            answer = ollama_invoke(prompt, model="llama3:8b", temperature=0.2 + (attempt * 0.1),
                              format_spec="text", agent_name="answer_maker")
            result = answer.strip().strip('"')
            if result:
                return result
                
            logger.warning(f"Empty response generated on attempt {attempt + 1}/{max_retries}")
        except Exception as e:
            logger.warning(f"Response generation failed on attempt {attempt + 1}/{max_retries}: {str(e)}")
            
    # If all retries failed, extract relevant info from context
    context_words = context.split()
    question_words = set(question.lower().split())
    relevant_parts = []
    
    for i in range(len(context_words)):
        window = ' '.join(context_words[max(0, i-5):min(len(context_words), i+6)])
        if any(word in window.lower() for word in question_words):
            relevant_parts.append(window)
            
    if relevant_parts:
        return ' '.join(relevant_parts)
    else:
        return "Based on the provided context, a specific answer could not be generated."

def make_instruction_agent(question: str, context: str, max_retries: int = 3) -> str:
    """
    Converts a question into a clear instruction for the model.
    Higher temperature (0.7) for more creative instruction generation.
    Will retry up to max_retries times if generation fails.
    """
    for attempt in range(max_retries):
        try:
            prompt = f"""Convert this question into a clear instruction for an AI model.
The instruction should be direct, specific, and focused on the task.

Context: {context}

Question: {question}

Return ONLY the instruction text, no other text or formatting.
Example output: "Explain the specific role and responsibilities of X in context Y"
"""
            
            instruction = ollama_invoke(prompt, model="llama3:8b", temperature=0.7 - (attempt * 0.2),
                                    format_spec="text", agent_name="instruction_maker")
            result = instruction.strip().strip('"')
            if result:
                return result
            
            logger.warning(f"Empty instruction generated on attempt {attempt + 1}/{max_retries}")
        except Exception as e:
            logger.warning(f"Instruction generation failed on attempt {attempt + 1}/{max_retries}: {str(e)}")
            
    # If all retries failed, return a basic instruction based on the question
    return f"Based on the provided context, {question.strip('?')}?"

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
    try:
        # First try to parse it
        data = json.loads(json_str)
        
        # Check required fields
        if not all(k in data for k in ['instruction', 'context', 'response']):
            logger.error("Missing required fields in JSON")
            raise ValueError("JSON missing required fields")
            
        # Clean any extra whitespace/quotes
        data['instruction'] = data['instruction'].strip().strip('"')
        data['context'] = data['context'].strip().strip('"')
        data['response'] = data['response'].strip().strip('"')
        
        # Preserve metadata if present
        if 'metadata' in data:
            data['metadata'] = {
                k: v.strip().strip('"') if isinstance(v, str) else v 
                for k, v in data['metadata'].items()
            }
            
        return json.dumps(data)
    except Exception as e:
        logger.error(f"JSON validation failed: {str(e)}")
        logger.error(f"Input JSON: {json_str}")
        raise

def checker_agent(draft_json: str) -> str:
    """
    Checks logical consistency of the final Dolly JSON and returns valid JSON if needed.
    """
    prompt = f"""You are a JSON validator. Check this Dolly-format entry and return ONLY the validated JSON.
If valid, return the exact same JSON. If invalid, fix any issues and return the fixed JSON.

Input JSON to validate:
{draft_json}

Requirements:
1. Must have instruction, context, and response fields
2. All text fields must be properly escaped
3. Must be valid JSON format

Return ONLY the JSON object. No other text or explanations."""

    return ollama_invoke(prompt, model="llama3:8b", temperature=0.0,
                      format_spec="text", agent_name="checker_agent")

def create_dataset_entry(question: str, context: str) -> str:
    """
    Creates and validates a dataset entry from a question and context.
    Returns the JSON string for the entry.
    """
    prompt = f"""Create an instruction-response pair for this question and context.
Return ONLY a JSON object with 'instruction', 'context', and 'response' fields.

Context: {context}
Question: {question}

The instruction should be clear and direct.
The response should be comprehensive but concise.

Return ONLY the JSON object like this:
{{"instruction": "Clear instruction based on question", 
  "context": "The given context",
  "response": "Direct answer to instruction"}}"""

    result = ollama_invoke(prompt, model="llama3:8b", temperature=0.2,
                        format_spec="text", agent_name="entry_creator")
                        
    try:
        # Validate JSON
        entry = json.loads(result)
        if not all(k in entry for k in ['instruction', 'context', 'response']):
            raise ValueError("Missing required fields")
            
        # Clean any quotes/whitespace
        entry = {
            "instruction": entry['instruction'].strip().strip('"'),
            "context": entry['context'].strip().strip('"'),
            "response": entry['response'].strip().strip('"')
        }
        
        return json.dumps(entry)
    except Exception as e:
        logger.error(f"Failed to create entry: {str(e)}")
        logger.error(f"Raw output: {result}")
        raise

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
