import platform
import os
import socket
import subprocess
import requests
import json
import logging
import time
from datetime import datetime
from typing import List, Optional
import uuid
import re

logger = logging.getLogger(__name__)

# Initialize token cost tracking
_token_cost = {
    "datetime": datetime.now().isoformat(),
    "total_eval_count": 0,
    "total_eval_duration_ns": 0,
    "tasks": {}
}

def update_token_cost(eval_count: int = 1) -> None:
    """Update the token cost tracking file with new evaluations."""
    cost_file = "token_cost.json"
    try:
        if os.path.exists(cost_file):
            with open(cost_file, 'r') as f:
                costs = json.load(f)
        else:
            costs = {"total_eval_count": 0}
            
        costs["total_eval_count"] += eval_count
        
        with open(cost_file, 'w') as f:
            json.dump(costs, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error updating token cost: {str(e)}")

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
            
            # Extract eval count from response
            eval_count = result.get('eval_count', 1)  # Default to 1 if not present
            
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
                        update_token_cost(eval_count)  # Track successful evaluation with actual count
                        return json_str
                        
                    logger.error(f"No JSON array found in response: {text}")
                    return "[]"  # Return empty array as fallback
                    
                except Exception as e:
                    logger.error(f"JSON extraction failed: {str(e)}")
                    logger.error(f"Raw response: {text}")
                    return "[]"  # Return empty array as fallback
            
            logger.info(f"Successfully completed {agent_name} request")
            update_token_cost(eval_count)  # Track successful evaluation with actual count
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

    # Get questions and track token usage
    questions_text = ollama_invoke(prompt, model="llama3:8b", temperature=0.7,
                                 format_spec="text", agent_name="questioner_agent")
    
    # Format questions to array
    questions = process_chunk_with_retry(questions_text)
    return json.dumps(list(questions))

def question_checker_agent_invoke(questions_jsonl: str) -> str:
    """Check questions for quality and redundancy."""
    questions = []
    try:
        for line in questions_jsonl.split('\n'):
            if line.strip():
                q_obj = json.loads(line)
                if 'question' in q_obj:
                    questions.append(q_obj['question'])
    except Exception as e:
        logger.error(f"Failed to parse questions JSONL: {str(e)}")
        return "[]"

    if not questions:
        return "[]"

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

    # Check questions and track token usage
    result = ollama_invoke(prompt, model="llama3:8b", temperature=0.0,
                        format_spec="json", agent_name="question_checker_agent")
    
    try:
        cleaned_questions = json.loads(result)
        if not isinstance(cleaned_questions, list):
            return json.dumps(questions)
        if not cleaned_questions:
            return json.dumps(questions)
        return json.dumps(cleaned_questions)
    except Exception as e:
        logger.error(f"Failed to parse question checker output: {str(e)}")
        return json.dumps(questions)

def make_instruction_agent(question: str, context: str, max_retries: int = 3) -> str:
    """Convert question into instruction format."""
    for attempt in range(max_retries):
        try:
            prompt = f"""Convert this question into a clear instruction for an AI model.
The instruction should be detailed and specific.

Context: {context}
Question: {question}

Return ONLY the instruction as a single string. No other text."""

            # Generate instruction and track token usage
            instruction = ollama_invoke(prompt, model="llama3:8b", 
                                   temperature=0.7 + (attempt * 0.1),
                                   format_spec="text", 
                                   agent_name="instruction_maker")
            
            if instruction.strip():
                return instruction.strip()
                
        except Exception as e:
            logger.warning(f"Instruction generation failed on attempt {attempt + 1}/{max_retries}")
            
    return question  # Fallback to using question as instruction

def maker_agent(question: str, context: str, max_retries: int = 3) -> str:
    """Generate response for instruction."""
    for attempt in range(max_retries):
        try:
            prompt = f"""Generate a clear, accurate response to this instruction using ONLY the provided context.
Be direct and comprehensive.

Context: {context}
Instruction: {question}

Return ONLY your response, no other text."""

            # Generate response and track token usage
            response = ollama_invoke(prompt, model="llama3:8b", 
                                temperature=0.2 + (attempt * 0.1),
                                format_spec="text", 
                                agent_name="response_maker")
            
            if response.strip():
                return response.strip()
                
        except Exception as e:
            logger.warning(f"Response generation failed on attempt {attempt + 1}/{max_retries}")
            
    # Fallback to extracting from context
    return extract_relevant_context(context, question)

def checker_agent(draft_json: str) -> str:
    """Verify response quality and consistency."""
    try:
        entry = json.loads(draft_json)
        prompt = f"""Verify this instruction-response pair is high quality and consistent.
Check that:
1. Response directly answers the instruction
2. Response uses ONLY information from context
3. Response is clear and well-formed
4. No formatting issues or artifacts

Instruction: {entry.get('instruction', '')}
Context: {entry.get('context', '')}
Response: {entry.get('response', '')}

If the response needs improvement, fix it and return the corrected JSON.
Otherwise return the original JSON unchanged.
Return ONLY the JSON object."""

        # Check response and track token usage
        result = ollama_invoke(prompt, model="llama3:8b", temperature=0.1,
                           format_spec="json", agent_name="response_checker")
        
        return result
    except Exception as e:
        logger.error(f"Response checking failed: {str(e)}")
        return draft_json

def formatter_agent(raw_answer: str, question: str, context: str) -> str:
    """Format response into clean dataset entry."""
    prompt = f"""Format this instruction-response pair into clean JSON.
Remove any artifacts or inconsistencies.
Keep ONLY essential information.

Format must be EXACTLY:
{{
    "instruction": "clear instruction",
    "context": "relevant context",
    "response": "clear response"
}}

Raw input:
Instruction: {question}
Context: {context}
Response: {raw_answer}

Return ONLY the formatted JSON object."""

    # Format entry and track token usage
    result = ollama_invoke(prompt, model="llama3:8b", temperature=0.1,
                        format_spec="json", agent_name="entry_formatter")
    
    try:
        entry = json.loads(result)
        required = ["instruction", "context", "response"]
        if all(k in entry for k in required):
            return json.dumps(entry)
    except Exception as e:
        logger.error(f"Formatting failed: {str(e)}")
        
    # Fallback to manual formatting
    return json.dumps({
        "instruction": question.strip(),
        "context": context.strip(),
        "response": raw_answer.strip()
    })

def retry_until_success(func, *args, timeout=60, initial_delay=1, backoff_factor=2, **kwargs):
    """Retry a function until success or timeout.
    
    Args:
        func: Function to retry
        args: Positional arguments for func
        timeout: Maximum seconds to try before giving up (default: 60)
        initial_delay: Initial delay between retries in seconds (default: 1)
        backoff_factor: Multiply delay by this factor after each retry (default: 2)
        kwargs: Keyword arguments for func
        
    Returns:
        Result from func
        
    Raises:
        TimeoutError: If no valid result is obtained within timeout period
    """
    start_time = time.time()
    delay = initial_delay
    attempt = 1
    
    def is_valid_result(result):
        """Check if result is valid (not None and has content)."""
        if result is None:
            return False
        if isinstance(result, str) and not result.strip():
            return False
        if isinstance(result, (list, dict)) and not result:
            return False
        return True
    
    while (time.time() - start_time) < timeout:
        try:
            result = func(*args, **kwargs)
            if is_valid_result(result):
                logger.info(f"Succeeded on attempt {attempt} after {time.time() - start_time:.1f}s")
                update_token_cost()  # Track successful evaluation
                return result
                
            logger.warning(
                f"Attempt {attempt} returned invalid result after {time.time() - start_time:.1f}s. "
                f"Retrying in {delay}s..."
            )
            
        except Exception as e:
            logger.warning(
                f"Attempt {attempt} failed with error after {time.time() - start_time:.1f}s: {str(e)}. "
                f"Retrying in {delay}s..."
            )
            
        time.sleep(delay)
        delay = min(delay * backoff_factor, timeout/4)  # Cap delay at 1/4 of timeout
        attempt += 1
        
    total_time = time.time() - start_time
    error_msg = f"Failed to get valid result after {attempt} attempts and {total_time:.1f}s"
    logger.error(error_msg)
    raise TimeoutError(error_msg)

def create_dataset_entry(question: str, context: str, node_key: str = None) -> str:
    """Create complete dataset entry with proper flow and validation at each step."""
    entry_id = str(uuid.uuid4())
    logger.info(f"Creating dataset entry {entry_id} for node {node_key}")
    
    try:
        # 1. Convert question to instruction with retries
        logger.info(f"[{entry_id}] Converting question to instruction")
        instruction = retry_until_success(
            make_instruction_agent,
            question, 
            context,
            timeout=60  # 1 minute timeout
        )
        
        # Clean instruction text
        instruction = clean_llm_text(instruction)
        logger.info(f"[{entry_id}] Generated instruction: {instruction}")
            
        # 2. Generate response with retries
        logger.info(f"[{entry_id}] Generating response")
        response = retry_until_success(
            maker_agent,
            instruction, 
            context,
            timeout=60
        )
        
        # Clean response text
        response = clean_llm_text(response)
        logger.info(f"[{entry_id}] Generated response: {response}")
        
        # 3. Create draft JSON with proper escaping
        draft_entry = {
            "instruction": instruction,
            "context": context.strip(),
            "response": response
        }
        
        # Properly escape JSON
        draft_json = json.dumps(draft_entry, ensure_ascii=False)
        
        # 4. Check response quality with retries
        logger.info(f"[{entry_id}] Checking response quality")
        checked_entry = json.loads(draft_json)
        if not all(k in checked_entry for k in ["instruction", "context", "response"]):
            raise ValueError("Checked JSON missing required fields")
        
        # 5. Format final entry with retries
        logger.info(f"[{entry_id}] Formatting final entry")
        formatted_json = retry_until_success(
            formatter_agent,
            checked_entry["response"],
            checked_entry["instruction"],
            checked_entry["context"],
            timeout=60
        )
        
        # 6. Final validation and cleanup
        final_entry = json.loads(formatted_json)
        
        # Validate required fields
        if not all(k in final_entry for k in ["instruction", "context", "response"]):
            raise ValueError("Final JSON missing required fields")
            
        # Validate field content
        if any(len(final_entry[k].strip()) < 10 for k in ["instruction", "response"]):
            raise ValueError("Final JSON has invalid field lengths")
            
        # Clean fields
        for k in ["instruction", "context", "response"]:
            if isinstance(final_entry[k], (dict, list)):
                raise ValueError("Final JSON has nested objects")
            # Apply final cleaning to instruction and response
            if k in ["instruction", "response"]:
                final_entry[k] = clean_llm_text(final_entry[k])
            else:
                final_entry[k] = final_entry[k].strip()
        
        # Final JSON string with proper escaping
        final_json = json.dumps(final_entry, ensure_ascii=False)
        logger.info(f"[{entry_id}] Successfully created dataset entry")
        
        # Log metadata to datasetlog.jsonl
        metadata = {
            "entry_id": entry_id,
            "node_key": node_key,
            "created_at": datetime.now().isoformat(),
            "source_question": question
        }
        
        with open("data/datasetlog.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")
            f.flush()
            
        return final_json
            
    except Exception as e:
        logger.error(f"[{entry_id}] Fatal error in dataset entry creation: {str(e)}")
        raise  # Re-raise to handle at higher level

def clean_llm_text(text: str) -> str:
    """Clean text from LLM responses of special characters and artifacts.
    
    Removes:
    - Multiple types of quotes
    - Special brackets
    - Multiple spaces
    - Leading/trailing whitespace
    """
    # Remove various types of quotes and brackets
    text = re.sub(r'[\[\]"\'`]', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

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

def extract_relevant_context(context, question):
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
