import platform
import os
import socket
import subprocess
import requests
import json
import logging
import time
from datetime import datetime

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
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def start_ollama(ip):
    """Start Ollama in Ubuntu-22.04 with the specified IP and ensure model is available"""
    logger.info(f"Starting Ollama in Ubuntu-22.04 on {ip}...")
    try:
        # Run ollama serve in Ubuntu-22.04
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
                    break
            except:
                time.sleep(1)
                continue
        else:
            logger.error("Timed out waiting for Ollama to start")
            return False

        # Pull the model
        logger.info("Pulling model llama3:8b...")
        subprocess.run(
            ['wsl.exe', '-d', 'Ubuntu-22.04', '--', 'ollama', 'pull', 'llama3:8b'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to start Ollama: {str(e)}")
        return False

def ensure_ollama_running():
    """
    Check if Ollama API is responding, if not try to start it.
    Uses current machine's IP address and starts Ollama in Ubuntu if needed.
    """
    ip = get_machine_ip()
    url = f"http://{ip}:11434/api/version"
    
    try:
        # Test if Ollama is already running
        resp = requests.get(url)
        if resp.status_code == 200:
            logger.info(f"Successfully connected to Ollama on {ip}")
            return True, ip
    except requests.exceptions.ConnectionError:
        # Try to start Ollama
        if start_ollama(ip):
            return True, ip
        else:
            logger.error(f"Could not start Ollama. Please ensure:")
            logger.error("1. Ubuntu-22.04 is running in WSL")
            logger.error("2. Ollama is installed in Ubuntu-22.04")
            logger.error(f"3. Port 11434 is available on {ip}")
            return False, ip
    
    return False, ip

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

def ollama_invoke(prompt, model="llama3:8b", temperature=0.0, format_spec=None, agent_name="ollama_invoke", max_retries=3):
    """
    Invoke Ollama endpoint with prompt.
    Will automatically start Ollama in Ubuntu if it's not running.
    """
    # Ensure Ollama is running and get host
    is_running, ip = ensure_ollama_running()
    if not is_running:
        raise RuntimeError(f"Failed to connect to Ollama server. Please check the logs for details.")
        
    logger.debug(f"Invoking Ollama with model={model}, temperature={temperature}")
    url = f"http://{ip}:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "model": model,
        "stream": False,
        "temperature": temperature
    }
    
    if format_spec:
        data["format"] = format_spec
    
    last_error = None
    for attempt in range(max_retries):
        try:
            # Log request details at debug level
            logger.debug(f"Sending request to Ollama API (attempt {attempt + 1}/{max_retries}):")
            logger.debug(f"URL: {url}")
            logger.debug(f"Headers: {json.dumps(headers, indent=2)}")
            logger.debug(f"Data: {json.dumps(data, indent=2)}")
            
            resp = requests.post(url, headers=headers, json=data)
            
            # If there's an error, log everything at error level
            if resp.status_code != 200:
                logger.error(f"Ollama API error (status {resp.status_code}):")
                logger.error(f"Request URL: {url}")
                logger.error(f"Request headers: {json.dumps(headers, indent=2)}")
                logger.error(f"Request data: {json.dumps(data, indent=2)}")
                logger.error(f"Response headers: {dict(resp.headers)}")
                logger.error(f"Response text: {resp.text}")
                
                if resp.status_code == 404 and "model not found" in resp.text.lower():
                    # Try to refresh the model
                    logger.info(f"Model {model} not found, attempting to refresh...")
                    try:
                        subprocess.run(["ollama", "pull", model], check=True)
                        time.sleep(1)  # Give it a moment to load
                        continue  # Try the request again
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to refresh model: {str(e)}")
                
            resp.raise_for_status()
            response_data = resp.json()
            
            # Log token metrics
            eval_count = len(prompt.split())  # Simple token count estimation
            eval_duration_ns = int(float(response_data.get("eval_duration", 2.5)) * 1e9)
            log_token_cost(agent_name, eval_count, eval_duration_ns)
            
            return response_data["response"].strip()
            
        except requests.exceptions.HTTPError as e:
            last_error = e
            if attempt == max_retries - 1:  # Last attempt
                if e.response.status_code == 404:
                    logger.error(f"Ollama API returned 404 - Model {model} not found or API endpoint incorrect")
                    logger.error("Please ensure:")
                    logger.error("1. Ollama server is running (ollama serve)")
                    logger.error(f"2. Model '{model}' is available (ollama list)")
                    logger.error(f"3. API endpoint is correct ({url})")
                raise
            else:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(1)  # Wait before retry
                
        except Exception as e:
            last_error = e
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f"Unexpected error calling Ollama API: {str(e)}")
                logger.error("Request details:")
                logger.error(f"URL: {url}")
                logger.error(f"Headers: {json.dumps(headers, indent=2)}")
                logger.error(f"Data: {json.dumps(data, indent=2)}")
                raise
            else:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(1)  # Wait before retry
    
    raise last_error  # Should never get here, but just in case

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
        partial_cleaned = ollama_invoke(prompt, model="llama2", temperature=0.0, agent_name="cleaner_agent")
        cleaned_result.append(partial_cleaned)
        start = end
    return "\n".join(cleaned_result).strip()

def questioner_agent_invoke(knowledge_context):
    """
    Generates as many interesting legal questions as possible (no redundancy).
    Must return a JSON array of strings (the questions).
    """
    prompt = f"""
You are a questioner agent. Generate interesting legal questions from this knowledge:
{knowledge_context}
Output only JSON.
"""
    return ollama_invoke(prompt, model="llama2", temperature=0.2, 
                        format_spec="json", agent_name="questioner_agent")

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
    return ollama_invoke(prompt, model="llama2", temperature=0.0, 
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
    return ollama_invoke(prompt, model="llama2", temperature=0.2, agent_name="maker_agent")

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
    
    return ollama_invoke(prompt, model="llama2", temperature=0.0, 
                        format_spec=format_spec, agent_name="formatter_agent")

def checker_agent(draft_json):
    """
    Checks logical consistency of the final Dolly JSON and returns valid JSON if needed.
    """
    prompt = f"""
You are a checker agent. Check this JSON for logical consistency:
{draft_json}
Return corrected JSON if needed. No other text.
"""
    return ollama_invoke(prompt, model="llama2", temperature=0.0, 
                        format_spec="json", agent_name="checker_agent")
