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

def ollama_invoke(prompt, model="llama3:8b", temperature=0.0, format_spec=None, agent_name="ollama_invoke", max_retries=3):
    """Call the Ollama API with retry logic"""
    logger.info(f"Calling Ollama API with model {model}")
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Get IP address for Ollama
            running, ip = ensure_ollama_running()
            if not running:
                raise Exception("Could not connect to Ollama server")
                
            # Make API request
            url = f"http://{ip}:11434/api/generate"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False,  # Disable streaming for complete response
                "temperature": temperature
            }
            
            # Add format spec if provided
            if format_spec:
                data["format"] = format_spec
            
            logger.info(f"Successfully connected to Ollama on {ip}")
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 404:
                error_msg = response.json().get("error", "Unknown error")
                logger.error(f"Ollama API error (status 404):")
                logger.error(f"Request URL: {url}")
                logger.error(f"Request headers: {json.dumps(headers, indent=2)}")
                logger.error(f"Request data: {json.dumps(data, indent=2)}")
                logger.error(f"Response headers: {dict(response.headers)}")
                logger.error(f"Response text: {response.text}")
                
                if retry_count < max_retries - 1:
                    retry_count += 1
                    logger.warning(f"Request failed (attempt {retry_count}/{max_retries}), retrying...")
                    continue
                else:
                    logger.error("Ollama API returned 404 - Model not found or API endpoint incorrect")
                    logger.error("Please ensure:")
                    logger.error("1. Ollama server is running (ollama serve)")
                    logger.error(f"2. Model '{model}' is available (ollama list)")
                    logger.error(f"3. API endpoint is correct ({url})")
                    raise Exception(f"404 Client Error: Not Found for url: {url}")
            
            response.raise_for_status()
            response_json = response.json()
            
            # Log token metrics
            eval_count = len(prompt.split())  # Simple token count estimation
            eval_duration_ns = int(float(response_json.get("eval_duration", 2.5)) * 1e9)
            log_token_cost(agent_name, eval_count, eval_duration_ns)
            
            return response_json["response"].strip()
            
        except Exception as e:
            if retry_count < max_retries - 1:
                retry_count += 1
                logger.warning(f"Request failed (attempt {retry_count}/{max_retries}), retrying...")
                continue
            else:
                raise e

    raise Exception("Max retries exceeded")

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
        partial_cleaned = ollama_invoke(prompt, model="llama3:8b", temperature=0.0, agent_name="cleaner_agent")
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
    return ollama_invoke(prompt, model="llama3:8b", temperature=0.2, 
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

def checker_agent(draft_json):
    """
    Checks logical consistency of the final Dolly JSON and returns valid JSON if needed.
    """
    prompt = f"""
You are a checker agent. Check this JSON for logical consistency:
{draft_json}
Return corrected JSON if needed. No other text.
"""
    return ollama_invoke(prompt, model="llama3:8b", temperature=0.0, 
                        format_spec="json", agent_name="checker_agent")

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
