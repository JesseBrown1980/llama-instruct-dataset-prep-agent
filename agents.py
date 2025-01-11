import requests
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def log_token_cost(agent_name, eval_count, eval_duration_ns):
    """Log token cost metrics to token_cost.json"""
    cost_data = {
        "task": agent_name,
        "datetime": "2025-01-11T11:47:55-08:00",
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration_ns
    }
    with open('token_cost.json', 'w') as f:
        json.dump(cost_data, f, indent=4)

def ollama_invoke(prompt, model="llama3.1", temperature=0.0, format_spec=None, agent_name="ollama_invoke"):
    """
    Invoke local Ollama endpoint with prompt. Adjust URL and model name as needed.
    """
    logger.debug(f"Invoking Ollama with model={model}, temperature={temperature}")
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "model": model,
        "stream": False
    }
    
    if format_spec:
        data["format"] = format_spec
        
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    response_data = resp.json()
    
    # Log token metrics
    eval_count = len(prompt.split())  # Simple token count estimation
    eval_duration_ns = int(float(response_data.get("eval_duration", 2.5)) * 1e9)
    log_token_cost(agent_name, eval_count, eval_duration_ns)
    
    return response_data["response"].strip()

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
        partial_cleaned = ollama_invoke(prompt, model="llama3.1", temperature=0.0, agent_name="cleaner_agent")
        cleaned_result.append(partial_cleaned)
        start = end
    return "\n".join(cleaned_result).strip()

def questioner_agent_invoke(knowledge_context):
    """
    Generates as many interesting legal questions as possible (no redundancy).
    Must return a JSON array of strings (the questions).
    """
    prompt = f"""
You are a question generator agent. Given this knowledge context, generate as many interesting legal questions as you can. 
Return a JSON array of question strings. No pleasantries, no repetition.

Context:
{knowledge_context}
Output only JSON.
"""
    return ollama_invoke(prompt, model="llama3.1", temperature=0.2, 
                        format_spec="json", agent_name="questioner_agent")

def question_checker_agent_invoke(question_list_json):
    """
    Checks a JSON array of questions for redundancy or errors.
    Returns corrected JSON array if needed. No pleasantries.
    """
    prompt = f"""
You are a logical checker. The input is a JSON array of questions. Check for redundancy or errors. Return a corrected JSON array. 
No pleasantries. Only valid JSON.

Questions:
{question_list_json}
"""
    return ollama_invoke(prompt, model="llama3.1", temperature=0.0, 
                        format_spec="json", agent_name="question_checker_agent")

def maker_agent(question, knowledge_context):
    """
    Produces a concise answer based on knowledge_context for a single question.
    """
    prompt = f"""
You create a concise answer. 
Context:
{knowledge_context}
Question:
{question}
No pleasantries.
"""
    return ollama_invoke(prompt, model="llama3.1", temperature=0.2, agent_name="maker_agent")

def formatter_agent(raw_answer, question, context):
    """
    Converts raw_answer into Dolly/LLAMA style JSON with format specification
    """
    prompt = f"""Format this into a proper response:
Question: {question}
Context: {context}
Answer: {raw_answer}
Format as JSON."""

    format_spec = {
        "type": "object",
        "properties": {
            "instruction": {"type": "string"},
            "context": {"type": "string"},
            "response": {"type": "string"}
        },
        "required": ["instruction", "context", "response"]
    }
    
    return ollama_invoke(prompt, model="llama3.1", temperature=0.0, 
                        format_spec=format_spec, agent_name="formatter_agent")

def checker_agent(draft_json):
    """
    Checks logical consistency of the final Dolly JSON and returns valid JSON if needed.
    """
    prompt = f"""
Verify this JSON is valid and logically consistent:
{draft_json}
Return corrected JSON if needed. No other text.
"""
    return ollama_invoke(prompt, model="llama3.1", temperature=0.0, 
                        format_spec="json", agent_name="checker_agent")
