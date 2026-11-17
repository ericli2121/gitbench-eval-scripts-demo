#!/usr/bin/env python3
"""Real LLM agent eval script using local or cloud LLM.

Supports:
- Anthropic Claude (set ANTHROPIC_API_KEY)
- Local Ollama (install ollama, pull llama3.2)
"""
import argparse
import json
import os
import time
from pathlib import Path


def create_llm_client():
    """Create LLM client based on LLM_MODE from .env file."""
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed

    llm_mode = os.environ.get("LLM_MODE", "").lower()

    if llm_mode == "local":
        # Use Ollama
        try:
            import requests
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                return "ollama", requests
            else:
                raise RuntimeError("Ollama not responding at localhost:11434")
        except Exception as e:
            raise RuntimeError(
                f"LLM_MODE=local but Ollama not available.\n"
                f"1. Install Ollama: https://ollama.ai\n"
                f"2. Start Ollama: ollama serve\n"
                f"3. Pull a model: ollama pull llama3.2\n"
                f"Error: {e}"
            )
    elif llm_mode == "claude":
        # Use Anthropic Claude
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_MODE=claude but ANTHROPIC_API_KEY not set.\n"
                "Set ANTHROPIC_API_KEY in .env file or environment."
            )
        try:
            import anthropic
            return "anthropic", anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise RuntimeError(
                "anthropic package not installed.\n"
                "Install with: pip install anthropic"
            )
    else:
        raise RuntimeError(
            "LLM_MODE not set or invalid.\n"
            "Add to .env file:\n"
            "  LLM_MODE=local   (for Ollama)\n"
            "  LLM_MODE=claude  (for Anthropic Claude)\n"
        )


def call_anthropic(client, messages, model="claude-3-5-haiku-20241022"):
    """Call Anthropic Claude."""
    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=messages
    )
    duration_ms = (time.time() - start) * 1000

    content = "".join(block.text for block in response.content if block.type == "text")

    return {
        "content": content,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "duration_ms": duration_ms,
    }


def call_ollama(client, messages, model=None):
    """Call local Ollama."""
    if model is None:
        model = os.environ.get("OLLAMA_MODEL", "llama3.2")

    start = time.time()

    # Convert messages to Ollama format
    prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)

    response = client.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.text}")

    result = response.json()
    duration_ms = (time.time() - start) * 1000

    return {
        "content": result["response"],
        "usage": {
            "input_tokens": result.get("prompt_eval_count", 0),
            "output_tokens": result.get("eval_count", 0),
        },
        "duration_ms": duration_ms,
    }


def evaluate_response(prompt_type, response, expected=None):
    """Score the agent's response."""
    content = response.lower()

    if prompt_type == "code_generation":
        # Check if code is present
        if "def " in content or "function " in content or "```" in content:
            return 1.0
        return 0.0

    elif prompt_type == "reasoning":
        # Check for step-by-step reasoning
        if any(marker in content for marker in ["step 1", "first", "1.", "because", "therefore"]):
            return 1.0
        return 0.5

    elif prompt_type == "extraction":
        # Check if expected data is extracted
        if expected:
            if expected.lower() in content:
                return 1.0
            return 0.0
        return 0.5

    elif prompt_type == "creative":
        # Just check if response is substantial
        if len(content) > 100:
            return 1.0
        return 0.3

    # Default: non-empty response
    return 1.0 if len(content) > 10 else 0.0


def main():
    parser = argparse.ArgumentParser(description="LLM agent eval")
    parser.add_argument("--testcase-id", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--expected-dir", default=None)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Load testcase config
    config_file = input_dir / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text())
    else:
        config = {}

    # Load prompt
    prompt_file = input_dir / "prompt.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"No prompt.txt in {input_dir}")

    prompt = prompt_file.read_text().strip()
    prompt_type = config.get("type", "general")

    # Load expected output if available
    expected = None
    if args.expected_dir:
        expected_file = Path(args.expected_dir) / "expected.txt"
        if expected_file.exists():
            expected = expected_file.read_text().strip()

    start_time = time.time()
    trace = []

    try:
        # Create LLM client
        provider, client = create_llm_client()

        # System message
        trace.append({
            "role": "system",
            "content": "You are a helpful AI assistant.",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        # User message
        trace.append({
            "role": "user",
            "content": prompt,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        # Call LLM
        messages = [{"role": "user", "content": prompt}]

        if provider == "anthropic":
            llm_result = call_anthropic(client, messages)
        else:
            llm_result = call_ollama(client, messages)

        # Agent response
        trace.append({
            "role": "assistant",
            "content": llm_result["content"],
            "token_usage": llm_result["usage"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_ms": llm_result["duration_ms"],
            "metadata": {"provider": provider},
        })

        # Evaluate
        score = evaluate_response(prompt_type, llm_result["content"], expected)

        total_duration = (time.time() - start_time) * 1000

        result = {
            "$schema": "gitbench-result-v1",
            "status": "pass" if score >= 0.7 else "fail",
            "score": score,
            "max_score": 1.0,
            "duration_ms": round(total_duration, 1),
            "token_usage": {
                "input_tokens": llm_result["usage"]["input_tokens"],
                "output_tokens": llm_result["usage"]["output_tokens"],
                "total_tokens": llm_result["usage"]["input_tokens"] + llm_result["usage"]["output_tokens"],
            },
            "result_data": {
                "provider": provider,
                "response": llm_result["content"][:500],  # Truncate for storage
            },
            "trace": trace,
        }

    except Exception as exc:
        result = {
            "$schema": "gitbench-result-v1",
            "status": "error",
            "score": 0.0,
            "max_score": 1.0,
            "error_message": str(exc),
            "trace": trace,
        }

    # Write result
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
