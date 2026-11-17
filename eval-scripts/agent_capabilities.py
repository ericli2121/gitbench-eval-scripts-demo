#!/usr/bin/env python3
"""Agent Capabilities Evaluation - Test LLM agent across different task types.

This eval tests general agent capabilities including:
- Following instructions
- Reasoning through problems
- Generating appropriate responses
"""
import argparse
import json
import os
import time
from pathlib import Path


def create_llm_client():
    """Create LLM client based on environment."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise RuntimeError("anthropic package not installed")


def evaluate_response(prompt_type, response, expected=None):
    """Score the agent's response."""
    content = response.lower()

    # Check for substantial, relevant response
    if len(content) < 50:
        return 0.3

    # Check for reasoning indicators
    has_reasoning = any(marker in content for marker in [
        "because", "therefore", "first", "second", "step", "reason", "since"
    ])

    # Check for structure
    has_structure = any(marker in content for marker in [
        "\n\n", "1.", "2.", "- ", "* ", "```"
    ])

    score = 0.5
    if has_reasoning:
        score += 0.25
    if has_structure:
        score += 0.25

    return min(score, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Agent capabilities eval")
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

    start_time = time.time()
    trace = []

    try:
        client = create_llm_client()

        # User message
        trace.append({
            "role": "user",
            "content": prompt,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        # Call LLM
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        duration_ms = (time.time() - start_time) * 1000
        content = "".join(block.text for block in response.content if block.type == "text")

        # Agent response
        trace.append({
            "role": "assistant",
            "content": content,
            "token_usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_ms": duration_ms,
        })

        # Evaluate
        score = evaluate_response(prompt_type, content)

        result = {
            "$schema": "gitbench-result-v1",
            "status": "pass" if score >= 0.7 else "fail",
            "score": score,
            "max_score": 1.0,
            "duration_ms": round(duration_ms, 1),
            "token_usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            "result_data": {
                "response_length": len(content),
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
