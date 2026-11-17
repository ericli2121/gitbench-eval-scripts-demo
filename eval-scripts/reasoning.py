#!/usr/bin/env python3
"""Reasoning Evaluation - Test agent's logical reasoning abilities.

This eval tests:
- Step-by-step reasoning
- Logical deduction
- Mathematical reasoning
- Causal understanding
"""
import argparse
import json
import os
import time
from pathlib import Path


def create_llm_client():
    """Create LLM client."""
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


def evaluate_reasoning(response, expected=None):
    """Score reasoning quality."""
    content = response.lower()

    # Check for reasoning indicators
    reasoning_markers = [
        "step 1", "step 2", "first", "second", "third",
        "because", "therefore", "thus", "hence", "since",
        "if", "then", "so", "given", "assume",
        "1.", "2.", "3."
    ]

    marker_count = sum(1 for marker in reasoning_markers if marker in content)

    # Check for conclusion
    has_conclusion = any(word in content for word in [
        "conclusion", "answer", "result", "solution", "therefore"
    ])

    # Base score on reasoning depth
    if marker_count >= 3 and has_conclusion:
        score = 1.0
    elif marker_count >= 2:
        score = 0.8
    elif marker_count >= 1:
        score = 0.6
    else:
        score = 0.4

    # Check against expected answer if provided
    if expected and expected.lower() in content:
        score = min(score + 0.2, 1.0)

    return score


def main():
    parser = argparse.ArgumentParser(description="Reasoning eval")
    parser.add_argument("--testcase-id", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--expected-dir", default=None)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Load prompt
    prompt_file = input_dir / "prompt.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"No prompt.txt in {input_dir}")

    prompt = prompt_file.read_text().strip()

    # Load expected answer if available
    expected = None
    if args.expected_dir:
        expected_file = Path(args.expected_dir) / "expected.txt"
        if expected_file.exists():
            expected = expected_file.read_text().strip()

    start_time = time.time()
    trace = []

    try:
        client = create_llm_client()

        trace.append({
            "role": "user",
            "content": prompt,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        duration_ms = (time.time() - start_time) * 1000
        content = "".join(block.text for block in response.content if block.type == "text")

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

        # Evaluate reasoning
        score = evaluate_reasoning(content, expected)

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

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
