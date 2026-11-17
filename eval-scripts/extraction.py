#!/usr/bin/env python3
"""Data Extraction Evaluation - Test agent's ability to extract structured data.

This eval tests:
- Extracting data from text
- Parsing structured formats (JSON, CSV, tables)
- Returning data in requested format
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


def evaluate_extraction(response, expected=None):
    """Score extraction quality."""
    content = response.lower()

    # Check for structured data indicators
    has_json = "{" in content and "}" in content
    has_list = "[" in content and "]" in content
    has_table = "|" in content or "," in content

    # Check if data was extracted (not just explanation)
    has_data = any([has_json, has_list, has_table])

    if not has_data:
        return 0.3

    # Check if expected data is present
    if expected:
        if expected.lower() in content:
            return 1.0
        else:
            return 0.6

    return 0.8


def main():
    parser = argparse.ArgumentParser(description="Data extraction eval")
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

    # Load expected output if available
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

        # Evaluate extraction quality
        score = evaluate_extraction(content, expected)

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
