#!/usr/bin/env python3
"""Code Security Audit - Advanced vulnerability detection (PAID)

Provider: SecureAI Labs
Price: $0.50 per run
Checks for: SQL injection, XSS, CSRF, and OWASP Top 10 vulnerabilities
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


def check_security_vulnerabilities(code):
    """Check code for common security vulnerabilities."""
    vulnerabilities = []

    # SQL injection patterns
    if any(pattern in code.lower() for pattern in ['execute(', 'query(', 'sql']):
        if not any(safe in code.lower() for safe in ['prepared', 'parameterized', '?']):
            vulnerabilities.append("Potential SQL injection - use parameterized queries")

    # XSS patterns
    if 'innerHTML' in code or 'dangerouslySetInnerHTML' in code:
        vulnerabilities.append("Potential XSS - sanitize user input before rendering")

    # Hardcoded credentials
    if any(pattern in code.lower() for pattern in ['password=', 'api_key=', 'secret=']):
        vulnerabilities.append("Hardcoded credentials detected - use environment variables")

    return vulnerabilities


def main():
    parser = argparse.ArgumentParser(description="Code security audit eval")
    parser.add_argument("--testcase-id", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--expected-dir", default=None)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Check for API key (this is a paid eval)
    secureai_key = os.environ.get("SECUREAI_API_KEY")
    if not secureai_key:
        result = {
            "$schema": "gitbench-result-v1",
            "status": "error",
            "score": 0.0,
            "max_score": 1.0,
            "error_message": "SECUREAI_API_KEY not set. This is a paid eval. Get your key at https://secureai-labs.com",
            "trace": [],
        }
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    # Load prompt
    prompt_file = input_dir / "prompt.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"No prompt.txt in {input_dir}")

    prompt = prompt_file.read_text().strip()

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

        # Check for security vulnerabilities
        vulnerabilities = check_security_vulnerabilities(content)

        # Score based on vulnerabilities found
        if len(vulnerabilities) == 0:
            score = 1.0
            status = "pass"
        elif len(vulnerabilities) <= 2:
            score = 0.6
            status = "fail"
        else:
            score = 0.3
            status = "fail"

        result = {
            "$schema": "gitbench-result-v1",
            "status": status,
            "score": score,
            "max_score": 1.0,
            "duration_ms": round(duration_ms, 1),
            "token_usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            "result_data": {
                "vulnerabilities_found": len(vulnerabilities),
                "vulnerabilities": vulnerabilities,
                "provider": "SecureAI Labs",
                "eval_type": "security"
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
