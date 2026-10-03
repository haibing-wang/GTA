#!/usr/bin/env python3
"""GTA Project Environment & Dataset Diagnostics Script.

Runs sanity checks on:
1. Python version and essential dependencies
2. Dataset directory layout (GTA-Atomic and GTA-Workflow)
3. Tool server accessibility
4. LLM API endpoint & environment variables
5. Evaluator configuration
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

for p in [str(REPO_ROOT / "opencompass"), str(REPO_ROOT / "agentlego")]:
    if p not in sys.path:
        sys.path.insert(0, p)

def color(text: str, code: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

def green(text: str) -> str:
    return color(text, "32")

def yellow(text: str) -> str:
    return color(text, "33")

def red(text: str) -> str:
    return color(text, "31")

def cyan(text: str) -> str:
    return color(text, "36")

def check_python():
    print(cyan("\n=== 1. Python Environment ==="))
    v = sys.version_info
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {v.major}.{v.minor}.{v.micro}")
    if v.major == 3 and v.minor in (10, 11):
        print(f"[{green('OK')}] Python version {v.major}.{v.minor} is optimal for OpenCompass / AgentLego.")
    elif v.major == 3 and v.minor >= 12:
        print(f"[{yellow('WARN')}] Python {v.major}.{v.minor} is detected. Note: OpenCompass and mmcv usually recommend Python 3.10 or 3.11.")
    else:
        print(f"[{red('FAIL')}] Python {v.major}.{v.minor} is not recommended.")

    packages = {
        "requests": "General HTTP requests",
        "openai": "Responses API and LLM scoring",
        "tiktoken": "Token counting for OpenAI models",
        "mmengine": "OpenCompass engine framework",
        "lagent": "Agent framework for tool use",
        "agentlego": "Tool server and tool execution",
    }
    for pkg, desc in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"[{green('OK')}] {pkg} ({version}) - {desc}")
        except ImportError:
            print(f"[{yellow('MISSING')}] {pkg} - {desc}")

def check_datasets():
    print(cyan("\n=== 2. Dataset Layout Check ==="))
    data_dir = REPO_ROOT / "opencompass" / "data"
    print(f"Expected Data Directory: {data_dir}")

    # GTA-2 / GTA-Workflow
    gta_v2 = data_dir / "gta_dataset_v2"
    toolmeta_v2 = gta_v2 / "toolmeta.json"
    end_v2 = gta_v2 / "end.json"
    if gta_v2.exists():
        print(f"[{green('FOUND')}] GTA-Workflow directory: {gta_v2}")
        print(f"  - toolmeta.json: {'[OK]' if toolmeta_v2.exists() else red('[MISSING]')}")
        print(f"  - end.json:      {'[OK]' if end_v2.exists() else red('[MISSING]')}")
    else:
        print(f"[{yellow('NOT FOUND')}] GTA-Workflow directory ({gta_v2})")
        print("  -> Download GTA-Workflow from: https://github.com/open-compass/GTA/releases/download/v0.2.0/gta_workflow_dataset.zip")
        print("  -> Unzip into opencompass/data/gta_dataset_v2")

    # GTA-1 / GTA-Atomic
    gta_v1 = data_dir / "gta_dataset"
    toolmeta_v1 = gta_v1 / "toolmeta.json"
    if gta_v1.exists():
        print(f"[{green('FOUND')}] GTA-Atomic directory: {gta_v1}")
        print(f"  - toolmeta.json: {'[OK]' if toolmeta_v1.exists() else red('[MISSING]')}")
    else:
        print(f"[{yellow('NOT FOUND')}] GTA-Atomic directory ({gta_v1})")
        print("  -> Download GTA-Atomic from: https://github.com/open-compass/GTA/releases/download/v0.1.0/gta_dataset.zip")
        print("  -> Unzip into opencompass/data/gta_dataset")

def check_tool_server():
    print(cyan("\n=== 3. Tool Server Check ==="))
    tool_url = os.getenv("GTA_TOOL_SERVER", "http://127.0.0.1:16181")
    print(f"Testing Tool Server URL: {tool_url}")
    try:
        req = urllib.request.Request(tool_url + "/tools", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = resp.read().decode("utf-8")
            print(f"[{green('ONLINE')}] Tool Server is running and responding!")
    except Exception as e:
        print(f"[{yellow('OFFLINE')}] Tool Server is not reachable ({e}).")
        print("  -> If you plan to run end-to-end tool calls, start AgentLego tool server:")
        print("     agentlego-server start --port 16181 --extra ./benchmark.py `cat benchmark_toollist_v2.txt` --host 0.0.0.0")

def check_env_vars():
    print(cyan("\n=== 4. Environment Variables & LLM Credentials ==="))
    vars_to_check = [
        ("OPENAI_API_KEY", "Inference LLM API Key", False),
        ("OPENAI_API_BASE", "Inference LLM Base URL (e.g. http://localhost:12580/v1/chat/completions)", False),
        ("EVAL_OPENAI_API_KEY", "Evaluator (GPT-5.2) API Key", False),
        ("EVAL_OPENAI_BASE_URL", "Evaluator OpenAI Base URL (must end with /v1)", False),
        ("EVAL_PROXY", "HTTP proxy for GPT-5.2 evaluation", False),
        ("SERPER_API_KEY", "GoogleSearch tool key (serper.dev)", False),
        ("MATHPIX_APP_ID", "MathOCR tool ID (mathpix.com)", False),
    ]
    for var, desc, required in vars_to_check:
        val = os.getenv(var)
        if val:
            masked = val[:6] + "..." + val[-4:] if len(val) > 10 else "***"
            print(f"[{green('SET')}] {var}={masked} ({desc})")
        else:
            status = red("NOT SET") if required else yellow("NOT SET")
            print(f"[{status}] {var} ({desc})")

def main():
    print(cyan("=================================================="))
    print(cyan("   GTA Benchmark & Evaluation Diagnostics Tool   "))
    print(cyan("=================================================="))
    check_python()
    check_datasets()
    check_tool_server()
    check_env_vars()
    print(cyan("\n=== Diagnostics Complete ==="))

if __name__ == "__main__":
    main()
