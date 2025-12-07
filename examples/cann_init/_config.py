# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""
Shared configuration for CANN Init examples.

All example scripts import test data and utilities from here.
Loads environment variables from .env file automatically.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from current directory
load_dotenv(Path(__file__).parent / ".env")

# Directories
SRC_DIR = Path(__file__).parent / "0_test_task_src"
OUTPUT_DIR = Path(__file__).parent / "output"


def ensure_output_dir(subdir: str = "") -> Path:
    """Ensure output directory exists and return path."""
    path = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_llm():
    """Get LLM instance from environment variables."""
    from evotoolkit.tools.llm import HttpsApi

    api_url = os.getenv("API_URL")
    api_key = os.getenv("API_KEY")
    model = os.getenv("MODEL", "gpt-4o")

    if not api_url or not api_key:
        raise ValueError(
            "API_URL and API_KEY must be set in .env file.\n"
            "Example .env:\n"
            "  API_URL=ai.api.xn--fiqs8s\n"
            "  API_KEY=sk-xxx\n"
            "  MODEL=claude-sonnet-4-5-20250929"
        )

    return HttpsApi(api_url=api_url, key=api_key, model=model)


def get_task(op_name: str = "Add", npu_type: str = "Ascend910B"):
    """Get CANNInitTask instance."""
    from evotoolkit.task.cann_init import CANNInitTask

    return CANNInitTask(op_name=op_name, npu_type=npu_type)
