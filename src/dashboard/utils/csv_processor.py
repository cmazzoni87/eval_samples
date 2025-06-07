"""Utilities for processing CSV files in the Streamlit dashboard."""

import pandas as pd
import json
import os
from pathlib import Path
import streamlit as st

def read_csv_file(uploaded_file):
    """Read an uploaded CSV file and return a pandas DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return None


def get_csv_columns(df):
    """Get a list of column names from a DataFrame."""
    if df is not None:
        return df.columns.tolist()
    return []


def preview_csv_data(df, max_rows=5):
    """Return a preview of the CSV data."""
    if df is not None:
        return df.head(max_rows)
    return None


def convert_to_jsonl(df, prompt_col, golden_answer_col, task_type, task_criteria, output_dir, name):
    """
    Convert CSV data to JSONL format for LLM benchmarking.
    
    Args:
        df: Pandas DataFrame with CSV data
        prompt_col: Column name for prompts
        golden_answer_col: Column name for golden answers
        task_type: Type of task for evaluation
        task_criteria: Criteria for evaluating the task
        output_dir: Directory to save the JSONL file
        name: Name for the evaluation
        
    Returns:
        Path to the created JSONL file
    """
    if df is None or prompt_col not in df.columns or golden_answer_col not in df.columns:
        st.error("Invalid CSV data or column names")
        return None
    
    # Create output directory if it doesn't exist
    prompt_eval_dir = Path(output_dir) / "prompt-evaluations"
    os.makedirs(prompt_eval_dir, exist_ok=True)
    
    # Generate JSONL file path
    jsonl_path = prompt_eval_dir / f"{name}.jsonl"
    
    # Convert DataFrame to JSONL format
    jsonl_data = []
    for _, row in df.iterrows():
        entry = {
            "text_prompt": row[prompt_col],
            "expected_output_tokens": 250,  # Default value
            "task": {
                "task_type": task_type,
                "task_criteria": task_criteria
            },
            "golden_answer": row[golden_answer_col]
        }
        jsonl_data.append(entry)
    
    # Write to JSONL file
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + '\n')
    
    return str(jsonl_path)


def create_model_profiles_jsonl(models, output_dir):
    """
    Create a JSONL file with model profiles.
    
    Args:
        models: List of dictionaries with model configuration
        output_dir: Directory to save the JSONL file
        
    Returns:
        Path to the created JSONL file
    """
    prompt_eval_dir = Path(output_dir) / "prompt-evaluations"
    os.makedirs(prompt_eval_dir, exist_ok=True)
    
    jsonl_path = prompt_eval_dir / "model_profiles.jsonl"
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for model in models:
            entry = {
                "model_id": model["id"],
                "region": model["region"],
                "inference_profile": "standard",
                "input_token_cost": model["input_cost"] * 1000,  # Convert to per 1000 tokens
                "output_token_cost": model["output_cost"] * 1000  # Convert to per 1000 tokens
            }
            f.write(json.dumps(entry) + '\n')
    
    return str(jsonl_path)


def create_judge_profiles_jsonl(judges, output_dir):
    """
    Create a JSONL file with judge model profiles.
    
    Args:
        judges: List of dictionaries with judge model configuration
        output_dir: Directory to save the JSONL file
        
    Returns:
        Path to the created JSONL file
    """
    prompt_eval_dir = Path(output_dir) / "prompt-evaluations"
    os.makedirs(prompt_eval_dir, exist_ok=True)
    
    jsonl_path = prompt_eval_dir / "judge_profiles.jsonl"
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for judge in judges:
            entry = {
                "model_id": judge["id"],
                "region": judge["region"],
                "input_cost_per_1k": judge["input_cost"] * 1000,  # Convert to per 1000 tokens
                "output_cost_per_1k": judge["output_cost"] * 1000  # Convert to per 1000 tokens
            }
            f.write(json.dumps(entry) + '\n')
    
    return str(jsonl_path)