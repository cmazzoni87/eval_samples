"""Utilities for running benchmark evaluations from the dashboard."""

import subprocess
import threading
import time
import os
from pathlib import Path
import streamlit as st
from .state_management import update_evaluation_status
from .csv_processor import (
    convert_to_jsonl, 
    create_model_profiles_jsonl, 
    create_judge_profiles_jsonl
)

def run_benchmark_async(evaluation_config):
    """
    Run a benchmark evaluation asynchronously in a separate thread.
    
    Args:
        evaluation_config: Dictionary with evaluation configuration
    """
    # Create a thread to run the benchmark
    thread = threading.Thread(
        target=run_benchmark_process,
        args=(evaluation_config,)
    )
    thread.daemon = True
    thread.start()


def run_benchmark_process(evaluation_config):
    """
    Run the benchmark evaluation in a subprocess.
    
    Args:
        evaluation_config: Dictionary with evaluation configuration
    """
    eval_id = evaluation_config["id"]
    
    try:
        # Update status to running
        update_evaluation_status(eval_id, "running", 5)
        
        # Create output directory if it doesn't exist
        output_dir = Path(evaluation_config["output_dir"])
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert CSV data to JSONL
        update_evaluation_status(eval_id, "running", 10)
        jsonl_path = convert_to_jsonl(
            evaluation_config["csv_data"],
            evaluation_config["prompt_column"],
            evaluation_config["golden_answer_column"],
            evaluation_config["task_type"],
            evaluation_config["task_criteria"],
            evaluation_config["output_dir"],
            evaluation_config["name"]
        )
        
        if not jsonl_path:
            update_evaluation_status(eval_id, "failed", 0)
            return
        
        # Create model profiles JSONL
        update_evaluation_status(eval_id, "running", 15)
        models_jsonl = create_model_profiles_jsonl(
            evaluation_config["selected_models"],
            evaluation_config["output_dir"]
        )
        
        # Create judge profiles JSONL
        update_evaluation_status(eval_id, "running", 20)
        judges_jsonl = create_judge_profiles_jsonl(
            evaluation_config["judge_models"],
            evaluation_config["output_dir"]
        )
        
        # Prepare command arguments
        jsonl_filename = os.path.basename(jsonl_path)
        update_evaluation_status(eval_id, "running", 25)
        
        cmd = [
            "python", 
            "src/benchmarks_run.py",
            jsonl_filename,
            "--output_dir", evaluation_config["output_dir"],
            "--parallel_calls", str(evaluation_config["parallel_calls"]),
            "--invocations_per_scenario", str(evaluation_config["invocations_per_scenario"]),
            "--sleep_between_invocations", str(evaluation_config["sleep_between_invocations"]),
            "--experiment_counts", str(evaluation_config["experiment_counts"]),
            "--experiment_name", evaluation_config["name"],
            "--temperature_variations", str(evaluation_config["temperature_variations"])
        ]
        
        if evaluation_config["user_defined_metrics"]:
            cmd.extend(["--user_defined_metrics", evaluation_config["user_defined_metrics"]])
        
        # Run the benchmark command
        update_evaluation_status(eval_id, "running", 30)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor progress
        while True:
            # Check if process is still running
            if process.poll() is not None:
                break
            
            # Simulate progress updates (could be more sophisticated with actual progress parsing)
            current_progress = evaluation_config.get("progress", 30)
            if current_progress < 95:
                new_progress = min(current_progress + 5, 95)
                update_evaluation_status(eval_id, "running", new_progress)
            
            time.sleep(5)
        
        # Get command output
        stdout, stderr = process.communicate()
        
        # Check for errors
        if process.returncode != 0:
            update_evaluation_status(eval_id, "failed", 0)
            print(f"Benchmark failed: {stderr}")
            return
        
        # Update status to completed
        update_evaluation_status(eval_id, "completed", 100)
        
        # Find the latest HTML report
        reports = list(output_dir.glob("llm_benchmark_report_*.html"))
        if reports:
            latest_report = max(reports, key=os.path.getmtime)
            for i, eval_config in enumerate(st.session_state.evaluations):
                if eval_config["id"] == eval_id:
                    st.session_state.evaluations[i]["results"] = str(latest_report)
                    break
    
    except Exception as e:
        update_evaluation_status(eval_id, "failed", 0)
        print(f"Error running benchmark: {str(e)}")


def get_evaluation_progress(eval_id):
    """Get the progress of an evaluation."""
    for eval_config in st.session_state.evaluations:
        if eval_config["id"] == eval_id:
            return eval_config["progress"]
    return 0