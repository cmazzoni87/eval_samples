"""Utilities for running benchmark evaluations from the dashboard."""

import subprocess
import threading
import time
import os
import json
from pathlib import Path
import streamlit as st
from .state_management import update_evaluation_status
from .constants import DEFAULT_OUTPUT_DIR
from .csv_processor import (
    convert_to_jsonl, 
    create_model_profiles_jsonl, 
    create_judge_profiles_jsonl
)

# Store evaluation configs locally for thread safety
_thread_local_evaluations = {}

def run_benchmark_async(evaluation_config):
    """
    Run a benchmark evaluation asynchronously in a separate thread.
    
    Args:
        evaluation_config: Dictionary with evaluation configuration
    """
    # Store a copy of the evaluation config for thread safety
    eval_id = evaluation_config["id"]
    _thread_local_evaluations[eval_id] = evaluation_config.copy()
    
    # Create a thread to run the benchmark
    thread = threading.Thread(
        target=run_benchmark_process,
        args=(eval_id,)
    )
    thread.daemon = True
    thread.start()
    
    # Update status in the main thread before starting the background process
    if "evaluations" in st.session_state:
        update_evaluation_status(eval_id, "running", 5)


def run_benchmark_process(eval_id):
    """
    Run the benchmark evaluation in a subprocess.
    
    Args:
        eval_id: ID of the evaluation to run
    """
    # Get the evaluation config from thread-local storage
    if eval_id not in _thread_local_evaluations:
        print(f"Error: Evaluation {eval_id} not found in thread-local storage")
        return
        
    evaluation_config = _thread_local_evaluations[eval_id]
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(evaluation_config["output_dir"])
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a status file to track progress
        status_file = output_dir / f"eval_{eval_id}_status.json"
        _update_status_file(status_file, "running", 5)
        
        # Convert CSV data to JSONL
        _update_status_file(status_file, "running", 10)
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
            _update_status_file(status_file, "failed", 0)
            return
        
        # Create model profiles JSONL
        _update_status_file(status_file, "running", 15)
        models_jsonl = create_model_profiles_jsonl(
            evaluation_config["selected_models"],
            evaluation_config["output_dir"]
        )
        
        # Create judge profiles JSONL
        _update_status_file(status_file, "running", 20)
        judges_jsonl = create_judge_profiles_jsonl(
            evaluation_config["judge_models"],
            evaluation_config["output_dir"]
        )
        
        # Prepare command arguments
        jsonl_filename = os.path.basename(jsonl_path)
        _update_status_file(status_file, "running", 25)
        
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
        _update_status_file(status_file, "running", 30)
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
            current_status = _read_status_file(status_file)
            current_progress = current_status.get("progress", 30)
            if current_progress < 95:
                new_progress = min(current_progress + 5, 95)
                _update_status_file(status_file, "running", new_progress)
            
            time.sleep(5)
        
        # Get command output
        stdout, stderr = process.communicate()
        
        # Check for errors
        if process.returncode != 0:
            _update_status_file(status_file, "failed", 0)
            print(f"Benchmark failed: {stderr}")
            return
        
        # Update status to completed
        _update_status_file(status_file, "completed", 100)
        
        # Find the latest HTML report
        reports = list(output_dir.glob("llm_benchmark_report_*.html"))
        if reports:
            latest_report = max(reports, key=os.path.getmtime)
            # Update the status file with the report path
            _update_status_file(status_file, "completed", 100, str(latest_report))
    
    except Exception as e:
        _update_status_file(status_file, "failed", 0)
        print(f"Error running benchmark: {str(e)}")
    finally:
        # Clean up thread-local storage
        if eval_id in _thread_local_evaluations:
            del _thread_local_evaluations[eval_id]


def _update_status_file(status_file, status, progress, results=None):
    """Update the status file with the current status."""
    status_data = {
        "status": status,
        "progress": progress,
        "updated_at": time.time()
    }
    if results:
        status_data["results"] = results
        
    with open(status_file, 'w') as f:
        json.dump(status_data, f)


def _read_status_file(status_file):
    """Read the status file."""
    if not status_file.exists():
        return {"status": "unknown", "progress": 0}
    
    try:
        with open(status_file, 'r') as f:
            return json.load(f)
    except:
        return {"status": "unknown", "progress": 0}


def sync_evaluations_from_files():
    """
    Sync evaluation statuses from status files.
    Call this function periodically from the main thread.
    """
    if "evaluations" not in st.session_state:
        return
        
    # Get all evaluations
    evaluations = st.session_state.evaluations
    
    for eval_config in evaluations:
        eval_id = eval_config["id"]
        output_dir = Path(eval_config.get("output_dir", "benchmark_results"))
        status_file = output_dir / f"eval_{eval_id}_status.json"
        
        if status_file.exists():
            status_data = _read_status_file(status_file)
            
            # Update evaluation status in session state
            update_evaluation_status(
                eval_id, 
                status_data.get("status", eval_config["status"]),
                status_data.get("progress", eval_config["progress"])
            )
            
            # Update results if available
            if "results" in status_data and status_data["results"]:
                for i, e in enumerate(st.session_state.evaluations):
                    if e["id"] == eval_id:
                        st.session_state.evaluations[i]["results"] = status_data["results"]
                        break


def get_evaluation_progress(eval_id):
    """Get the progress of an evaluation."""
    # First try to get from session state
    if "evaluations" in st.session_state:
        for eval_config in st.session_state.evaluations:
            if eval_config["id"] == eval_id:
                return eval_config["progress"]
    
    # If not in session state, try status file
    for output_dir in [Path("benchmark_results"), Path(DEFAULT_OUTPUT_DIR)]:
        status_file = output_dir / f"eval_{eval_id}_status.json"
        if status_file.exists():
            status_data = _read_status_file(status_file)
            return status_data.get("progress", 0)
    
    return 0