"""Utilities for running benchmark evaluations from the dashboard."""

import subprocess
import threading
import time
import os
import json
import logging
import sys
from pathlib import Path
import streamlit as st
from datetime import datetime
from .state_management import update_evaluation_status
from .constants import DEFAULT_OUTPUT_DIR
from .csv_processor import (
    convert_to_jsonl, 
    create_model_profiles_jsonl, 
    create_judge_profiles_jsonl
)

# Set up dashboard logger
from .constants import PROJECT_ROOT
DASHBOARD_LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(DASHBOARD_LOG_DIR, exist_ok=True)

# Configure root logger for dashboard using in-memory logging
dashboard_logger = logging.getLogger('dashboard')
dashboard_logger.setLevel(logging.DEBUG)

# Use in-memory logging with an in-memory buffer
from io import StringIO
# Global variables to store stdout/stderr captures
stdout_capture = StringIO()
stderr_capture = StringIO()
memory_handler = logging.StreamHandler(StringIO())
memory_handler.setLevel(logging.DEBUG)
memory_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
memory_handler.setFormatter(memory_format)
dashboard_logger.addHandler(memory_handler)

# Stream handler for console output
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_format = logging.Formatter('%(levelname)s - %(message)s')
stream_handler.setFormatter(stream_format)
dashboard_logger.addHandler(stream_handler)

dashboard_logger.info("Dashboard logger initialized (in-memory mode)")

# Store evaluation configs locally for thread safety
_thread_local_evaluations = {}

# Store pending evaluations for merging
_pending_evaluations = []
_merged_evaluation_running = False

# Sequential execution queue and control variables
_sequential_queue = []
_sequential_running = False
_sequential_thread = None

def merge_evaluations(evaluation_configs):
    """
    Merge multiple evaluation configurations into a single evaluation.
    
    Args:
        evaluation_configs: List of evaluation configuration dictionaries
        
    Returns:
        Dictionary with merged evaluation configuration
    """
    if not evaluation_configs:
        dashboard_logger.error("No evaluations to merge")
        return None
    
    # Create a new unique ID for the merged evaluation
    from uuid import uuid4
    merged_id = str(uuid4())
    
    # Use the first evaluation as the base for configuration parameters
    base_config = evaluation_configs[0].copy()
    
    # Combine all CSV dataframes
    import pandas as pd
    merged_df = pd.DataFrame()
    for config in evaluation_configs:
        if merged_df.empty:
            merged_df = config["csv_data"].copy()
        else:
            merged_df = pd.concat([merged_df, config["csv_data"]], ignore_index=True)
    
    # Create a merged name incorporating task types
    evaluation_names = [config["name"] for config in evaluation_configs]
    task_types = [config.get("task_type", "") for config in evaluation_configs if config.get("task_type")]
    
    # Create a more descriptive name
    if task_types:
        # Use up to 3 unique task types in the name
        unique_types = sorted(set(task_types))
        task_summary = "_".join(unique_types[:3])
        if len(unique_types) > 3:
            task_summary += "_etc"
        merged_name = f"merged_{len(evaluation_configs)}_{task_summary}"
    else:
        # Fallback to generic name
        merged_name = f"merged_{len(evaluation_configs)}_evaluations"
    
    # Get the union of all selected models across evaluations
    all_models = set()
    for config in evaluation_configs:
        for model in config["selected_models"]:
            model_key = (model["id"], model["region"])
            all_models.add(model_key)
    
    # Convert back to list of dictionaries
    merged_models = []
    for model_id, region in all_models:
        # Find this model in any of the configs to get the costs
        for config in evaluation_configs:
            for model in config["selected_models"]:
                if model["id"] == model_id and model["region"] == region:
                    merged_models.append(model)
                    break
            else:
                continue
            break
    
    # Get the union of all judge models
    all_judges = set()
    for config in evaluation_configs:
        for judge in config["judge_models"]:
            judge_key = (judge["id"], judge["region"])
            all_judges.add(judge_key)
    
    # Convert back to list of dictionaries
    merged_judges = []
    for judge_id, region in all_judges:
        # Find this judge in any of the configs to get the costs
        for config in evaluation_configs:
            for judge in config["judge_models"]:
                if judge["id"] == judge_id and judge["region"] == region:
                    merged_judges.append(judge)
                    break
            else:
                continue
            break
    
    # Collect unique task types and task criteria
    unique_task_types = set()
    unique_task_criteria = set()
    for config in evaluation_configs:
        if config.get("task_type"):
            unique_task_types.add(config["task_type"])
        if config.get("task_criteria"):
            unique_task_criteria.add(config["task_criteria"])
    
    # Join unique task types and criteria
    combined_task_type = ", ".join(sorted(unique_task_types)) if unique_task_types else base_config["task_type"]
    combined_task_criteria = ", ".join(sorted(unique_task_criteria)) if unique_task_criteria else base_config["task_criteria"]
    
    # Create merged configuration
    merged_config = {
        "id": merged_id,
        "name": merged_name,
        "created_at": pd.Timestamp.now().isoformat(),
        "status": "pending",
        "progress": 0,
        "csv_data": merged_df,
        "prompt_column": base_config["prompt_column"],
        "golden_answer_column": base_config["golden_answer_column"],
        "task_type": combined_task_type,
        "task_criteria": combined_task_criteria,
        "selected_models": merged_models,
        "judge_models": merged_judges,
        # Use lowest values of concurrency parameters to be conservative
        "parallel_calls": min([config.get("parallel_calls", 4) for config in evaluation_configs]),
        "invocations_per_scenario": min([config.get("invocations_per_scenario", 2) for config in evaluation_configs]),
        "sleep_between_invocations": max([config.get("sleep_between_invocations", 3) for config in evaluation_configs]),
        "experiment_counts": min([config.get("experiment_counts", 1) for config in evaluation_configs]),
        "temperature_variations": min([config.get("temperature_variations", 0) for config in evaluation_configs]),
        "user_defined_metrics": base_config.get("user_defined_metrics", None)
    }
    
    dashboard_logger.info(f"Created merged evaluation '{merged_name}' with ID {merged_id}")
    dashboard_logger.info(f"Merged {len(evaluation_configs)} evaluations with {len(merged_df)} total prompts")
    dashboard_logger.info(f"Using {len(merged_models)} models and {len(merged_judges)} judge models")
    dashboard_logger.info(f"Combined task types: {combined_task_type}")
    dashboard_logger.info(f"Combined task criteria: {combined_task_criteria}")
    
    return merged_config


def run_merged_evaluations(pending_evals=None):
    """
    Run merged evaluations from the pending list.
    
    Args:
        pending_evals: Optional list of evaluation IDs to merge and run
    """
    global _pending_evaluations, _merged_evaluation_running
    
    # If merged evaluation is already running, don't start another
    if _merged_evaluation_running:
        dashboard_logger.warning("A merged evaluation is already running. Skipping.")
        return
    
    try:
        _merged_evaluation_running = True
        
        # Use provided pending evaluations or the global list
        evaluations_to_merge = []
        
        if pending_evals:
            # Find the evaluation configs for the provided IDs
            for eval_id in pending_evals:
                for eval_config in _pending_evaluations:
                    if eval_config["id"] == eval_id:
                        evaluations_to_merge.append(eval_config)
                        break
        else:
            # Use all pending evaluations
            evaluations_to_merge = _pending_evaluations.copy()
            _pending_evaluations = []  # Clear the global list
        
        if not evaluations_to_merge:
            dashboard_logger.warning("No evaluations to merge and run")
            _merged_evaluation_running = False
            return
        
        # Merge the evaluations
        merged_config = merge_evaluations(evaluations_to_merge)
        if not merged_config:
            dashboard_logger.error("Failed to create merged evaluation")
            _merged_evaluation_running = False
            return
        
        # Get the IDs of the evaluations that were merged
        merged_eval_ids = [eval_config["id"] for eval_config in evaluations_to_merge]
        dashboard_logger.info(f"Merged evaluation IDs: {merged_eval_ids}")
        
        # Add the merged evaluation to the session state and remove the original evaluations
        if hasattr(st, 'session_state') and 'evaluations' in st.session_state:
            # Remove the original evaluations from session state
            st.session_state.evaluations = [
                eval_config for eval_config in st.session_state.evaluations 
                if eval_config["id"] not in merged_eval_ids
            ]
            
            # Add the merged evaluation
            st.session_state.evaluations.append(merged_config)
            
            # Also remove from active_evaluations and completed_evaluations lists
            if hasattr(st.session_state, 'active_evaluations'):
                st.session_state.active_evaluations = [
                    eval_config for eval_config in st.session_state.active_evaluations 
                    if eval_config["id"] not in merged_eval_ids
                ]
            
            if hasattr(st.session_state, 'completed_evaluations'):
                st.session_state.completed_evaluations = [
                    eval_config for eval_config in st.session_state.completed_evaluations 
                    if eval_config["id"] not in merged_eval_ids
                ]
                
            dashboard_logger.info(f"Removed {len(merged_eval_ids)} original evaluations from UI lists")
        
        # Run the merged evaluation
        run_benchmark_async(merged_config)
        
        dashboard_logger.info(f"Started merged evaluation '{merged_config['name']}' with ID {merged_config['id']}")
    
    except Exception as e:
        dashboard_logger.exception(f"Error running merged evaluations: {str(e)}")
    finally:
        _merged_evaluation_running = False


def add_to_pending_evaluations(evaluation_config):
    """
    Add an evaluation to the pending list for later merging.
    
    Args:
        evaluation_config: Evaluation configuration dictionary
    
    Returns:
        Boolean indicating success
    """
    global _pending_evaluations
    
    try:
        # Check if evaluation is already in pending list
        eval_id = evaluation_config["id"]
        if any(eval_config["id"] == eval_id for eval_config in _pending_evaluations):
            dashboard_logger.warning(f"Evaluation '{evaluation_config['name']}' already in pending list")
            return True
            
        # Make a copy to avoid reference issues
        _pending_evaluations.append(evaluation_config.copy())
        dashboard_logger.info(f"Added evaluation '{evaluation_config['name']}' to pending list (total: {len(_pending_evaluations)})")
        
        # Add a flag to the evaluation to mark it as pending for merge
        # This helps the UI show it differently
        if hasattr(st, 'session_state') and 'evaluations' in st.session_state:
            for i, eval_config in enumerate(st.session_state.evaluations):
                if eval_config["id"] == eval_id:
                    st.session_state.evaluations[i]["pending_merge"] = True
                    dashboard_logger.debug(f"Marked evaluation {eval_id} as pending_merge=True")
                    break
                    
        return True
    except Exception as e:
        dashboard_logger.error(f"Error adding evaluation to pending list: {str(e)}")
        return False
        

def run_benchmark_async(evaluation_config):
    """
    Run a benchmark evaluation asynchronously in a separate thread.
    
    Args:
        evaluation_config: Dictionary with evaluation configuration
    """
    # Store a copy of the evaluation config for thread safety
    eval_id = evaluation_config["id"]
    _thread_local_evaluations[eval_id] = evaluation_config.copy()
    
    dashboard_logger.info(f"Starting benchmark evaluation: {evaluation_config['name']} (ID: {eval_id})")
    dashboard_logger.debug(f"Evaluation configuration: {json.dumps({k: v for k, v in evaluation_config.items() if k != 'csv_data'})}")
    
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
        dashboard_logger.info(f"Updated evaluation status for {eval_id} to 'running'")


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
        # Get project root
        from .constants import PROJECT_ROOT, DEFAULT_OUTPUT_DIR
        
        # Create output directory if it doesn't exist (using absolute path)
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create logs directory for this evaluation - use PROJECT_ROOT/logs
        from .constants import PROJECT_ROOT
        logs_dir = Path(PROJECT_ROOT) / "logs" / f"eval_{eval_id}"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create a status file to track progress
        status_file = output_dir / f"eval_{eval_id}_status.json"
        _update_status_file(status_file, "in-progress", 0, logs_dir=str(logs_dir))
        
        # Start time to track session evaluations
        eval_start_time = time.time()
        _update_status_file(status_file, "in-progress", 0, logs_dir=str(logs_dir), start_time=eval_start_time)
        
        # Convert CSV data to JSONL
        dashboard_logger.info(f"Converting CSV data to JSONL for evaluation {eval_id}")
        try:
            jsonl_path = convert_to_jsonl(
                evaluation_config["csv_data"],
                evaluation_config["prompt_column"],
                evaluation_config["golden_answer_column"],
                evaluation_config["task_type"],
                evaluation_config["task_criteria"],
                "",
                evaluation_config["name"]
            )
            if not jsonl_path:
                dashboard_logger.error(f"Failed to convert CSV data to JSONL for evaluation {eval_id}")
                _update_status_file(status_file, "failed", 0, error="Failed to convert CSV data to JSONL format")
                return
            dashboard_logger.info(f"Successfully created JSONL file at {jsonl_path}")
        except Exception as e:
            dashboard_logger.exception(f"Exception while converting CSV data to JSONL: {str(e)}")
            _update_status_file(status_file, "failed", 0, error=f"CSV conversion error: {str(e)}")
            return
        
        # Create unique model profiles JSONL for this evaluation
        dashboard_logger.info(f"Creating model profiles JSONL for evaluation {eval_id}")
        try:
            # Generate unique filenames for this evaluation
            model_file_name = f"model_profiles_{eval_id}.jsonl"
            judge_file_name = f"judge_profiles_{eval_id}.jsonl"
            
            models_jsonl = create_model_profiles_jsonl(
                evaluation_config["selected_models"],
                "",
                custom_filename=model_file_name
            )
            dashboard_logger.info(f"Successfully created model profiles at {models_jsonl}")
        except Exception as e:
            dashboard_logger.exception(f"Exception while creating model profiles: {str(e)}")
            _update_status_file(status_file, "failed", 0, error=f"Model profiles error: {str(e)}")
            return
        
        # Create unique judge profiles JSONL
        dashboard_logger.info(f"Creating judge profiles JSONL for evaluation {eval_id}")
        try:
            judges_jsonl = create_judge_profiles_jsonl(
                evaluation_config["judge_models"],
                "",
                custom_filename=judge_file_name
            )
            dashboard_logger.info(f"Successfully created judge profiles at {judges_jsonl}")
        except Exception as e:
            dashboard_logger.exception(f"Exception while creating judge profiles: {str(e)}")
            _update_status_file(status_file, "failed", 0, error=f"Judge profiles error: {str(e)}")
            return
        
        # Prepare command arguments
        jsonl_filename = os.path.basename(jsonl_path)
        
        # Create in-memory buffers instead of log files
        # Get current script directory for reliable relative paths
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        cmd = [
            "python", 
            os.path.join(script_dir, "benchmarks_run.py"),
            jsonl_filename,
            "--output_dir", str(output_dir),
            "--report", "False",
            "--parallel_calls", str(evaluation_config["parallel_calls"]),
            "--invocations_per_scenario", str(evaluation_config["invocations_per_scenario"]),
            "--sleep_between_invocations", str(evaluation_config["sleep_between_invocations"]),
            "--experiment_counts", str(evaluation_config["experiment_counts"]),
            "--experiment_name", evaluation_config["name"],
            "--temperature_variations", str(evaluation_config["temperature_variations"]),
            "--model_file_name", model_file_name,
            "--judge_file_name", judge_file_name
        ]
        
        if evaluation_config["user_defined_metrics"]:
            cmd.extend(["--user_defined_metrics", evaluation_config["user_defined_metrics"]])

        # Log the command being executed
        dashboard_logger.info(f"Executing benchmark command for evaluation {eval_id}:")
        dashboard_logger.info(" ".join(cmd))
        
        # Create stdout/stderr capture variables
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        # Run the benchmark command with output capture
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Run from src directory
            )
            
            dashboard_logger.info(f"Started subprocess with PID {process.pid}")
            
            # Set up threads to read process output
            def read_stdout():
                for line in iter(process.stdout.readline, ''):
                    stdout_capture.write(line)
                    dashboard_logger.debug(f"STDOUT: {line.strip()}")
            
            def read_stderr():
                for line in iter(process.stderr.readline, ''):
                    stderr_capture.write(line)
                    dashboard_logger.error(f"STDERR: {line.strip()}")
            
            stdout_thread = threading.Thread(target=read_stdout)
            stderr_thread = threading.Thread(target=read_stderr)
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Monitor evaluation state
            poll_count = 0
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    dashboard_logger.info(f"Process completed with return code {process.returncode}")
                    break
                
                # Periodically log that we're still monitoring the process
                if poll_count % 6 == 0:  # Every minute (6 * 10 seconds)
                    dashboard_logger.info(f"Process {process.pid} still running (poll count: {poll_count})")
                    
                    # Periodically check for reports being generated
                    reports = list(output_dir.glob(f"*{evaluation_config['name']}*.html"))
                    if reports:
                        dashboard_logger.info(f"Found {len(reports)} HTML reports while process is running")
                
                # Check for error indicators in captured stderr
                stderr_content = stderr_capture.getvalue()
                
                # Look for critical errors
                if "Error:" in stderr_content or "Exception:" in stderr_content:
                    error_excerpt = stderr_content.split("Error:")[1].strip()[:500] if "Error:" in stderr_content else stderr_content.split("Exception:")[1].strip()[:500]
                    dashboard_logger.error(f"Detected error in stderr: {error_excerpt}")
                    _update_status_file(status_file, "failed", 0, 
                                      logs_dir=str(logs_dir), 
                                      error=stderr_content[:500])
                    process.terminate()
                    dashboard_logger.info(f"Terminated process {process.pid} due to error")
                    break
                
                # Wait before checking again
                time.sleep(10)
                poll_count += 1
            
            # Make sure we've read all output
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            # Process completed - check final status
            return_code = process.wait()
            
            # Check for errors
            if return_code != 0:
                stderr_content = stderr_capture.getvalue()
                dashboard_logger.error(f"Process failed with return code {return_code}")
                dashboard_logger.error(f"Error content: {stderr_content[:1000]}")
                _update_status_file(status_file, "failed", 0, 
                                  logs_dir=str(logs_dir),
                                  error=stderr_content[:500])
                return
            
            dashboard_logger.info(f"Process completed successfully")
            
        except Exception as e:
            dashboard_logger.exception(f"Exception during subprocess execution: {str(e)}")
            _update_status_file(status_file, "failed", 0, 
                              logs_dir=str(logs_dir),
                              error=f"Subprocess error: {str(e)}")
            return
        
        # Find the latest HTML report
        reports = list(output_dir.glob(f"*{evaluation_config['name']}*.html"))
        if reports:
            latest_report = max(reports, key=os.path.getmtime)
            # Update the status file with the report path
            _update_status_file(status_file, "completed", 100, 
                               logs_dir=str(logs_dir),
                               results=str(latest_report),
                               end_time=time.time())
        else:
            _update_status_file(status_file, "completed", 100, 
                               logs_dir=str(logs_dir),
                               end_time=time.time())
    
    except Exception as e:
        _update_status_file(status_file, "failed", 0, 
                           logs_dir=str(logs_dir) if 'logs_dir' in locals() else None,
                           error=str(e))
        print(f"Error running benchmark: {str(e)}")
    finally:
        # Clean up thread-local storage
        if eval_id in _thread_local_evaluations:
            del _thread_local_evaluations[eval_id]


def _update_status_file(status_file, status, progress, results=None, logs_dir=None, error=None, start_time=None, end_time=None):
    """
    Update the status file with the current status.
    
    Args:
        status_file: Path to the status file
        status: Current status (in-progress, failed, completed)
        progress: Progress percentage (0-100)
        results: Path to results file if available
        logs_dir: Directory containing log files
        error: Error message if status is failed
        start_time: Start time of the evaluation
        end_time: End time of the evaluation
    """
    status_data = {
        "status": status,
        "updated_at": time.time()
    }
    
    # Only include progress for backward compatibility
    if progress is not None:
        status_data["progress"] = progress
    
    # Add optional fields if provided
    if results:
        status_data["results"] = results
    if logs_dir:
        status_data["logs_dir"] = logs_dir
    if error:
        status_data["error"] = error
    if start_time:
        status_data["start_time"] = start_time
    if end_time:
        status_data["end_time"] = end_time
        status_data["duration"] = end_time - status_data.get("start_time", start_time or end_time)
    
    with open(status_file, 'w') as f:
        json.dump(status_data, f)


def add_to_sequential_queue(evaluation_configs):
    """
    Add evaluations to the sequential execution queue.
    
    Args:
        evaluation_configs: List of evaluation configuration dictionaries to run sequentially
    
    Returns:
        Boolean indicating success
    """
    global _sequential_queue, _sequential_running, _sequential_thread
    
    try:
        # Add evaluations to the queue
        for config in evaluation_configs:
            if config not in _sequential_queue:
                _sequential_queue.append(config.copy())
                dashboard_logger.info(f"Added evaluation '{config['name']}' to sequential queue (total: {len(_sequential_queue)})")
                
                # Mark the evaluation as queued in session state
                if hasattr(st, 'session_state') and 'evaluations' in st.session_state:
                    for i, eval_config in enumerate(st.session_state.evaluations):
                        if eval_config["id"] == config["id"]:
                            st.session_state.evaluations[i]["status"] = "queued"
                            st.session_state.evaluations[i]["queued_sequential"] = True
                            dashboard_logger.debug(f"Marked evaluation {config['id']} as queued_sequential=True")
                            break
        
        # Start the sequential execution thread if it's not already running
        if not _sequential_running:
            _sequential_thread = threading.Thread(
                target=_process_sequential_queue,
                args=()
            )
            _sequential_thread.daemon = True
            _sequential_thread.start()
            dashboard_logger.info("Started sequential execution thread")
        
        return True
    except Exception as e:
        dashboard_logger.error(f"Error adding evaluations to sequential queue: {str(e)}")
        return False


def _process_sequential_queue():
    """
    Process evaluations in the sequential queue one at a time.
    This function runs in a separate thread.
    """
    global _sequential_queue, _sequential_running
    
    # Set the running flag
    _sequential_running = True
    
    try:
        dashboard_logger.info(f"Sequential execution thread started with {len(_sequential_queue)} evaluations in queue")
        
        # Process evaluations one at a time
        while _sequential_queue:
            # Get the next evaluation from the queue
            eval_config = _sequential_queue.pop(0)
            eval_id = eval_config["id"]
            
            dashboard_logger.info(f"Starting sequential execution of evaluation '{eval_config['name']}' (ID: {eval_id})")
            dashboard_logger.info(f"Remaining in queue: {len(_sequential_queue)}")
            
            try:
                # Instead of using run_benchmark_async which starts a thread,
                # we'll directly create the necessary files first to avoid race conditions
                
                # Create a status file to track progress
                from .constants import DEFAULT_OUTPUT_DIR, PROJECT_ROOT
                output_dir = Path(DEFAULT_OUTPUT_DIR)
                os.makedirs(output_dir, exist_ok=True)
                
                # Create logs directory for this evaluation
                logs_dir = Path(PROJECT_ROOT) / "logs" / f"eval_{eval_id}"
                os.makedirs(logs_dir, exist_ok=True)
                
                # Create status file
                status_file = output_dir / f"eval_{eval_id}_status.json"
                _update_status_file(status_file, "in-progress", 0, logs_dir=str(logs_dir))
                
                # Start time to track session evaluations
                eval_start_time = time.time()
                _update_status_file(status_file, "in-progress", 0, logs_dir=str(logs_dir), start_time=eval_start_time)
                
                # Convert CSV data to JSONL
                dashboard_logger.info(f"Converting CSV data to JSONL for evaluation {eval_id}")
                try:
                    jsonl_path = convert_to_jsonl(
                        eval_config["csv_data"],
                        eval_config["prompt_column"],
                        eval_config["golden_answer_column"],
                        eval_config["task_type"],
                        eval_config["task_criteria"],
                        "",
                        eval_config["name"]
                    )
                    if not jsonl_path:
                        dashboard_logger.error(f"Failed to convert CSV data to JSONL for evaluation {eval_id}")
                        _update_status_file(status_file, "failed", 0, error="Failed to convert CSV data to JSONL format")
                        continue
                    dashboard_logger.info(f"Successfully created JSONL file at {jsonl_path}")
                except Exception as e:
                    dashboard_logger.exception(f"Exception while converting CSV data to JSONL: {str(e)}")
                    _update_status_file(status_file, "failed", 0, error=f"CSV conversion error: {str(e)}")
                    continue
                
                # Create model profiles JSONL
                dashboard_logger.info(f"Creating model profiles JSONL for evaluation {eval_id}")
                try:
                    model_file_name = f"model_profiles_{eval_id}.jsonl"
                    models_jsonl = create_model_profiles_jsonl(
                        eval_config["selected_models"],
                        "",
                        custom_filename=model_file_name
                    )
                    dashboard_logger.info(f"Successfully created model profiles at {models_jsonl}")
                except Exception as e:
                    dashboard_logger.exception(f"Exception while creating model profiles: {str(e)}")
                    _update_status_file(status_file, "failed", 0, error=f"Model profiles error: {str(e)}")
                    continue
                
                # Create judge profiles JSONL
                dashboard_logger.info(f"Creating judge profiles JSONL for evaluation {eval_id}")
                try:
                    judge_file_name = f"judge_profiles_{eval_id}.jsonl"
                    judges_jsonl = create_judge_profiles_jsonl(
                        eval_config["judge_models"],
                        "",
                        custom_filename=judge_file_name
                    )
                    dashboard_logger.info(f"Successfully created judge profiles at {judges_jsonl}")
                except Exception as e:
                    dashboard_logger.exception(f"Exception while creating judge profiles: {str(e)}")
                    _update_status_file(status_file, "failed", 0, error=f"Judge profiles error: {str(e)}")
                    continue
                
                # Now that all files are created, run the benchmark async
                dashboard_logger.info(f"All files created successfully, starting benchmark for {eval_id}")
                run_benchmark_async(eval_config)
                
                # Update evaluation status in session state
                if hasattr(st, 'session_state') and 'evaluations' in st.session_state:
                    for i, e in enumerate(st.session_state.evaluations):
                        if e["id"] == eval_id:
                            st.session_state.evaluations[i]["queued_sequential"] = False
                            break
                
                # Wait for the evaluation to complete before starting the next one
                # status_file is already defined and created above
                dashboard_logger.info(f"Waiting for evaluation '{eval_config['name']}' to complete...")
                
                # Wait for the evaluation to start (status file to be created)
                wait_start = time.time()
                while not status_file.exists() and time.time() - wait_start < 60:  # Wait up to 60 seconds
                    time.sleep(2)
                
                if not status_file.exists():
                    dashboard_logger.warning(f"Status file not created for evaluation '{eval_config['name']}' after 60 seconds")
                    continue
                
                # Wait for the evaluation to complete
                while True:
                    status_data = _read_status_file(status_file)
                    current_status = status_data.get("status", "unknown")
                    
                    if current_status in ["completed", "failed"]:
                        dashboard_logger.info(f"Evaluation '{eval_config['name']}' {current_status}")
                        break
                    
                    # Log progress periodically
                    dashboard_logger.debug(f"Evaluation '{eval_config['name']}' status: {current_status}, progress: {status_data.get('progress', 0)}%")
                    time.sleep(10)  # Check every 10 seconds
                
                # Add a small delay between evaluations to ensure clean separation
                time.sleep(5)
                dashboard_logger.info(f"Moving to next evaluation in sequential queue. Remaining: {len(_sequential_queue)}")
                
            except Exception as e:
                dashboard_logger.error(f"Error executing evaluation '{eval_config['name']}' in sequential queue: {str(e)}")
                # Continue with the next evaluation
        
        dashboard_logger.info("Sequential execution queue completed")
    
    except Exception as e:
        dashboard_logger.exception(f"Error in sequential execution thread: {str(e)}")
    finally:
        # Reset the running flag
        _sequential_running = False


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
    # Make sure session state is initialized
    if "evaluations" not in st.session_state:
        dashboard_logger.warning("No evaluations found in session state")
        print("No evaluations found in session state")
        return
        
    # Get all evaluations and print for debugging
    evaluations = st.session_state.evaluations
    dashboard_logger.info(f"Syncing status for {len(evaluations)} evaluations")
    print(f"Syncing status for {len(evaluations)} evaluations with IDs: {[e['id'] for e in evaluations]}")
    
    for eval_config in evaluations:
        eval_id = eval_config["id"]
        # Use the absolute path from constants
        from .constants import DEFAULT_OUTPUT_DIR
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        status_file = output_dir / f"eval_{eval_id}_status.json"
        
        dashboard_logger.debug(f"Checking status file for evaluation {eval_id}: {status_file}")
        
        if status_file.exists():
            dashboard_logger.debug(f"Status file found for evaluation {eval_id}")
            status_data = _read_status_file(status_file)
            
            # Log status changes
            old_status = eval_config.get("status", "unknown")
            new_status = status_data.get("status", old_status)
            
            if old_status != new_status:
                dashboard_logger.info(f"Evaluation {eval_id} status changed: {old_status} -> {new_status}")
            
            # Update evaluation status in session state
            update_evaluation_status(
                eval_id, 
                new_status,
                status_data.get("progress", eval_config.get("progress", 0))
            )
            
            # Update additional fields from status file
            for key in ["logs_dir", "error", "start_time", "end_time", "duration"]:
                if key in status_data:
                    for i, e in enumerate(st.session_state.evaluations):
                        if e["id"] == eval_id:
                            st.session_state.evaluations[i][key] = status_data[key]
            
            # Update results if available
            if "results" in status_data and status_data["results"]:
                dashboard_logger.info(f"Results found for evaluation {eval_id}: {status_data['results']}")
                for i, e in enumerate(st.session_state.evaluations):
                    if e["id"] == eval_id:
                        st.session_state.evaluations[i]["results"] = status_data["results"]
                        break
        else:
            dashboard_logger.debug(f"No status file found for evaluation {eval_id}")
            
    dashboard_logger.info("Status sync completed")


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