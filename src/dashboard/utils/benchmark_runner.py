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
dashboard_log_file = os.path.join(DASHBOARD_LOG_DIR, f'dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Configure root logger for dashboard
dashboard_logger = logging.getLogger('dashboard')
dashboard_logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler(dashboard_log_file)
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)
dashboard_logger.addHandler(file_handler)

# Stream handler for console output
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_format = logging.Formatter('%(levelname)s - %(message)s')
stream_handler.setFormatter(stream_format)
dashboard_logger.addHandler(stream_handler)

dashboard_logger.info(f"Dashboard logger initialized. Log file: {dashboard_log_file}")

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
        
        # Create log files
        stdout_log = logs_dir / "stdout.log"
        stderr_log = logs_dir / "stderr.log"
        
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
        
        # Run the benchmark command with log file redirects
        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Run from src directory
                )
                
                dashboard_logger.info(f"Started subprocess with PID {process.pid}")
                
                # Monitor evaluation state by checking logs and reports
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
                    
                    # Check for error indicators in stderr
                    stderr_content = ""
                    if os.path.exists(stderr_log) and os.path.getsize(stderr_log) > 0:
                        with open(stderr_log, 'r') as f:
                            stderr_content = f.read()
                    
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
                
                # Process completed - check final status
                return_code = process.wait()
                
                # Check for errors
                if return_code != 0:
                    with open(stderr_log, 'r') as f:
                        stderr_content = f.read()
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