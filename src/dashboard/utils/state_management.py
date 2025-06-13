"""Session state management for the Streamlit dashboard."""

import streamlit as st
import uuid
from datetime import datetime
import os
from .constants import (
    DEFAULT_OUTPUT_DIR, DEFAULT_PARALLEL_CALLS, 
    DEFAULT_INVOCATIONS_PER_SCENARIO, DEFAULT_SLEEP_BETWEEN_INVOCATIONS,
    DEFAULT_EXPERIMENT_COUNTS, DEFAULT_TEMPERATURE_VARIATIONS
)

def initialize_session_state():
    """Initialize all session state variables with default values."""
    # Initialize evaluation settings
    if "evaluations" not in st.session_state:
        st.session_state.evaluations = []
    
    if "active_evaluations" not in st.session_state:
        st.session_state.active_evaluations = []
    
    if "completed_evaluations" not in st.session_state:
        st.session_state.completed_evaluations = []
    
    if "current_evaluation_config" not in st.session_state:
        st.session_state.current_evaluation_config = {
            "id": None,
            "name": f"Benchmark-{datetime.now().strftime("%Y%m%d%H%M%S")}",
            "csv_data": None,
            "prompt_column": None,
            "golden_answer_column": None,
            "task_type": "",
            "task_criteria": "",
            "output_dir": DEFAULT_OUTPUT_DIR,
            "parallel_calls": DEFAULT_PARALLEL_CALLS,
            "invocations_per_scenario": DEFAULT_INVOCATIONS_PER_SCENARIO,
            "sleep_between_invocations": DEFAULT_SLEEP_BETWEEN_INVOCATIONS,
            "experiment_counts": DEFAULT_EXPERIMENT_COUNTS,
            "temperature_variations": DEFAULT_TEMPERATURE_VARIATIONS,
            "selected_models": [],
            "judge_models": [],
            "user_defined_metrics": "",
            "status": "configuring",
            "progress": 0,
            "created_at": None,
            "updated_at": None,
            "results": None
        }
    
    # Ensure output directory exists
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)


def create_new_evaluation():
    """Create a new evaluation configuration with default values."""
    return {
        "id": str(uuid.uuid4()),
        "name": f"Benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "csv_data": None,
        "prompt_column": None,
        "golden_answer_column": None,
        "task_type": "",
        "task_criteria": "",
        "output_dir": DEFAULT_OUTPUT_DIR,
        "parallel_calls": DEFAULT_PARALLEL_CALLS,
        "invocations_per_scenario": DEFAULT_INVOCATIONS_PER_SCENARIO,
        "sleep_between_invocations": DEFAULT_SLEEP_BETWEEN_INVOCATIONS,
        "experiment_counts": DEFAULT_EXPERIMENT_COUNTS,
        "temperature_variations": DEFAULT_TEMPERATURE_VARIATIONS,
        "selected_models": [],
        "judge_models": [],
        "user_defined_metrics": "",
        "status": "configuring",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "results": None
    }


def save_current_evaluation():
    """Save the current evaluation configuration to the list of evaluations."""
    # Debug current state
    print(f"Current session state before saving: {len(st.session_state.evaluations)} evaluations")
    
    if st.session_state.current_evaluation_config["id"] is None:
        # This is a new evaluation
        new_eval = st.session_state.current_evaluation_config.copy()
        new_eval["id"] = str(uuid.uuid4())
        new_eval["created_at"] = datetime.now().isoformat()
        new_eval["updated_at"] = datetime.now().isoformat()
        new_eval["status"] = "configuring"  # Ensure status is set
        
        # Add to evaluations list
        st.session_state.evaluations.append(new_eval)
        print(f"Added new evaluation with ID: {new_eval['id']}, Name: {new_eval['name']}")
        print(f"Session state after adding: {len(st.session_state.evaluations)} evaluations")
        
        # Reset current config for next evaluation
        reset_current_evaluation()
    else:
        # This is an update to an existing evaluation
        eval_id = st.session_state.current_evaluation_config["id"]
        updated = False
        
        # Update existing evaluation
        for i, eval_config in enumerate(st.session_state.evaluations):
            if eval_config["id"] == eval_id:
                st.session_state.current_evaluation_config["updated_at"] = datetime.now().isoformat()
                st.session_state.evaluations[i] = st.session_state.current_evaluation_config.copy()
                print(f"Updated evaluation with ID: {eval_id}, Name: {st.session_state.evaluations[i]['name']}")
                updated = True
                break
                
        if not updated:
            # If not found (unusual case), add as new
            print(f"Evaluation with ID {eval_id} not found in list, adding as new")
            new_eval = st.session_state.current_evaluation_config.copy() 
            new_eval["updated_at"] = datetime.now().isoformat()
            st.session_state.evaluations.append(new_eval)
            
        # Reset current config for next evaluation
        reset_current_evaluation()


def reset_current_evaluation():
    """Reset the current evaluation configuration to default values."""
    st.session_state.current_evaluation_config = create_new_evaluation()


def load_evaluation(eval_id):
    """Load an evaluation configuration by ID."""
    for eval_config in st.session_state.evaluations:
        if eval_config["id"] == eval_id:
            st.session_state.current_evaluation_config = eval_config.copy()
            return


def update_evaluation_status(eval_id, status, progress=None):
    """Update the status of an evaluation."""
    # Make sure session state is initialized
    if "evaluations" not in st.session_state:
        initialize_session_state()
        
    try:
        for i, eval_config in enumerate(st.session_state.evaluations):
            if eval_config["id"] == eval_id:
                st.session_state.evaluations[i]["status"] = status
                st.session_state.evaluations[i]["updated_at"] = datetime.now().isoformat()
                if progress is not None:
                    st.session_state.evaluations[i]["progress"] = progress
                
                # Also update active and completed lists
                if status == "running":
                    if eval_id not in [e["id"] for e in st.session_state.active_evaluations]:
                        st.session_state.active_evaluations.append(eval_config.copy())
                elif status == "completed":
                    # Remove from active list
                    st.session_state.active_evaluations = [e for e in st.session_state.active_evaluations if e["id"] != eval_id]
                    # Add to completed list if not already there
                    if eval_id not in [e["id"] for e in st.session_state.completed_evaluations]:
                        st.session_state.completed_evaluations.append(eval_config.copy())
                
                return
    except Exception as e:
        # Handle cases where session state might not be accessible (like in a thread)
        print(f"Warning: Could not update session state: {str(e)}")
        # Just continue - the file-based status will be used instead