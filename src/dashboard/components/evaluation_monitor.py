"""Evaluation monitor component for the Streamlit dashboard."""

import streamlit as st
import pandas as pd
from ..utils.benchmark_runner import run_benchmark_async

class EvaluationMonitorComponent:
    """Component for monitoring active evaluations."""
    
    def render(self):
        """Render the evaluation monitor component."""
        
        st.subheader("Active Evaluations")
        
        # Check if there are any active evaluations
        if not st.session_state.active_evaluations:
            st.info("No active evaluations. Go to Setup tab to create and run evaluations.")
        else:
            # Display active evaluations with progress bars
            for i, eval_config in enumerate(st.session_state.active_evaluations):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{eval_config['name']}**")
                        st.progress(eval_config["progress"] / 100)
                        
                        # Display details
                        st.write(f"Task: {eval_config['task_type']}")
                        st.write(f"Models: {len(eval_config['selected_models'])}")
                        st.write(f"Status: {eval_config['status'].capitalize()}")
                    
                    with col2:
                        # Allow cancellation (not implemented in this version)
                        st.button(
                            "Cancel",
                            key=f"cancel_{i}",
                            help="Cancel this evaluation (not implemented in this version)"
                        )
                
                st.divider()
        
        st.subheader("Available Evaluations")
        
        # Get all evaluations that are not active
        available_evals = [
            e for e in st.session_state.evaluations 
            if e["id"] not in [a["id"] for a in st.session_state.active_evaluations]
            and e["status"] != "completed"
        ]
        
        if not available_evals:
            st.info("No available evaluations. Go to Setup tab to create new evaluations.")
        else:
            # Create a table of available evaluations
            eval_data = []
            for eval_config in available_evals:
                eval_data.append({
                    "ID": eval_config["id"],
                    "Name": eval_config["name"],
                    "Task Type": eval_config["task_type"],
                    "Models": len(eval_config["selected_models"]),
                    "Status": eval_config["status"].capitalize(),
                    "Created": pd.to_datetime(eval_config["created_at"]).strftime("%Y-%m-%d %H:%M")
                })
            
            eval_df = pd.DataFrame(eval_data)
            st.dataframe(eval_df)
            
            # Allow running selected evaluations
            st.subheader("Run Selected Evaluations")
            
            # Multiselect for evaluation IDs
            selected_eval_ids = st.multiselect(
                "Select evaluations to run",
                options=[e["id"] for e in available_evals],
                format_func=lambda x: next((e["name"] for e in available_evals if e["id"] == x), x)
            )
            
            if selected_eval_ids:
                st.button(
                    "Run Selected Evaluations",
                    on_click=self._run_selected_evaluations,
                    args=(selected_eval_ids,)
                )
    
    def _run_selected_evaluations(self, eval_ids):
        """Run the selected evaluations."""
        for eval_id in eval_ids:
            for eval_config in st.session_state.evaluations:
                if eval_config["id"] == eval_id:
                    run_benchmark_async(eval_config)
                    st.success(f"Evaluation '{eval_config['name']}' started.")
                    break