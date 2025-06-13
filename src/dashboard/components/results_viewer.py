"""Results viewer component for the Streamlit dashboard."""

import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
from ..utils.benchmark_runner import sync_evaluations_from_files

class ResultsViewerComponent:
    """Component for viewing evaluation results."""
    
    def render(self):
        """Render the results viewer component."""
        # Sync evaluation statuses from files
        sync_evaluations_from_files()
        
        st.subheader("Completed Evaluations")
        
        # Check if there are any completed evaluations
        completed_evals = [
            e for e in st.session_state.evaluations 
            if e["status"] == "completed"
        ]
        
        if not completed_evals:
            st.info("No completed evaluations yet. Run evaluations to see results here.")
        else:
            # Create a table of completed evaluations
            eval_data = []
            for eval_config in completed_evals:
                eval_data.append({
                    "ID": eval_config["id"],
                    "Name": eval_config["name"],
                    "Task Type": eval_config["task_type"],
                    "Models": len(eval_config["selected_models"]),
                    "Completed": pd.to_datetime(eval_config["updated_at"]).strftime("%Y-%m-%d %H:%M")
                })
            
            eval_df = pd.DataFrame(eval_data)
            st.dataframe(eval_df)
            
            # Add refresh button
            st.button(
                "Refresh Results",
                on_click=sync_evaluations_from_files
            )
            
            # Select an evaluation to view results
            selected_eval_id = st.selectbox(
                "Select evaluation to view results",
                options=[e["id"] for e in completed_evals],
                format_func=lambda x: next((e["name"] for e in completed_evals if e["id"] == x), x)
            )
            
            if selected_eval_id:
                self._show_evaluation_results(selected_eval_id)
        
        # The Generated Reports section has been removed as reports are now linked in a different section
        # after evaluation is executed.
    
    def _show_evaluation_results(self, eval_id):
        """Show detailed results for a specific evaluation."""
        # First try to find status file for the most up-to-date information
        output_dir = Path(st.session_state.current_evaluation_config["output_dir"])
        status_file = output_dir / f"eval_{eval_id}_status.json"
        
        # Check if status file exists and has results
        results_path = None
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                    if "results" in status_data:
                        results_path = status_data["results"]
            except:
                pass
        
        # Find the evaluation configuration
        eval_config = None
        for e in st.session_state.evaluations:
            if e["id"] == eval_id:
                eval_config = e
                break
        
        if not eval_config:
            st.error("Evaluation not found")
            return
        
        # Display evaluation details
        st.write(f"### {eval_config['name']}")
        st.write(f"**Task Type:** {eval_config['task_type']}")
        st.write(f"**Task Criteria:** {eval_config['task_criteria']}")
        
        # Display models used
        st.write("#### Models Evaluated")
        models_df = pd.DataFrame(eval_config["selected_models"])
        st.dataframe(models_df)
        
        # Display judges used
        st.write("#### Judge Models")
        judges_df = pd.DataFrame(eval_config["judge_models"])
        st.dataframe(judges_df)
        
        # Display report if available
        # First check results from status file, then from session state
        if results_path or eval_config.get("results"):
            report_path = results_path or eval_config.get("results")
            st.write("#### Evaluation Report")
            st.write("The following HTML report was generated:")
            st.write(f"[{os.path.basename(report_path)}]({report_path})")
            
            # Provide option to view report
            st.button(
                "View Report",
                on_click=self._show_report,
                args=(report_path,)
            )
    
    def _show_report(self, report_path):
        """Provide a link to the HTML report."""
        # Check if report exists
        if not os.path.exists(report_path):
            st.error(f"Report file not found: {report_path}")
            return
        
        # Create a file URL for the report
        report_filename = os.path.basename(report_path)
        file_url = f"{os.path.abspath(report_path)}"
        
        # Display link to open the report in browser
        st.markdown(f"### Report: {report_filename}")
        st.markdown(f"Click below to open the report in your browser:")
        st.markdown(f"[📊 Open Full Report in Browser]({file_url})", unsafe_allow_html=True)
        
        # Also show the file path for users who want to navigate to it directly
        st.info(f"Report location: {os.path.abspath(report_path)}")