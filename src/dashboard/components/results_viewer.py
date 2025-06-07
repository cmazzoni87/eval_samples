"""Results viewer component for the Streamlit dashboard."""

import streamlit as st
import pandas as pd
import os
from pathlib import Path

class ResultsViewerComponent:
    """Component for viewing evaluation results."""
    
    def render(self):
        """Render the results viewer component."""
        
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
            
            # Select an evaluation to view results
            selected_eval_id = st.selectbox(
                "Select evaluation to view results",
                options=[e["id"] for e in completed_evals],
                format_func=lambda x: next((e["name"] for e in completed_evals if e["id"] == x), x)
            )
            
            if selected_eval_id:
                self._show_evaluation_results(selected_eval_id)
        
        st.subheader("Generated Reports")
        
        # Find all HTML reports in the benchmark results directory
        benchmark_dir = Path(st.session_state.current_evaluation_config["output_dir"])
        html_reports = []
        
        if benchmark_dir.exists():
            for report_file in benchmark_dir.glob("llm_benchmark_report_*.html"):
                html_reports.append({
                    "filename": report_file.name,
                    "path": str(report_file),
                    "created": pd.Timestamp(os.path.getmtime(report_file), unit='s').strftime("%Y-%m-%d %H:%M")
                })
        
        if not html_reports:
            st.info("No HTML reports found. Run evaluations to generate reports.")
        else:
            # Create a table of reports
            reports_df = pd.DataFrame(html_reports)
            st.dataframe(reports_df)
            
            # Select a report to view
            selected_report = st.selectbox(
                "Select report to view",
                options=[r["path"] for r in html_reports],
                format_func=lambda x: next((r["filename"] for r in html_reports if r["path"] == x), x)
            )
            
            if selected_report:
                self._show_report(selected_report)
    
    def _show_evaluation_results(self, eval_id):
        """Show detailed results for a specific evaluation."""
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
        if eval_config.get("results"):
            st.write("#### Evaluation Report")
            st.write("The following HTML report was generated:")
            st.write(f"[{os.path.basename(eval_config['results'])}]({eval_config['results']})")
            
            # Provide option to view report
            st.button(
                "View Report",
                on_click=self._show_report,
                args=(eval_config["results"],)
            )
    
    def _show_report(self, report_path):
        """Display an HTML report."""
        # Check if report exists
        if not os.path.exists(report_path):
            st.error(f"Report file not found: {report_path}")
            return
        
        # Read HTML content
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Display HTML
        st.components.v1.html(html_content, height=600, scrolling=True)