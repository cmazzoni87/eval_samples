"""Evaluation monitor component for the Streamlit dashboard."""

import streamlit as st
import pandas as pd
import time
import os
import json
import logging
import base64
from datetime import datetime
from pathlib import Path
from ..utils.benchmark_runner import run_benchmark_async, sync_evaluations_from_files, dashboard_logger

class EvaluationMonitorComponent:
    """Component for monitoring active evaluations."""
    
    def render(self):
        """Render the evaluation monitor component."""
        # Create a placeholder for notifications at the very top of the page
        notification_placeholder = st.empty()
        
        dashboard_logger.info("Rendering evaluation monitor component")
        
        # Debug information about current session state
        print(f"Current evaluations in session state: {len(st.session_state.evaluations)}")
        for i, eval_config in enumerate(st.session_state.evaluations):
            print(f"Evaluation {i+1}: ID={eval_config['id']}, Name={eval_config['name']}, Status={eval_config['status']}")
        
        # Sync evaluation statuses from files
        sync_evaluations_from_files()
        
        # Simple tracking of evaluation status without notifications
        if 'last_status_check' not in st.session_state:
            st.session_state.last_status_check = {}
            
        # Update tracked statuses but without creating notifications
        for eval_config in st.session_state.evaluations:
            eval_id = eval_config.get("id")
            current_status = eval_config.get("status")
            
            # Update tracked status
            if eval_id not in st.session_state.last_status_check:
                st.session_state.last_status_check[eval_id] = current_status
            elif st.session_state.last_status_check[eval_id] != current_status:
                # Status has changed - just log it
                dashboard_logger.info(f"Evaluation {eval_id} status changed: {st.session_state.last_status_check[eval_id]} -> {current_status}")
                st.session_state.last_status_check[eval_id] = current_status
                
        # No notifications or auto-refresh - just a manual refresh button
        st.button("Refresh Evaluations", on_click=sync_evaluations_from_files)
            
        # No auto-refresh - just display the last refresh time
        current_time = time.time()
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = current_time
            
        # Calculate time since last refresh
        time_since_refresh = current_time - st.session_state.last_refresh_time
        st.session_state.last_refresh_time = current_time
        
        # Show last refresh time
        st.caption(f"Last refreshed: {datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}")
            
        # Add a UI indicator for the log file location
        from ..utils.constants import PROJECT_ROOT
        log_dir = os.path.join(PROJECT_ROOT, 'logs')
        st.info(f"📋 Benchmark logs available at: {log_dir} (Only 360-benchmark logs are saved to disk)")
        
        # Get current session time
        current_session_start = st.session_state.get('session_start_time', time.time())
        if 'session_start_time' not in st.session_state:
            st.session_state.session_start_time = current_session_start
            dashboard_logger.info(f"Set session start time to {current_session_start}")
        
        # Retrieve all evaluations for this session
        dashboard_logger.debug("Retrieving session evaluations")
        session_evals = self._get_session_evaluations(current_session_start)
        
        # Separate active and recently completed evaluations
        active_evals = [e for e in session_evals if e.get('status') in ['in-progress', 'running']]
        completed_evals = [e for e in session_evals if e.get('status') == 'completed' and 
                          e.get('end_time', 0) > current_time - 60]  # Show completed in last minute
        failed_evals = [e for e in session_evals if e.get('status') == 'failed' and
                       e.get('end_time', 0) > current_time - 60]  # Show failed in last minute
        
        # Display active and recent evaluations
        st.subheader("Active & Recent Evaluations")
        all_display_evals = active_evals + completed_evals + failed_evals
        
        if not all_display_evals:
            st.info("No active evaluations in this session. Go to Setup tab to create and run evaluations.")
        else:
            dashboard_logger.info(f"Displaying {len(all_display_evals)} evaluations (Active: {len(active_evals)}, " +
                                  f"Recently Completed: {len(completed_evals)}, Failed: {len(failed_evals)})")
            
            # Display evaluations with status indicators
            for i, eval_config in enumerate(all_display_evals):
                # Highlight the evaluation if it matches the one clicked in a notification
                highlight_style = ""
                if "highlight_eval_id" in st.session_state and st.session_state.highlight_eval_id == eval_config["id"]:
                    highlight_style = "border: 2px solid #FFA500; background-color: rgba(255, 165, 0, 0.1); border-radius: 5px; padding: 10px;"
                    # Clear the highlight after showing it once
                    st.session_state.highlight_eval_id = None
                
                with st.container():
                    if highlight_style:
                        st.markdown(f'<div style="{highlight_style}">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        # Display status as colored indicator
                        status = eval_config.get('status', 'unknown')
                        
                        # Check if this evaluation has a report to link to
                        has_report = (status == "completed" and 
                                     'results' in eval_config and 
                                     eval_config['results'] and 
                                     os.path.exists(eval_config['results']))
                        
                        # Display name - as link if report exists
                        if has_report:
                            report_path = eval_config['results']
                            file_url = f"{os.path.abspath(report_path)}"
                            st.markdown(f"**[{eval_config['name']}]({file_url})**", unsafe_allow_html=True)
                        else:
                            st.write(f"**{eval_config['name']}**")
                        
                        # Display status indicator
                        if status in ['in-progress', 'running']:
                            st.markdown("🔄 **Status**: <span style='color:blue'>In Progress</span>", unsafe_allow_html=True)
                        elif status == "failed":
                            st.markdown("❌ **Status**: <span style='color:red'>Failed</span>", unsafe_allow_html=True)
                        elif status == "completed":
                            st.markdown("✅ **Status**: <span style='color:green'>Completed</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"⚠️ **Status**: {status.capitalize()}")
                    
                    with col2:
                        # Display details
                        st.write(f"Task: {eval_config['task_type']}")
                        st.write(f"Models: {len(eval_config['selected_models'])}")
                        
                        # Display elapsed time if available
                        if 'start_time' in eval_config:
                            end_time = eval_config.get('end_time', time.time())
                            elapsed = end_time - eval_config['start_time']
                            st.write(f"Elapsed: {self._format_time(elapsed)}")
                    
                    with col3:
                        # For completed evaluations, offer report options
                        if status == "completed":
                            # Check if a report already exists
                            if 'results' in eval_config and eval_config['results'] and os.path.exists(eval_config['results']):
                                # Log that we have a report
                                dashboard_logger.info(f"Evaluation has report: {eval_config['results']}")
                                st.markdown("📊 **Report Available**", unsafe_allow_html=True)
                            else:
                                # Offer to generate a report
                                if st.button(f"📊 Generate Report", key=f"gen_report_{i}"):
                                    self._generate_report(eval_config)
                                    dashboard_logger.info(f"Generating report for evaluation {eval_config['id']}")
                        
                        # Add view logs button
                        if 'logs_dir' in eval_config and os.path.exists(eval_config['logs_dir']):
                            if st.button("View Logs", key=f"logs_{i}"):
                                self._show_logs(eval_config)
                                dashboard_logger.info(f"Showing logs for evaluation {eval_config['id']}")
                        
                        # Debug button to view full evaluation details
                        if st.button("Debug Info", key=f"debug_{i}"):
                            dashboard_logger.info(f"Showing debug info for evaluation {eval_config['id']}")
                            with st.expander("Evaluation Details"):
                                st.json({k: str(v) if k == 'csv_data' else v for k, v in eval_config.items()})
                
                # Show error if present
                if 'error' in eval_config and eval_config['error']:
                    with st.expander("Show Error"):
                        st.error(eval_config['error'])
                        dashboard_logger.error(f"Evaluation {eval_config['id']} error: {eval_config['error']}")
                
                # Close the highlight div if it was opened
                if "highlight_eval_id" in st.session_state and st.session_state.highlight_eval_id == eval_config["id"]:
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.divider()
            
            # Add refresh button for active evaluations
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Refresh Now"):
                    sync_evaluations_from_files()
                    dashboard_logger.info("Manually refreshed evaluation statuses")
                    st.session_state.pending_rerun = True
            with col2:
                st.caption("")
        
        # Display Available Evaluations Section
        st.subheader("Available Evaluations")
        
        # Debug session state
        print(f"Checking for available evaluations in {len(st.session_state.evaluations)} total evaluations")
        
        # Get all evaluations regardless of status (we'll filter in the UI if needed)
        available_evals = list(st.session_state.evaluations)
        
        # Print available evaluations for debugging
        for i, e in enumerate(available_evals):
            print(f"Evaluation {i+1}: ID={e['id']}, Name={e['name']}, Status={e['status']}")
        
        if not available_evals:
            st.info("No available evaluations. Go to Setup tab to create new evaluations.")
        else:
            dashboard_logger.info(f"Found {len(available_evals)} available evaluations")
            # Create a table of available evaluations
            eval_data = []
            
            # First create a list of reports that are available
            report_links = {}
            completed_evals_without_reports = []
            for eval_config in available_evals:
                # Check if this evaluation has a report
                if (eval_config.get("status") == "completed" and 
                    'results' in eval_config and 
                    eval_config['results'] and 
                    os.path.exists(eval_config['results'])):
                    report_links[eval_config["id"]] = f"file://{os.path.abspath(eval_config['results'])}"
                elif eval_config.get("status") == "completed":
                    # Keep track of completed evaluations without reports
                    completed_evals_without_reports.append(eval_config["id"])
            
            # Then create the table data
            for eval_config in available_evals:
                # Prepare the name field - as a link if report exists
                if eval_config["id"] in report_links:
                    name_field = f"<a href='{report_links[eval_config['id']]}'>{eval_config['name']}</a>"
                else:
                    name_field = eval_config["name"]
                
                # Determine what to display in the Report column
                if eval_config["id"] in report_links:
                    report_field = "📊 Available"
                elif eval_config["id"] in completed_evals_without_reports:
                    report_field = "🔄 Generate"
                else:
                    report_field = ""
                    
                eval_data.append({
                    "ID": eval_config["id"],
                    "Name": name_field,
                    "Task Type": eval_config["task_type"],
                    "Models": len(eval_config["selected_models"]),
                    "Status": eval_config["status"].capitalize(),
                    "Created": pd.to_datetime(eval_config["created_at"]).strftime("%Y-%m-%d %H:%M") if eval_config.get("created_at") else "N/A",
                    "Report": report_field
                })
            
            # Display the table
            eval_df = pd.DataFrame(eval_data)
            
            # Show the dataframe with clickable elements
            clicked = st.dataframe(
                eval_df, 
                use_container_width=True, 
                column_config={
                    "Name": st.column_config.Column("Name", width="medium", help="Click name to open report if available"),
                    "Report": st.column_config.Column("Report", width="small", help="Click to generate report for completed evaluations")
                },
                hide_index=True
            )
            
            # Add ability to generate reports for completed evaluations without reports
            if completed_evals_without_reports:
                st.caption("Click '🔄 Generate' in the Report column to generate reports for completed evaluations")
                
                selected_eval_id = st.selectbox(
                    "Select evaluation to generate report for",
                    options=completed_evals_without_reports,
                    format_func=lambda x: next((e["name"] for e in available_evals if e["id"] == x), x)
                )
                
                if st.button("Generate Report", key="gen_report_available"):
                    # Find the selected evaluation config
                    selected_eval = next((e for e in available_evals if e["id"] == selected_eval_id), None)
                    if selected_eval:
                        self._generate_report(selected_eval)
                    else:
                        st.error("Selected evaluation not found")
            
            # Add section to run selected evaluations
            st.subheader("Run Selected Evaluations")
            
            # Multiselect for evaluation IDs
            selected_eval_ids = st.multiselect(
                "Select evaluations to run",
                options=[e["id"] for e in available_evals],
                format_func=lambda x: next((e["name"] for e in available_evals if e["id"] == x), x)
            )
            
            if selected_eval_ids:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Run Selected Evaluations", key="run_selected_btn"):
                        self._run_selected_evaluations(selected_eval_ids)
                with col2:
                    if st.button("Merge Selected Evaluations", key="merge_selected_btn"):
                        self._merge_selected_evaluations(selected_eval_ids)
            
        # No automatic reruns - removed auto-refresh logic
    
    def _get_session_evaluations(self, session_start_time):
        """Get all evaluations for the current session, including completed ones."""
        session_evals = []
        
        # Get from session state
        if hasattr(st.session_state, 'evaluations'):
            for eval_config in st.session_state.evaluations:
                # Check if this evaluation was started in this session
                status_file = Path(eval_config.get("output_dir", "benchmark_results")) / f"eval_{eval_config['id']}_status.json"
                if status_file.exists():
                    try:
                        with open(status_file, 'r') as f:
                            status_data = json.load(f)
                            # Include if started in this session
                            if status_data.get('start_time', 0) >= session_start_time:
                                # Merge status data with eval config
                                eval_data = eval_config.copy()
                                eval_data.update(status_data)
                                session_evals.append(eval_data)
                    except:
                        pass
                        
        return session_evals
        
        # This code is no longer used - the available evaluations section was rewritten
        # and moved earlier in the render method
        pass
        
        # This code is no longer used - the available evaluations section was rewritten
        # and moved earlier in the render method
        pass
    
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
    
    def _format_time(self, seconds):
        """Format seconds into a readable time string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def _show_logs(self, eval_config):
        """Show logs for an evaluation."""
        logs_dir = eval_config.get('logs_dir')
        if not logs_dir or not os.path.exists(logs_dir):
            st.error("Logs directory not found.")
            return
        
        # Search for benchmark log file
        from ..utils.constants import PROJECT_ROOT
        log_dir = os.path.join(PROJECT_ROOT, 'logs')
        benchmark_logs = list(Path(log_dir).glob(f"360-benchmark-*-{eval_config['name']}.log"))
        
        if benchmark_logs:
            benchmark_log = benchmark_logs[0]
            with st.expander("Benchmark Log", expanded=True):
                with open(benchmark_log, 'r') as f:
                    log_content = f.read()
                st.code(log_content)
        else:
            st.warning("No benchmark log file found for this evaluation. Only 360-benchmark logs are saved to disk.")
            
        # Access stdout/stderr captures if available 
        try:
            # This is a simplified approach - ideally we'd make these available through an API
            from ..utils.benchmark_runner import stdout_capture, stderr_capture
            
            if hasattr(stdout_capture, 'getvalue') and stdout_capture.getvalue():
                with st.expander("Standard Output (In-Memory)", expanded=True):
                    st.code(stdout_capture.getvalue())
            
            if hasattr(stderr_capture, 'getvalue') and stderr_capture.getvalue():
                with st.expander("Error Output (In-Memory)"):
                    st.code(stderr_capture.getvalue())
        except (ImportError, NameError, AttributeError):
            st.info("In-memory logs not available - only benchmark logs are saved to disk.")
    
    def _generate_report(self, eval_config):
        """Generate a report for a completed evaluation."""
        try:
            # Import the visualize_results module
            from ...visualize_results import create_html_report
            from ..utils.constants import PROJECT_ROOT
            
            # Get the output directory from the evaluation config
            output_dir = eval_config.get("output_dir", "benchmark_results")
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(PROJECT_ROOT, output_dir)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create status indicator
            with st.spinner("Generating report... This may take a moment."):
                # Call the report generator
                report_path = create_html_report(output_dir, timestamp)
                
                # Update the evaluation config with the report path
                for i, e in enumerate(st.session_state.evaluations):
                    if e["id"] == eval_config["id"]:
                        st.session_state.evaluations[i]["results"] = str(report_path)
                        # Update status file
                        status_file = Path(output_dir) / f"eval_{eval_config['id']}_status.json"
                        if status_file.exists():
                            try:
                                with open(status_file, 'r') as f:
                                    status_data = json.load(f)
                                status_data["results"] = str(report_path)
                                with open(status_file, 'w') as f:
                                    json.dump(status_data, f)
                            except Exception as e:
                                dashboard_logger.error(f"Error updating status file: {str(e)}")
                        break
                
                # Show success message
                st.success(f"Report generated: {os.path.basename(str(report_path))}")
                
                # Display link to open the report
                file_url = f"{os.path.abspath(str(report_path))}"
                st.markdown(f"[📊 Open Report]({file_url})", unsafe_allow_html=True)
                
                # Set pending rerun to refresh UI
                st.session_state.pending_rerun = True
                
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            dashboard_logger.exception(f"Error generating report: {str(e)}")

    def _merge_selected_evaluations(self, eval_ids):
        """Merge and run the selected evaluations as a single evaluation."""
        from ..utils.benchmark_runner import run_merged_evaluations
        
        dashboard_logger.info(f"Merging selected evaluations: {eval_ids}")
        
        if len(eval_ids) < 2:
            st.warning("Please select at least two evaluations to merge.")
            return
            
        # Get the evaluation configs to merge
        evals_to_merge = []
        for eval_id in eval_ids:
            for eval_config in st.session_state.evaluations:
                if eval_config["id"] == eval_id:
                    evals_to_merge.append(eval_config)
                    break
        
        if len(evals_to_merge) < 2:
            st.error("Could not find all selected evaluations.")
            return
            
        # Run the merged evaluations
        with st.spinner("Merging evaluations... This may take a moment."):
            try:
                # Run merged evaluations
                run_merged_evaluations(eval_ids)
                st.success(f"Successfully merged and started {len(eval_ids)} evaluations")
                
                # Show log file location to user
                from ..utils.constants import PROJECT_ROOT
                log_dir = os.path.join(PROJECT_ROOT, 'logs')
                st.info(f"Check benchmark logs in: {log_dir} (Only 360-benchmark logs are saved to disk)")
                
                # Force refresh of UI state
                sync_evaluations_from_files()
                
            except Exception as e:
                st.error(f"Error merging evaluations: {str(e)}")
                dashboard_logger.exception(f"Error merging evaluations: {str(e)}")

    def _run_selected_evaluations(self, eval_ids):
        """Run the selected evaluations."""
        dashboard_logger.info(f"Running selected evaluations: {eval_ids}")
        
        # Track successful starts for UI feedback
        started_evals = []
        failed_evals = []
        
        # Process each selected evaluation
        for eval_id in eval_ids:
            for eval_config in st.session_state.evaluations:
                if eval_config["id"] == eval_id:
                    try:
                        # Make sure the evaluation configuration is valid
                        if not eval_config.get("selected_models") or not eval_config.get("judge_models"):
                            raise ValueError("Missing required configuration: models or judge models")
                            
                        # Run the benchmark
                        run_benchmark_async(eval_config)
                        started_evals.append(eval_config["name"])
                        dashboard_logger.info(f"Successfully started evaluation: {eval_config['name']} (ID: {eval_id})")
                    except Exception as e:
                        error_msg = f"Error starting evaluation '{eval_config['name']}': {str(e)}"
                        dashboard_logger.exception(error_msg)
                        failed_evals.append((eval_config["name"], str(e)))
                    break
        
        # Show success/failure messages
        if started_evals:
            st.success(f"Started evaluations: {', '.join(started_evals)}")
            
            # Show log file location to user
            from ..utils.constants import PROJECT_ROOT
            log_dir = os.path.join(PROJECT_ROOT, 'logs')
            st.info(f"Check benchmark logs in: {log_dir} (Only 360-benchmark logs are saved to disk)")
            
        if failed_evals:
            for name, error in failed_evals:
                st.error(f"Failed to start '{name}': {error}")
                
        # Force refresh of UI state
        if started_evals:
            sync_evaluations_from_files()