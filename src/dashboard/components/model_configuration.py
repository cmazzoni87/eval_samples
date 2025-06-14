"""Model configuration component for the Streamlit dashboard."""

import streamlit as st
import pandas as pd
from ..utils.constants import (
    DEFAULT_BEDROCK_MODELS, 
    DEFAULT_OPENAI_MODELS,
    DEFAULT_COST_MAP,
    DEFAULT_JUDGES_COST,
    DEFAULT_JUDGES,
    AWS_REGIONS
)
from ..utils.state_management import save_current_evaluation
from ..utils.benchmark_runner import run_benchmark_async


class ModelConfigurationComponent:
    """Component for configuring models and judge models."""
    
    def render(self):
        """Render the model configuration component."""
        
        # Region selection
        selected_region = st.selectbox(
            "AWS Region",
            options=AWS_REGIONS,
            index=1,  # Default to us-east-2
            key="aws_region"
        )
        
        # Available models tabs (Bedrock, OpenAI)
        tab1, tab2 = st.tabs(["Bedrock Models", "3P Models"])
        
        with tab1:
            bedrock_models = [model[0] for model in DEFAULT_BEDROCK_MODELS]
            self._render_model_dropdown(bedrock_models, "bedrock", selected_region)
        
        with tab2:
            openai_models = [model[0] for model in DEFAULT_OPENAI_MODELS]
            self._render_model_dropdown(openai_models, "openai", selected_region)
        
        # Selected models display
        st.subheader("Selected Models")
        if not st.session_state.current_evaluation_config["selected_models"]:
            st.info("No models selected. Please select at least one model to evaluate.")
        else:
            selected_models_df = pd.DataFrame(st.session_state.current_evaluation_config["selected_models"])
            selected_models_df = selected_models_df.rename(columns={
                "id": "Model ID",
                "region": "AWS Region",
                "input_cost": "Input Cost (per token)",
                "output_cost": "Output Cost (per token)"
            })
            st.dataframe(selected_models_df)
            
            # Button to remove all selected models
            st.button(
                "Clear Selected Models",
                on_click=self._clear_selected_models
            )
        
        # Judge model selection
        st.subheader("Judge Models")
        self._render_judge_selection(selected_region)
        
        # If we have selected judge models, display them
        if st.session_state.current_evaluation_config["judge_models"]:
            judge_models_df = pd.DataFrame(st.session_state.current_evaluation_config["judge_models"])
            judge_models_df = judge_models_df.rename(columns={
                "id": "Model ID",
                "region": "AWS Region",
                "input_cost": "Input Cost (per token)",
                "output_cost": "Output Cost (per token)"
            })
            st.dataframe(judge_models_df)
            
            # Button to remove all judge models
            st.button(
                "Clear Judge Models",
                on_click=self._clear_judge_models,
                key="clear_judges"
            )
        
        # Show validation status
        is_valid = self._is_configuration_valid()
        missing_items = self._get_missing_configuration_items()
        
        if not is_valid and missing_items:
            st.warning(f"Please complete the following before saving: {', '.join(missing_items)}")
        
        # Action buttons - only save and reset, no direct run
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(
                "Save Configuration",
                disabled=not is_valid,
            ):
                save_current_evaluation()
                st.success(f"Configuration '{st.session_state.current_evaluation_config['name']}' saved successfully!")
                # Debug information
                print(f"Saved configuration to session state. Total evaluations: {len(st.session_state.evaluations)}")
                print(f"Evaluation IDs: {[e['id'] for e in st.session_state.evaluations]}")
        
        with col2:
            st.button(
                "Reset Configuration",
                on_click=self._reset_configuration
            )
    
    def _render_model_dropdown(self, model_list, prefix, region):
        """Render the model selection UI with dropdown."""
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            selected_model = st.selectbox(
                "Select Model",
                options=model_list,
                key=f"{prefix}_model_select"
            )
        
        # Get default costs
        default_input_cost = DEFAULT_COST_MAP.get(selected_model, {"input": 0.001, "output": 0.002})["input"]
        default_output_cost = DEFAULT_COST_MAP.get(selected_model, {"input": 0.001, "output": 0.002})["output"]
        
        with col2:
            input_cost = st.number_input(
                "Input Cost",
                min_value=0.0,
                max_value=1.0,
                value=default_input_cost,
                step=0.0001,
                format="%.6f",
                key=f"{prefix}_input_cost"
            )
        
        with col3:
            output_cost = st.number_input(
                "Output Cost",
                min_value=0.0,
                max_value=1.0,
                value=default_output_cost,
                step=0.0001,
                format="%.6f",
                key=f"{prefix}_output_cost"
            )
        
        with col4:
            st.button(
                "Add Model",
                key=f"{prefix}_add_model",
                on_click=self._add_model,
                args=(selected_model, region, input_cost, output_cost)
            )
    
    def _render_judge_selection(self, region):
        """Render the judge model selection UI."""
        # Use Claude models as default judges
        judge_options = [m[0] for m in DEFAULT_JUDGES]
        judge_regions = {m[0]: m[1] for m in DEFAULT_JUDGES}
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            selected_judge = st.selectbox(
                "Select Judge Model",
                options=judge_options,
                key="judge_model_select"
            )
        
        # Get default costs
        default_input_cost = DEFAULT_JUDGES_COST.get(selected_judge, {"input": 0.001, "output": 0.002})["input"]
        default_output_cost = DEFAULT_JUDGES_COST.get(selected_judge, {"input": 0.001, "output": 0.002})["output"]
        region = judge_regions[selected_judge]
        with col2:
            judge_input_cost = st.number_input(
                "Input Cost",
                min_value=0.0,
                max_value=1.0,
                value=default_input_cost,
                step=0.0001,
                format="%.6f",
                key="judge_input_cost"
            )
        
        with col3:
            judge_output_cost = st.number_input(
                "Output Cost",
                min_value=0.0,
                max_value=1.0,
                value=default_output_cost,
                step=0.0001,
                format="%.6f",
                key="judge_output_cost"
            )
        
        with col4:
            st.button(
                "Add Judge",
                key="add_judge",
                on_click=self._add_judge_model,
                args=(selected_judge, region, judge_input_cost, judge_output_cost)
            )
    
    def _add_model(self, model_id, region, input_cost, output_cost):
        """Add a model to the selected models list."""
        # Check if model is already selected with same region
        for model in st.session_state.current_evaluation_config["selected_models"]:
            # Check if the model ID matches and either region matches or isn't present
            if model["id"] == model_id and model.get("region", "") == region:
                # Update costs and region if model already exists
                model["input_cost"] = input_cost
                model["output_cost"] = output_cost
                model["region"] = region
                return
        
        # Add new model
        st.session_state.current_evaluation_config["selected_models"].append({
            "id": model_id,
            "region": region,
            "input_cost": input_cost,
            "output_cost": output_cost
        })
    
    def _add_judge_model(self, model_id, region, input_cost, output_cost):
        """Add a judge model to the judge models list."""
        # Check if model is already selected with same region
        for model in st.session_state.current_evaluation_config["judge_models"]:
            # Check if the model ID matches and either region matches or isn't present
            if model["id"] == model_id and model.get("region", "") == region:
                # Update costs and region if model already exists
                model["input_cost"] = input_cost
                model["output_cost"] = output_cost
                model["region"] = region
                return
        
        # Add new model
        st.session_state.current_evaluation_config["judge_models"].append({
            "id": model_id,
            "region": region,
            "input_cost": input_cost,
            "output_cost": output_cost
        })
    
    def _clear_selected_models(self):
        """Clear all selected models."""
        st.session_state.current_evaluation_config["selected_models"] = []
    
    def _clear_judge_models(self):
        """Clear all judge models."""
        st.session_state.current_evaluation_config["judge_models"] = []
    
    def _reset_configuration(self):
        """Reset the current configuration to default values."""
        # Keep CSV data and column selections, reset everything else
        csv_data = st.session_state.current_evaluation_config["csv_data"]
        prompt_column = st.session_state.current_evaluation_config["prompt_column"]
        golden_answer_column = st.session_state.current_evaluation_config["golden_answer_column"]
        
        st.session_state.current_evaluation_config = {
            "id": None,
            "name": f"Benchmark-{pd.Timestamp.now().strftime('%Y%m%d')}",
            "csv_data": csv_data,
            "prompt_column": prompt_column,
            "golden_answer_column": golden_answer_column,
            "task_type": "",
            "task_criteria": "",
            "output_dir": st.session_state.current_evaluation_config["output_dir"],
            "parallel_calls": st.session_state.current_evaluation_config["parallel_calls"],
            "invocations_per_scenario": st.session_state.current_evaluation_config["invocations_per_scenario"],
            "sleep_between_invocations": st.session_state.current_evaluation_config["sleep_between_invocations"],
            "experiment_counts": st.session_state.current_evaluation_config["experiment_counts"],
            "temperature_variations": st.session_state.current_evaluation_config["temperature_variations"],
            "selected_models": [],
            "judge_models": [],
            "user_defined_metrics": "",
            "status": "configuring",
            "progress": 0,
            "created_at": None,
            "updated_at": None,
            "results": None
        }
    
    def _get_missing_configuration_items(self):
        """Get a list of missing configuration items."""
        config = st.session_state.current_evaluation_config
        missing_items = []
        
        # Check for CSV data with prompt and golden answer columns
        if config["csv_data"] is None:
            missing_items.append("CSV data")
        elif not config["prompt_column"] or not config["golden_answer_column"]:
            missing_items.append("prompt and golden answer column selection")
        
        # Check for task type and criteria
        if not config["task_type"]:
            missing_items.append("task type")
        if not config["task_criteria"]:
            missing_items.append("task criteria")
        
        # Check for at least one target model
        if not config["selected_models"]:
            missing_items.append("at least one target model")
        
        # Check for at least one judge model
        if not config["judge_models"]:
            missing_items.append("at least one judge model")
        
        return missing_items
    
    def _is_configuration_valid(self):
        """Check if the current configuration is valid."""
        return len(self._get_missing_configuration_items()) == 0
    
    def _run_evaluation(self):
        """Save the configuration and run the evaluation."""
        # First save the configuration
        save_current_evaluation()
        
        # Get the saved evaluation ID
        eval_id = None
        for eval_config in st.session_state.evaluations:
            if eval_config["name"] == st.session_state.current_evaluation_config["name"]:
                eval_id = eval_config["id"]
                break
        
        if eval_id:
            # Run the evaluation asynchronously
            for eval_config in st.session_state.evaluations:
                if eval_config["id"] == eval_id:
                    run_benchmark_async(eval_config)
                    st.success(f"Evaluation '{eval_config['name']}' started. Go to Monitor tab to view progress.")
                    break