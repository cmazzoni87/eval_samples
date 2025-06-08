import streamlit as st
import os
import sys
import time
from pathlib import Path

# Add the project root to path to allow importing dashboard modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dashboard components
from src.dashboard.components.evaluation_setup import EvaluationSetupComponent
from src.dashboard.components.model_configuration import ModelConfigurationComponent
from src.dashboard.components.evaluation_monitor import EvaluationMonitorComponent
from src.dashboard.components.results_viewer import ResultsViewerComponent
from src.dashboard.utils.state_management import initialize_session_state
from src.dashboard.utils.constants import APP_TITLE, SIDEBAR_INFO

# Initialize session state at module level to ensure it's available before component rendering
if "evaluations" not in st.session_state:
    initialize_session_state()

def main():
    """Main Streamlit dashboard application."""
    # Set page title and layout with custom icon
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="./assets/aws.svg",
        layout="wide"
    )
    
    # Initialize session state again to ensure all variables are set
    initialize_session_state()
    
    # Header with logo and title
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image("./assets/aws.svg", width=150)
    
    with col2:
        st.title(APP_TITLE)
        st.markdown("Create, manage, and visualize LLM benchmark evaluations using LLM-as-a-JURY methodology")
    
    # Sidebar with info
    with st.sidebar:
        st.markdown(SIDEBAR_INFO)
        st.divider()
        
        # Navigation tabs in sidebar
        tab_names = ["Setup", "Monitor", "Results"]
        active_tab = st.radio("Navigation", tab_names)
    
    # Main area - show different components based on active tab
    if active_tab == "Setup":
        # Use tabs instead of nested expanders
        setup_tab1, setup_tab2 = st.tabs(["Evaluation Setup", "Model Configuration"])
        
        with setup_tab1:
            EvaluationSetupComponent().render()
        
        with setup_tab2:
            ModelConfigurationComponent().render()
            
    elif active_tab == "Monitor":
        EvaluationMonitorComponent().render()
        
    elif active_tab == "Results":
        ResultsViewerComponent().render()

if __name__ == "__main__":
    main()