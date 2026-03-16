import streamlit as st

from utils.state import init_session_state
from components.styles import custom_css
from components.chat import display_chat_interface
from components.sidebar import display_sidebar
from utils.performance import init_performance_monitoring

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="GraphRAG Finance Assistant",
        page_icon="components/GRFA_icon.png",
        layout="wide"
    )

    # Initialize session state
    init_session_state()

    # Initialize performance monitoring
    init_performance_monitoring()

    # Add custom CSS
    custom_css()

    # Display sidebar
    display_sidebar()

    # Display chat interface
    display_chat_interface()

if __name__ == "__main__":
    import shutup
    shutup.please()
    main()
