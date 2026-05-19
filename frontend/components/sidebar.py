import base64
import streamlit as st
from pathlib import Path
from utils.api import clear_chat
from frontend_config.settings import examples

def _icon_base64() -> str:
    icon_path = Path(__file__).parent / "GRFA_icon.png"
    return base64.b64encode(icon_path.read_bytes()).decode()

def display_sidebar():
    """Display the application sidebar"""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px; padding: 4px 0 8px 0;">
                <img src="data:image/png;base64,{_icon_base64()}"
                     style="width:40px; height:40px; object-fit:contain; flex-shrink:0;">
                <span style="font-size:1.4rem; font-weight:700; line-height:1.2;">
                    GraphRAG Finance Assistant
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.use_stream = True
        st.session_state.show_thinking = False

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.button("🗑️ Clear Chat", key="clear_chat_system", on_click=clear_chat)

        st.markdown("---")

        # Example questions section
        st.header("Example Questions")
        example_questions = examples

        for question in example_questions:
            st.markdown(f"""
            <div style="background-color: #f7f7f7; padding: 8px;
                 border-radius: 4px; margin: 5px 0; font-size: 14px; cursor: pointer;">
                {question}
            </div>
            """, unsafe_allow_html=True)



