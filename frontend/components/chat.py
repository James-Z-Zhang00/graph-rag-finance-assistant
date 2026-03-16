import streamlit as st
import streamlit.components.v1 as components
import uuid
import re
import json
import traceback
from utils.api import send_message, send_feedback, clear_chat, send_message_stream

def reset_processing_lock():
    """Reset the processing lock state"""
    st.session_state.processing_lock = False

def display_chat_interface():
    """Display the main chat interface"""
    # Ensure lock variable exists with an initial value
    if "processing_lock" not in st.session_state:
        st.session_state.processing_lock = False

    # Show warning if a request is being processed
    if st.session_state.processing_lock:
        st.warning("Please wait for the current operation to complete...")
        # Allow force-resetting the lock
        if st.button("Force Reset Processing State", key="force_reset_lock"):
            st.session_state.processing_lock = False
            st.rerun()

    # Chat area — messages only; chat_input lives outside so it pins to page bottom
    chat_container = st.container()
    with chat_container:
        # Display existing messages
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                # Get the content to display
                content = msg["content"]

                # Handle deep_research_agent thinking sections
                if msg["role"] == "assistant":
                    # Determine whether to show thinking
                    show_thinking = (st.session_state.agent_type == "deep_research_agent" and
                                    st.session_state.get("show_thinking", False))

                    # Prefer raw_thinking field if present
                    if "raw_thinking" in msg and show_thinking:
                        thinking_process = msg["raw_thinking"]
                        answer_content = msg.get("processed_content", content)

                        # Format thinking as block-quote
                        thinking_lines = thinking_process.split('\n')
                        quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])

                        st.markdown(quoted_thinking)
                        st.markdown("\n\n")
                        st.markdown(answer_content)
                    # Check for <think> tags
                    elif "<think>" in content and "</think>" in content:
                        thinking_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)

                        if thinking_match:
                            thinking_process = thinking_match.group(1)
                            # Strip thinking section, keep the answer
                            answer_content = content.replace(f"<think>{thinking_process}</think>", "").strip()

                            if show_thinking:
                                thinking_lines = thinking_process.split('\n')
                                quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])
                                st.markdown(quoted_thinking)
                                st.markdown("\n\n")
                                st.markdown(answer_content)
                            else:
                                # Only show the answer
                                st.markdown(answer_content)
                        else:
                            # Extraction failed - remove tags and show content
                            cleaned_content = re.sub(r'<think>|</think>', '', content)
                            st.markdown(cleaned_content)
                    else:
                        # Plain answer with no thinking section
                        st.markdown(content)
                else:
                    # User messages display directly
                    st.markdown(content)

                # Add feedback buttons and source references for AI answers
                if msg["role"] == "assistant":
                    # Ensure a unique message ID exists
                    if "message_id" not in msg:
                        msg["message_id"] = str(uuid.uuid4())

                    # Find the corresponding user question
                    user_query = ""
                    if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                        user_query = st.session_state.messages[i-1]["content"]

                    feedback_key = f"{msg['message_id']}"
                    feedback_type_key = f"feedback_type_{feedback_key}"

                    # Container for feedback result display
                    feedback_container = st.empty()

                    if feedback_key not in st.session_state.feedback_given:
                        # Show feedback buttons
                        col1, col2, col3 = st.columns([0.1, 0.1, 0.8])

                        with col1:
                            thumbs_up_key = f"thumbs_up_{msg['message_id']}_{i}"
                            if st.button("👍", key=thumbs_up_key):
                                if "feedback_in_progress" not in st.session_state:
                                    st.session_state.feedback_in_progress = False

                                if st.session_state.feedback_in_progress:
                                    with feedback_container:
                                        st.warning("Please wait for the current operation to complete...")
                                else:
                                    st.session_state.feedback_in_progress = True
                                    try:
                                        with feedback_container:
                                            with st.spinner("Submitting feedback..."):
                                                response = send_feedback(
                                                    msg["message_id"],
                                                    user_query,
                                                    True,
                                                    st.session_state.session_id,
                                                    st.session_state.agent_type
                                                )

                                        st.session_state.feedback_given.add(feedback_key)
                                        st.session_state[feedback_type_key] = "positive"

                                        with feedback_container:
                                            if response and "action" in response:
                                                if "high quality" in response["action"]:
                                                    st.success("Thank you! This answer has been marked as high quality.", icon="🙂")
                                                else:
                                                    st.success("Thank you for your feedback!", icon="👍")
                                            else:
                                                st.info("Your feedback has been received.", icon="ℹ️")
                                    except Exception as e:
                                        st.error(f"Error submitting feedback: {str(e)}")
                                    finally:
                                        st.session_state.feedback_in_progress = False

                        with col2:
                            thumbs_down_key = f"thumbs_down_{msg['message_id']}_{i}"
                            if st.button("👎", key=thumbs_down_key):
                                if "feedback_in_progress" not in st.session_state:
                                    st.session_state.feedback_in_progress = False

                                if st.session_state.feedback_in_progress:
                                    with feedback_container:
                                        st.warning("Please wait for the current operation to complete...")
                                else:
                                    st.session_state.feedback_in_progress = True
                                    try:
                                        with feedback_container:
                                            with st.spinner("Submitting feedback..."):
                                                response = send_feedback(
                                                    msg["message_id"],
                                                    user_query,
                                                    False,
                                                    st.session_state.session_id,
                                                    st.session_state.agent_type
                                                )

                                        st.session_state.feedback_given.add(feedback_key)
                                        st.session_state[feedback_type_key] = "negative"

                                        with feedback_container:
                                            if response and "action" in response:
                                                if "cleared" in response["action"]:
                                                    st.error("Feedback received — this answer will not be reused.", icon="🔄")
                                                else:
                                                    st.error("Feedback received — we will work on improving.", icon="👎")
                                            else:
                                                st.info("Your feedback has been received.", icon="ℹ️")
                                    except Exception as e:
                                        st.error(f"Error submitting feedback: {str(e)}")
                                    finally:
                                        st.session_state.feedback_in_progress = False
                    else:
                        # Show previously given feedback type
                        feedback_type = st.session_state.get(feedback_type_key, None)
                        with feedback_container:
                            if feedback_type == "positive":
                                st.success("You gave this answer a thumbs up!", icon="👍")
                            elif feedback_type == "negative":
                                st.error("You suggested this answer could be improved.", icon="👎")
                            else:
                                st.info("Your feedback has been received.", icon="ℹ️")


    # Unified input container: agent selector + text input in one box
    _agent_labels = {
        "hybrid_agent":         "Hybrid Search",
        "deep_research_agent":  "Deep Research",
    }
    _label_to_key = {v: k for k, v in _agent_labels.items()}
    _labels = list(_agent_labels.values())
    _current_label = _agent_labels.get(st.session_state.get("agent_type", "hybrid_agent"), "Hybrid Search")

    with st.form("chat_input_form", clear_on_submit=True, border=False):
        col_agent, col_input, col_btn = st.columns([2, 8, 1])
        with col_agent:
            _selected_label = st.selectbox(
                "Agent",
                options=_labels,
                index=_labels.index(_current_label),
                label_visibility="collapsed",
                key="bottom_agent_label",
            )
        with col_input:
            prompt = st.text_input(
                "Message",
                placeholder="Enter your question...",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("➤", use_container_width=True)

    # Sync agent type (internal key) from the selected display label
    st.session_state.agent_type = _label_to_key.get(
        st.session_state.get("bottom_agent_label", "Hybrid Search"), "hybrid_agent"
    )

    # Block keyboard typing in the agent selectbox while keeping click-to-select functional
    components.html("""
    <script>
    (function() {
        function blockTyping() {
            try {
                var inputs = window.parent.document.querySelectorAll(
                    '[data-testid="stForm"] [data-testid="stSelectbox"] input'
                );
                inputs.forEach(function(inp) {
                    if (inp._typingBlocked) return;
                    inp._typingBlocked = true;
                    // Block all character input; allow navigation keys only
                    var navKeys = [9, 13, 27, 37, 38, 39, 40]; // Tab, Enter, Esc, Arrows
                    inp.addEventListener('keydown', function(e) {
                        if (navKeys.indexOf(e.keyCode) === -1) e.preventDefault();
                    }, true);
                    inp.addEventListener('keypress', function(e) {
                        e.preventDefault();
                    }, true);
                    // Clear any value that slips through
                    inp.addEventListener('input', function() {
                        inp.value = '';
                    }, true);
                });
            } catch(e) {}
        }
        setTimeout(blockTyping, 300);
        try {
            new MutationObserver(blockTyping)
                .observe(window.parent.document.body, { childList: true, subtree: true });
        } catch(e) {}
    })();
    </script>
    """, height=0)

    if submitted and prompt:
        if st.session_state.processing_lock:
            st.warning("Please wait for the current operation to complete...")
            return

        st.session_state.processing_lock = True

        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                message_placeholder = st.empty()
                full_response = ""
                thinking_content = ""

                use_stream = st.session_state.get("use_stream", True)

                if use_stream:
                    def handle_token(token, is_thinking=False):
                        nonlocal full_response, thinking_content
                        try:
                            if isinstance(token, str) and token.startswith("{") and token.endswith("}"):
                                try:
                                    json_data = json.loads(token)
                                    if "content" in json_data:
                                        token = json_data["content"]
                                    elif "status" in json_data:
                                        return
                                except json.JSONDecodeError:
                                    pass
                            if is_thinking:
                                thinking_content += token
                                quoted = '\n'.join([f"> {l}" for l in thinking_content.split('\n')])
                                message_placeholder.markdown(quoted)
                            else:
                                full_response += token
                                message_placeholder.markdown(full_response + "▌")
                        except Exception as e:
                            print(f"Error handling token: {str(e)}")

                    with st.spinner("Thinking..."):
                        try:
                            send_message_stream(prompt, handle_token)
                            if not full_response or (full_response.startswith("{") and full_response.endswith("}")):
                                response = send_message(prompt)
                                if response:
                                    full_response = response.get("answer", "")
                                    message_placeholder.markdown(full_response)
                        except Exception as e:
                            print(f"Streaming API failed: {str(e)}")
                            response = send_message(prompt)
                            if response:
                                full_response = response.get("answer", "")
                                message_placeholder.markdown(full_response)

                    message_placeholder.markdown(full_response)
                    message_obj = {
                        "role": "assistant",
                        "content": full_response,
                        "message_id": str(uuid.uuid4())
                    }
                    if thinking_content:
                        message_obj["raw_thinking"] = thinking_content
                        message_obj["processed_content"] = full_response
                else:
                    with st.spinner("Thinking..."):
                        response = send_message(prompt)

                    if response:
                        answer = response.get("answer", "Sorry, I was unable to process your request.")
                        message_placeholder.markdown(answer)
                        message_obj = {
                            "role": "assistant",
                            "content": answer,
                            "message_id": str(uuid.uuid4())
                        }
                        if "raw_thinking" in response:
                            message_obj["raw_thinking"] = response["raw_thinking"]
                            message_obj["processed_content"] = answer
                    else:
                        error_message = "Sorry, the server did not return a valid response."
                        message_placeholder.markdown(error_message)
                        message_obj = {
                            "role": "assistant",
                            "content": error_message,
                            "message_id": str(uuid.uuid4())
                        }

                st.session_state.messages.append(message_obj)

            except Exception as e:
                st.error(f"Error processing message: {str(e)}")
                traceback.print_exc()
            finally:
                st.session_state.processing_lock = False

        st.rerun()

def clear_chat_with_lock_reset():
    """Clear chat and reset the processing lock"""
    st.session_state.processing_lock = False
    clear_chat()
