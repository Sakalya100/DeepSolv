import streamlit as st
import uuid
from streamlit_chat import message
import time
from langchain_utils import get_langchain_response
from db_utils import insert_application_logs
from common_utils import get_page_content

def get_answer(user_query, session_id, model, debug=False):
    try:
        response, response_time, input_tokens, output_tokens, total_tokens, refined_query, retrieved_context = get_langchain_response(user_query, session_id)
        context = get_page_content(retrieved_context)
        # insert_application_logs(session_id, user_query, response, model, response_time, input_tokens, output_tokens, total_tokens)

        if not debug:
            return {"answer": response}
        else:
            return {
                "answer": response,
                "original_query": user_query,
                "refined_query": refined_query,
                "retrieved_context": context,
            }
    except Exception as e:
        return {"error": f"An error has occurred: {str(e)}"}

st.title("AI Chat Interface")

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for settings
st.sidebar.title("Settings")
model = st.sidebar.selectbox("Select Model", ["GPT-3.5",])
debug = st.sidebar.checkbox("Debug Mode")

# Chat interface
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"{i}_user")
    else:
        message(msg["content"], key=f"{i}_ai")

# User input
user_query = st.text_input("Ask a question:")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Get AI response
    with st.spinner("Thinking..."):
        response = get_answer(user_query, st.session_state.session_id, model, debug)
    
    # Display AI response
    if "error" in response:
        st.error(response["error"])
    else:
        ai_message = response["answer"]
        st.session_state.messages.append({"role": "assistant", "content": ai_message})
        message(ai_message)
        
        # Display debug information if enabled
        if debug:
            with st.expander("Debug Information"):
                st.write("Original Query:", response["original_query"])
                st.write("Refined Query:", response["refined_query"])
                st.write("Retrieved Context:", response["retrieved_context"])

    # Clear the input box
    # st.rerun()

# Display session ID
st.sidebar.text(f"Session ID: {st.session_state.session_id}")