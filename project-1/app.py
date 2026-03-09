import streamlit as st
import google.generativeai as genai
import time

apiKey = "AIzaSyBmVE005UU_60bZLH0SNwYSqxiz4ocbLjo" 

def get_gemini_response(user_query, chat_history):
    """
    Sends the user query and history to Gemini 2.5 Flash.
    Implements exponential backoff for reliability.
    """
    genai.configure(api_key=apiKey)
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    # Format history for Gemini API
    # Gemini expects a list of {role: "user"|"model", parts: [{text: "..."}]}
    history = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [{"text": msg["content"]}]})

    chat = model.start_chat(history=history)
    
    retries = 5
    for i in range(retries):
        try:
            response = chat.send_message(user_query)
            return response.text
        except Exception as e:
            if i == retries - 1:
                return f"Error: After multiple attempts, the API failed. {str(e)}"
            wait_time = 2**i
            time.sleep(wait_time)

# --- Streamlit UI ---
st.set_page_config(page_title="Gemini AI Chat", page_icon="🤖", layout="centered")

st.title("🚀 Gemini Chat Bot")
st.caption("A simple chatbot powered by Google's Gemini 2.5 Flash")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is on your mind?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*(Thinking...)*")
        
        # Get response from API
        full_response = get_gemini_response(prompt, st.session_state.messages[:-1])
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})