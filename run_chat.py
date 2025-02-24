import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Import app the same way as main.py
from graph.graph import app

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def chat_response(question: str):
    """Get response from the RAG application."""
    try:
        return app.invoke(input={"question": question})
    except Exception as e:
        st.error(f"Error processing the question: {str(e)}")
        return {"generation": "An error occurred while processing your question."}

def main():
    st.title("Corrective RAG")
    
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Make a question..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_response(prompt)
                st.markdown(response.get("generation", "Sorry, I couldn't generate a response."))
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.get("generation", "Sorry, I couldn't generate a response.")})

        # Display debug information in expander
        with st.expander("Debug Information"):
            st.json(response)

if __name__ == "__main__":
    main()