# streamlit_app.py

import streamlit as st
import sys
from pathlib import Path
import time
import json
import os

# Add src to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

from chatbot.groq_rag_chatbot import GroqRAGChatbot

# Page configuration
st.set_page_config(
    page_title="Chat with Wei Ming",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean, minimal design
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }

    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 3rem 0 2rem 0;
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
        line-height: 1.5;
    }

    /* Chat container */
    .chat-container {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 2rem;
        overflow: hidden;
    }

    /* Message styling */
    .message {
        padding: 1.25rem 1.5rem;
        border-bottom: 1px solid #f3f4f6;
        line-height: 1.6;
    }

    .message:last-child {
        border-bottom: none;
    }

    .user-message {
        background: #f9fafb;
        border-left: 3px solid #3b82f6;
    }

    .assistant-message {
        background: #ffffff;
    }

    .message-label {
        font-weight: 600;
        font-size: 0.875rem;
        color: #374151;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .message-content {
        color: #1f2937;
        font-size: 1rem;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 24px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem 1.25rem;
        font-size: 1rem;
        transition: all 0.2s ease;
        background: #ffffff;
    }

    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        border: none;
        background: #3b82f6;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    /* Suggestion pills */
    .suggestion-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin: 2rem 0;
    }

    .suggestion-pill {
        background: #f3f4f6;
        color: #374151;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid #e5e7eb;
        font-size: 0.875rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .suggestion-pill:hover {
        background: #e5e7eb;
        border-color: #d1d5db;
    }

    /* Thinking animation */
    .thinking {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #6b7280;
        font-style: italic;
        padding: 1rem 1.5rem;
        background: #f9fafb;
    }

    .thinking-dots {
        display: inline-flex;
        gap: 0.25rem;
    }

    .thinking-dot {
        width: 4px;
        height: 4px;
        background: #6b7280;
        border-radius: 50%;
        animation: thinking 1.4s infinite ease-in-out;
    }

    .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
    .thinking-dot:nth-child(2) { animation-delay: -0.16s; }

    @keyframes thinking {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #9ca3af;
        font-size: 0.875rem;
        border-top: 1px solid #f3f4f6;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_chatbot():
    """Initialize chatbot - cached to avoid reloading"""
    try:
        vector_store_dir = Path(__file__).parent / "data" / "vector_store"

        # Get Groq API key from environment or Streamlit secrets
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key and hasattr(st, 'secrets'):
            try:
                groq_api_key = st.secrets["GROQ_API_KEY"]
            except:
                pass

        chatbot = GroqRAGChatbot(str(vector_store_dir), groq_api_key=groq_api_key)
        return chatbot, None
    except Exception as e:
        return None, str(e)

def render_message(role, content, key=None):
    """Render a chat message with clean styling"""
    if role == "user":
        st.markdown(f"""
        <div class="message user-message">
            <div class="message-label">You</div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message assistant-message">
            <div class="message-label">Wei Ming's AI Assistant</div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)

def render_thinking():
    """Render thinking animation"""
    st.markdown("""
    <div class="thinking">
        <span>Thinking</span>
        <div class="thinking-dots">
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Initialize chatbot
    chatbot, error = initialize_chatbot()

    if error:
        st.error(f"❌ Failed to initialize chatbot: {error}")
        st.info("Please ensure you have set up your Groq API key.")

        # Show API key setup instructions
        st.markdown("""
        ### 🔑 Getting Your Free Groq API Key:
        1. Visit: https://console.groq.com
        2. Sign up for free account
        3. Go to API Keys section
        4. Create new API key
        5. Set it as environment variable: `GROQ_API_KEY=your_key_here`

        **Free tier includes 14,400 requests per day!**
        """)
        return

    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">Chat with Wei Ming</h1>
        <p class="hero-subtitle">
            AI assistant powered by RAG technology. Ask me anything about Wei Ming's projects,
            skills, experience, and background.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.show_suggestions = True

    # Suggestion pills (only show if no conversation started)
    if st.session_state.show_suggestions and len(st.session_state.messages) == 0:
        st.markdown('<div class="suggestion-pills">', unsafe_allow_html=True)

        suggestions = [
            "What are Wei Ming's technical skills?",
            "Tell me about his machine learning projects",
            "What's his educational background?",
            "How can I contact Wei Ming?",
            "What are his career goals?",
            "Deep learning experience"
        ]

        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 3]:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.current_question = suggestion
                    st.session_state.show_suggestions = False
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # Chat container
    if st.session_state.messages:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        for message in st.session_state.messages:
            render_message(message["role"], message["content"])

        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask me anything about Wei Ming...", key="chat_input"):
        st.session_state.show_suggestions = False
        process_message(prompt, chatbot)

    # Handle suggestion clicks
    if "current_question" in st.session_state:
        process_message(st.session_state.current_question, chatbot)
        del st.session_state.current_question

    # Footer
    st.markdown("""
    <div class="footer">
        Powered by Groq & Llama 3.2 • RAG Technology • 328 Knowledge Chunks
    </div>
    """, unsafe_allow_html=True)

def process_message(prompt, chatbot):
    """Process user message and generate response"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show thinking state
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        render_message(message["role"], message["content"])
    render_thinking()
    st.markdown('</div>', unsafe_allow_html=True)

    # Generate response
    try:
        result = chatbot.chat(prompt, top_k=3, max_tokens=400)
        response = result["response"]

        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        error_msg = f"I apologize, but I'm having trouble connecting to the language model. Please make sure Ollama is running with the model loaded. Error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

    st.rerun()

if __name__ == "__main__":
    main()
