#!/bin/bash
# launch_chatbot.sh - Easy launcher for Wei Ming's RAG Chatbot

echo "🤖 Starting Wei Ming's RAG Chatbot..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama not running. Please start it:"
    echo "   ollama serve"
    exit 1
fi

# Launch Streamlit
echo "🌐 Starting web interface..."
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

echo "✅ Chatbot should open in your browser!"
echo "🔗 If not, visit: http://localhost:8501"
