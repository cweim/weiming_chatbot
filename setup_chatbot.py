# setup_chatbot.py
"""
Complete setup script for Wei Ming's RAG Chatbot
"""

import subprocess
import sys
import os
import requests
import time
from pathlib import Path

def check_python_packages():
    """Check and install required Python packages"""
    print("📦 Checking Python packages...")

    required_packages = [
        "streamlit",
        "sentence-transformers",
        "faiss-cpu",
        "requests",
        "numpy",
        "pathlib"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"📥 Installing missing packages: {missing_packages}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("✅ Packages installed!")
    else:
        print("✅ All packages already installed!")

def check_ollama():
    """Check if Ollama is installed and running"""
    print("\n🦙 Checking Ollama...")

    # Check if Ollama is installed
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed!")
        else:
            print("❌ Ollama not found!")
            print_ollama_install_instructions()
            return False
    except FileNotFoundError:
        print("❌ Ollama not found!")
        print_ollama_install_instructions()
        return False

    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running!")

            # Check for Llama models
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]

            llama_models = [m for m in model_names if 'llama3.2' in m]

            if llama_models:
                print(f"✅ Found Llama 3.2 models: {llama_models}")
                return True
            else:
                print("⚠️  No Llama 3.2 models found!")
                print("📥 Would you like to download llama3.2:3b? (Recommended, ~2GB)")
                choice = input("Download? (y/n): ").lower()

                if choice == 'y':
                    print("📥 Downloading llama3.2:3b... This may take a few minutes.")
                    result = subprocess.run(["ollama", "pull", "llama3.2:3b"])
                    if result.returncode == 0:
                        print("✅ llama3.2:3b downloaded successfully!")
                        return True
                    else:
                        print("❌ Failed to download model")
                        return False
                else:
                    print("⚠️  You'll need to download a model manually:")
                    print("   ollama pull llama3.2:3b")
                    return False
        else:
            print("❌ Ollama server not responding!")
            print("🚀 Please start Ollama server: ollama serve")
            return False

    except requests.exceptions.RequestException:
        print("❌ Cannot connect to Ollama server!")
        print("🚀 Please start Ollama server in another terminal:")
        print("   ollama serve")
        return False

def print_ollama_install_instructions():
    """Print Ollama installation instructions"""
    print("\n📥 To install Ollama:")
    print("1. Visit: https://ollama.ai")
    print("2. Download for your OS")
    print("3. Install and restart terminal")
    print("4. Run: ollama serve")
    print("5. Run: ollama pull llama3.2:3b")

def check_vector_store():
    """Check if vector store exists"""
    print("\n🔍 Checking vector store...")

    vector_store_dir = Path("data/vector_store")
    required_files = [
        "embeddings.npy",
        "faiss_index.index",
        "metadata.json",
        "config.json"
    ]

    missing_files = []
    for file in required_files:
        if not (vector_store_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing vector store files: {missing_files}")
        print("\n🔧 Please run the setup pipeline first:")
        print("1. python run_processing.py")
        print("2. python src/embeddings/embedding_generator.py")
        print("3. python src/vector_store/faiss_manager.py")
        return False
    else:
        print("✅ Vector store ready!")

        # Show stats
        try:
            import json
            with open(vector_store_dir / "config.json") as f:
                config = json.load(f)
            print(f"   📊 {config['num_chunks']} chunks ready")
            print(f"   🧠 Using {config['model_name']}")
        except:
            pass

        return True

def create_launch_script():
    """Create easy launch script"""
    print("\n🚀 Creating launch script...")

    launch_script = """#!/bin/bash
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
"""

    with open("launch_chatbot.sh", "w") as f:
        f.write(launch_script)

    # Make executable
    os.chmod("launch_chatbot.sh", 0o755)
    print("✅ Created launch_chatbot.sh")

def main():
    print("🤖 Setting up Wei Ming's RAG Chatbot")
    print("="*50)

    # Step 1: Check Python packages
    check_python_packages()

    # Step 2: Check Ollama
    if not check_ollama():
        print("\n❌ Ollama setup incomplete. Please install and configure Ollama first.")
        return

    # Step 3: Check vector store
    if not check_vector_store():
        print("\n❌ Vector store not ready. Please run the content processing pipeline first.")
        return

    # Step 4: Create launcher
    create_launch_script()

    # Final instructions
    print("\n" + "="*50)
    print("🎉 SETUP COMPLETE!")
    print("="*50)

    print("\n🚀 To start your chatbot:")
    print("1. Make sure Ollama is running: ollama serve")
    print("2. Launch the web app: streamlit run streamlit_app.py")
    print("3. Or use the launcher: ./launch_chatbot.sh")

    print("\n🌐 Your chatbot will be available at:")
    print("   http://localhost:8501")

    print("\n💡 To make it accessible from other devices:")
    print("   streamlit run streamlit_app.py --server.address 0.0.0.0")

    print("\n📱 Features of your chatbot:")
    print("✅ Web-based chat interface")
    print("✅ RAG-powered responses using your portfolio")
    print("✅ Powered by Llama 3.2 (100% free)")
    print("✅ 328 knowledge chunks about Wei Ming")
    print("✅ Smart source attribution")
    print("✅ Real-time response generation")

if __name__ == "__main__":
    main()
