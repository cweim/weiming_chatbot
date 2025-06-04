# src/chatbot/rag_chatbot.py

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import requests
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from vector_store.faiss_manager import FAISSManager

class RAGChatbot:
    """RAG Chatbot powered by Llama via Ollama"""

    def __init__(self, vector_store_dir: str, llama_model: str = "llama3.2:3b"):
        """
        Initialize RAG chatbot

        Args:
            vector_store_dir: Directory containing FAISS index and embeddings
            llama_model: Ollama model name (e.g., "llama3.2:3b", "llama3.2:1b")
        """
        self.vector_store_dir = Path(vector_store_dir)
        self.llama_model = llama_model
        self.ollama_url = "http://localhost:11434/api/generate"

        # Load configuration
        with open(self.vector_store_dir / "config.json", 'r') as f:
            self.config = json.load(f)

        # Initialize embedding model
        print(f"🔄 Loading embedding model: {self.config['model_name']}")
        self.embedding_model = SentenceTransformer(self.config['model_name'])

        # Initialize FAISS manager
        print("🔄 Loading FAISS index...")
        self.faiss_manager = FAISSManager(self.config['embedding_dimension'])
        self.faiss_manager.load_index(str(self.vector_store_dir))

        print(f"✅ RAG Chatbot initialized with {self.faiss_manager.index.ntotal} chunks")

        # Test Ollama connection
        self._test_ollama_connection()

    def _test_ollama_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = requests.post(
                "http://localhost:11434/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]

                if self.llama_model in model_names:
                    print(f"✅ Ollama connected. Model '{self.llama_model}' ready!")
                else:
                    print(f"⚠️  Model '{self.llama_model}' not found.")
                    print(f"Available models: {model_names}")
                    print(f"To install: ollama pull {self.llama_model}")
            else:
                print("⚠️  Ollama server not responding properly")

        except requests.exceptions.RequestException:
            print("❌ Ollama not running. Please start Ollama:")
            print("1. Install Ollama: https://ollama.ai")
            print("2. Run: ollama serve")
            print(f"3. Pull model: ollama pull {self.llama_model}")

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks for a query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)

        # Search FAISS index
        results = self.faiss_manager.search(query_embedding, top_k=top_k)

        # Load actual chunk content
        chunks_file = self.vector_store_dir.parent / "processed" / "final_chunks.json"
        with open(chunks_file, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)

        # Create a lookup dictionary
        chunk_lookup = {chunk['id']: chunk['content'] for chunk in all_chunks}

        # Add full content to results
        for result in results:
            chunk_id = result['chunk_id']
            result['content'] = chunk_lookup.get(chunk_id, "Content not found")

        return results

    def generate_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate prompt for Llama with retrieved context"""

        # System prompt about Wei Ming
        system_prompt = """You are an AI assistant that helps people learn about Wei Ming Chin, a final-year engineering student at SUTD specializing in Design and Artificial Intelligence.

Use the provided context to answer questions accurately about Wei Ming's background, projects, skills, and experience. Be conversational, helpful, and specific.

If asked about contact information, be professional. If the context doesn't contain enough information to answer a question fully, say so honestly.

Always refer to him as "Wei Ming" in your responses."""

        # Build context from retrieved chunks
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            chunk_type = chunk['metadata'].get('type', 'content')
            context_text += f"\n--- Context {i} ({chunk_type}) ---\n"
            context_text += chunk['content'][:800]  # Limit chunk size
            context_text += "\n"

        # Combine into final prompt
        prompt = f"""{system_prompt}

CONTEXT INFORMATION:
{context_text}

QUESTION: {query}

ANSWER:"""

        return prompt

    def query_llama(self, prompt: str, max_tokens: int = 500) -> str:
        """Query Llama via Ollama API"""
        try:
            payload = {
                "model": self.llama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": max_tokens,
                    "stop": ["QUESTION:", "CONTEXT:"]
                }
            }

            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Error: Ollama returned status {response.status_code}"

        except requests.exceptions.Timeout:
            return "Error: Request timed out. Llama might be processing a complex query."
        except requests.exceptions.RequestException as e:
            return f"Error: Could not connect to Ollama. Is it running? ({str(e)})"

    def chat(self, query: str, top_k: int = 5, max_tokens: int = 500) -> Dict[str, Any]:
        """Main chat function - retrieve context and generate response"""

        # Step 1: Retrieve relevant context
        print(f"🔍 Retrieving context for: '{query}'")
        context_chunks = self.retrieve_context(query, top_k=top_k)

        # Step 2: Generate prompt
        prompt = self.generate_prompt(query, context_chunks)

        # Step 3: Query Llama
        print(f"🧠 Generating response with {self.llama_model}...")
        response = self.query_llama(prompt, max_tokens=max_tokens)

        # Step 4: Return structured result
        return {
            'query': query,
            'response': response,
            'context_chunks': [
                {
                    'id': chunk['chunk_id'],
                    'type': chunk['metadata'].get('type', 'unknown'),
                    'title': chunk['metadata'].get('title', 'No title'),
                    'score': chunk['score'],
                    'preview': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
                }
                for chunk in context_chunks
            ],
            'model_used': self.llama_model
        }


# Test the chatbot
if __name__ == "__main__":
    # Get project root and vector store path
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    vector_store_dir = project_root / "data" / "vector_store"

    # Initialize chatbot
    chatbot = RAGChatbot(str(vector_store_dir))

    # Test queries
    test_queries = [
        "What machine learning projects has Wei Ming worked on?",
        "Tell me about his technical skills and experience",
        "What are his career goals?",
        "How can I contact Wei Ming?",
        "What is his educational background?"
    ]

    print("\n" + "="*60)
    print("TESTING RAG CHATBOT")
    print("="*60)

    for query in test_queries:
        print(f"\n🔹 Query: {query}")
        print("-" * 50)

        result = chatbot.chat(query, top_k=3)

        print(f"🤖 Response: {result['response']}")
        print(f"\n📊 Sources used:")
        for chunk in result['context_chunks']:
            print(f"  - {chunk['type']}: {chunk['title']} (score: {chunk['score']:.3f})")

        print("\n" + "="*60)
