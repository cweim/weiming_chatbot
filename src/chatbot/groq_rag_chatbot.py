# src/chatbot/groq_rag_chatbot.py

import json
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import requests
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from vector_store.faiss_manager import FAISSManager

class GroqRAGChatbot:
    """RAG Chatbot powered by Groq's free Llama API"""

    def __init__(self, vector_store_dir: str, groq_api_key: str = None):
        """
        Initialize RAG chatbot with Groq

        Args:
            vector_store_dir: Directory containing FAISS index and embeddings
            groq_api_key: Groq API key (get free at https://console.groq.com)
        """
        self.vector_store_dir = Path(vector_store_dir)
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model_name = "llama-3.2-3b-preview"  # Groq's Llama 3.2 3B model

        if not self.groq_api_key:
            print("⚠️  No Groq API key found!")
            print("Get your free API key at: https://console.groq.com")
            print("Then set it as: GROQ_API_KEY=your_key_here")

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

        # Test Groq connection
        self._test_groq_connection()

    def _test_groq_connection(self):
        """Test if Groq API is working"""
        if not self.groq_api_key:
            print("❌ No Groq API key - responses will show error messages")
            return

        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }

            test_payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            }

            response = requests.post(
                self.groq_url,
                headers=headers,
                json=test_payload,
                timeout=10
            )

            if response.status_code == 200:
                print(f"✅ Groq API connected! Using {self.model_name}")
            elif response.status_code == 401:
                print("❌ Invalid Groq API key!")
            elif response.status_code == 429:
                print("⚠️  Groq rate limit reached - try again later")
            else:
                print(f"⚠️  Groq API issue: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to Groq API: {str(e)}")

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

    def generate_prompt_messages(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate messages for Groq API with retrieved context"""

        # System message about Wei Ming
        system_message = """You are an AI assistant that helps people learn about Wei Ming Chin, a final-year engineering student at SUTD specializing in Design and Artificial Intelligence.

Use the provided context to answer questions accurately about Wei Ming's background, projects, skills, and experience. Be conversational, helpful, and specific.

If asked about contact information, be professional. If the context doesn't contain enough information to answer a question fully, say so honestly.

Always refer to him as "Wei Ming" in your responses. Keep responses concise but informative."""

        # Build context from retrieved chunks
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            chunk_type = chunk['metadata'].get('type', 'content')
            context_text += f"\n--- Context {i} ({chunk_type}) ---\n"
            context_text += chunk['content'][:800]  # Limit chunk size
            context_text += "\n"

        # Create messages array for Groq
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"Context information:\n{context_text}\n\nQuestion: {query}\n\nPlease answer based on the context provided."
            }
        ]

        return messages

    def query_groq(self, messages: List[Dict[str, str]], max_tokens: int = 500) -> str:
        """Query Groq API"""
        if not self.groq_api_key:
            return "⚠️ Groq API key not configured. Please set your GROQ_API_KEY environment variable. Get a free key at https://console.groq.com"

        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "top_p": 0.9,
                "stream": False
            }

            response = requests.post(
                self.groq_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            elif response.status_code == 401:
                return "❌ Invalid Groq API key. Please check your key at https://console.groq.com"
            elif response.status_code == 429:
                return "⚠️ Rate limit reached. The free tier allows 14,400 requests per day. Please try again later."
            elif response.status_code == 400:
                return "❌ Request error. The message might be too long or contain invalid content."
            else:
                return f"❌ Groq API error: {response.status_code}. Please try again."

        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. Groq might be busy. Please try again."
        except requests.exceptions.RequestException as e:
            return f"❌ Connection error: {str(e)}. Please check your internet connection."

    def chat(self, query: str, top_k: int = 5, max_tokens: int = 500) -> Dict[str, Any]:
        """Main chat function - retrieve context and generate response"""

        # Step 1: Retrieve relevant context
        print(f"🔍 Retrieving context for: '{query}'")
        context_chunks = self.retrieve_context(query, top_k=top_k)

        # Step 2: Generate messages for Groq
        messages = self.generate_prompt_messages(query, context_chunks)

        # Step 3: Query Groq
        print(f"🧠 Generating response with {self.model_name}...")
        response = self.query_groq(messages, max_tokens=max_tokens)

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
            'model_used': self.model_name,
            'api_provider': 'groq'
        }


# Test the chatbot
if __name__ == "__main__":
    # Get project root and vector store path
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    vector_store_dir = project_root / "data" / "vector_store"

    # Initialize chatbot
    chatbot = GroqRAGChatbot(str(vector_store_dir))

    # Test queries
    test_queries = [
        "What machine learning projects has Wei Ming worked on?",
        "Tell me about his technical skills and experience",
        "What are his career goals?",
        "How can I contact Wei Ming?",
        "What is his educational background?"
    ]

    print("\n" + "="*60)
    print("TESTING GROQ RAG CHATBOT")
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
