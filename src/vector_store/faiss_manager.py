# src/vector_store/faiss_manager.py

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle

class FAISSManager:
    """Manage FAISS vector store for RAG retrieval"""

    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        """
        Initialize FAISS manager

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata = None

    def create_index_from_embeddings(self, embeddings_dir: str) -> str:
        """
        Create FAISS index from generated embeddings

        Args:
            embeddings_dir: Directory containing embeddings and metadata

        Returns:
            Path to saved index file
        """
        embeddings_path = Path(embeddings_dir)

        # Load embeddings and metadata
        print("📖 Loading embeddings and metadata...")
        embeddings = np.load(embeddings_path / "embeddings.npy")

        with open(embeddings_path / "metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        with open(embeddings_path / "config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)

        print(f"✅ Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

        # Create FAISS index
        self.embedding_dim = embeddings.shape[1]
        self.index = self._create_faiss_index(embeddings)

        # Save index and metadata
        index_file = embeddings_path / "faiss_index.index"
        metadata_file = embeddings_path / "faiss_metadata.pkl"

        # Save FAISS index
        faiss.write_index(self.index, str(index_file))
        print(f"💾 Saved FAISS index to {index_file}")

        # Save metadata for retrieval
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"💾 Saved metadata to {metadata_file}")

        # Update config
        config['faiss_index_file'] = str(index_file)
        config['faiss_metadata_file'] = str(metadata_file)
        config['index_type'] = self.index_type

        with open(embeddings_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ FAISS index created successfully!")
        self._print_index_info()

        return str(index_file)

    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create FAISS index based on type and data size"""
        n_embeddings, dim = embeddings.shape

        print(f"🔧 Creating FAISS index (type: {self.index_type})")

        if self.index_type == "flat" or n_embeddings < 1000:
            # Use flat index for small datasets or when specified
            index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity)
            print("📊 Using IndexFlatIP (exact search)")

        elif self.index_type == "ivf":
            # Inverted file index for larger datasets
            nlist = min(int(np.sqrt(n_embeddings)), 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            print(f"📊 Using IndexIVFFlat with {nlist} clusters")

            # Train the index
            print("🏋️ Training IVF index...")
            index.train(embeddings.astype(np.float32))

        else:  # Default to flat
            index = faiss.IndexFlatIP(dim)
            print("📊 Using IndexFlatIP (default)")

        # Add embeddings to index
        print("📥 Adding embeddings to index...")

        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings_normalized)

        index.add(embeddings_normalized)

        print(f"✅ Added {index.ntotal} vectors to index")

        return index

    def load_index(self, index_dir: str):
        """Load existing FAISS index and metadata"""
        index_path = Path(index_dir)

        index_file = index_path / "faiss_index.index"
        metadata_file = index_path / "faiss_metadata.pkl"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")

        # Load index
        self.index = faiss.read_index(str(index_file))
        print(f"✅ Loaded FAISS index from {index_file}")

        # Load metadata
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"✅ Loaded metadata for {len(self.metadata)} chunks")

        self._print_index_info()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of similar chunks with metadata and scores
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call create_index_from_embeddings() or load_index() first.")

        # Normalize query embedding
        query_normalized = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_normalized)

        # Search
        scores, indices = self.index.search(query_normalized, top_k)

        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:  # Valid result
                chunk_metadata = self.metadata[idx]
                results.append({
                    'rank': i + 1,
                    'score': float(score),
                    'chunk_id': chunk_metadata['id'],
                    'metadata': chunk_metadata['metadata'],
                    'source_type': chunk_metadata['source_type'],
                    'source_file': chunk_metadata['source_file'],
                    'word_count': chunk_metadata['word_count']
                })

        return results

    def _print_index_info(self):
        """Print information about the loaded index"""
        if self.index:
            print(f"\n📊 FAISS Index Info:")
            print(f"  Index type: {type(self.index).__name__}")
            print(f"  Dimension: {self.index.d}")
            print(f"  Total vectors: {self.index.ntotal}")
            print(f"  Is trained: {self.index.is_trained}")

    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """Get chunk metadata by ID"""
        for chunk in self.metadata:
            if chunk['id'] == chunk_id:
                return chunk
        return None


# Example usage and testing
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # Get project root
    current_dir = Path(__file__).parent  # src/vector_store/
    project_root = current_dir.parent.parent  # Go up to project root

    embeddings_dir = project_root / "data" / "vector_store"

    print(f"📂 Embeddings directory: {embeddings_dir}")

    # Check if embeddings exist
    if not (embeddings_dir / "embeddings.npy").exists():
        print("❌ Embeddings not found. Please run embedding generation first!")
        print("Run: python src/embeddings/embedding_generator.py")
        exit(1)

    # Load config to get embedding dimension
    with open(embeddings_dir / "config.json", 'r') as f:
        config = json.load(f)

    embedding_dim = config['embedding_dimension']
    model_name = config['model_name']

    print(f"📊 Embedding dimension: {embedding_dim}")
    print(f"🤖 Model: {model_name}")

    # Create FAISS manager
    faiss_manager = FAISSManager(embedding_dim, index_type="flat")

    # Create index
    index_file = faiss_manager.create_index_from_embeddings(str(embeddings_dir))

    print(f"\n🎉 FAISS index created successfully!")

    # Test search functionality
    print("\n" + "="*50)
    print("TESTING FAISS SEARCH")
    print("="*50)

    # Load sentence transformer for queries
    print("🔄 Loading sentence transformer for testing...")
    model = SentenceTransformer(model_name)

    # Test queries
    test_queries = [
        "What machine learning projects has Wei Ming worked on?",
        "Tell me about his technical skills",
        "What is his contact information?",
        "Deep learning experience",
        "Career goals and aspirations"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")

        # Generate query embedding
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Search
        results = faiss_manager.search(query_embedding, top_k=3)

        print(f"📊 Top {len(results)} results:")
        for result in results:
            print(f"  {result['rank']}. Score: {result['score']:.3f}")
            print(f"     ID: {result['chunk_id']}")
            print(f"     Type: {result['metadata'].get('type', 'unknown')}")
            print(f"     Title: {result['metadata'].get('title', 'No title')}")

        print("-" * 40)

    print("\n✅ FAISS vector store is ready for RAG!")
