# src/embeddings/embedding_generator.py

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

class EmbeddingGenerator:
    """Generate embeddings for content chunks using sentence transformers"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize with embedding model

        Args:
            model_name: HuggingFace model name for embeddings
                       - all-MiniLM-L6-v2: Fast, good quality, 384 dimensions
                       - all-mpnet-base-v2: Higher quality, 768 dimensions (slower)
        """
        self.model_name = model_name
        print(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate_embeddings_from_chunks(self, chunks_file: str, output_dir: str) -> Dict[str, Any]:
        """
        Generate embeddings for all chunks and save to files

        Args:
            chunks_file: Path to final_chunks.json
            output_dir: Directory to save embeddings and metadata

        Returns:
            Dictionary with embedding info
        """
        chunks_path = Path(chunks_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load chunks
        print(f"📖 Loading chunks from {chunks_path}")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"📊 Found {len(chunks)} chunks to process")

        # Extract content for embedding
        texts = []
        metadata = []

        for chunk in chunks:
            texts.append(chunk['content'])
            metadata.append({
                'id': chunk['id'],
                'metadata': chunk['metadata'],
                'source_type': chunk['source_type'],
                'source_file': chunk['source_file'],
                'word_count': chunk['word_count']
            })

        # Generate embeddings with progress bar
        print(f"🧠 Generating embeddings using {self.model_name}...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"📐 Embedding shape: {embeddings.shape}")

        # Save embeddings and metadata
        embeddings_file = output_path / "embeddings.npy"
        metadata_file = output_path / "metadata.json"
        config_file = output_path / "config.json"

        # Save as numpy array for efficiency
        np.save(embeddings_file, embeddings)
        print(f"💾 Saved embeddings to {embeddings_file}")

        # Save metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved metadata to {metadata_file}")

        # Save configuration
        config = {
            'model_name': self.model_name,
            'embedding_dimension': embeddings.shape[1],
            'num_chunks': embeddings.shape[0],
            'chunks_file': str(chunks_path),
            'created_timestamp': str(Path().resolve())
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Saved config to {config_file}")

        # Print summary
        self._print_embedding_summary(embeddings, metadata)

        return {
            'embeddings_file': str(embeddings_file),
            'metadata_file': str(metadata_file),
            'config_file': str(config_file),
            'num_embeddings': len(embeddings),
            'embedding_dim': embeddings.shape[1]
        }

    def _print_embedding_summary(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Print summary of generated embeddings"""
        print("\n📊 Embedding Summary:")
        print(f"  Total embeddings: {len(embeddings)}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        print(f"  Total size: {embeddings.nbytes / 1024 / 1024:.2f} MB")

        # Count by content type
        type_counts = {}
        priority_counts = {}

        for meta in metadata:
            content_type = meta['metadata'].get('type', 'unknown')
            priority = meta['metadata'].get('priority', 'unknown')

            type_counts[content_type] = type_counts.get(content_type, 0) + 1
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        print("\n  By Content Type:")
        for content_type, count in sorted(type_counts.items()):
            print(f"    {content_type}: {count}")

        print("\n  By Priority:")
        for priority, count in sorted(priority_counts.items()):
            print(f"    {priority}: {count}")

    def test_similarity(self, query: str, embeddings_dir: str, top_k: int = 3):
        """
        Test similarity search with a query

        Args:
            query: Test query string
            embeddings_dir: Directory with embeddings
            top_k: Number of results to return
        """
        embeddings_path = Path(embeddings_dir)

        # Load embeddings and metadata
        embeddings = np.load(embeddings_path / "embeddings.npy")
        with open(embeddings_path / "metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Calculate similarities
        similarities = np.dot(embeddings, query_embedding.T).flatten()

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        print(f"\n🔍 Query: '{query}'")
        print(f"📊 Top {top_k} similar chunks:")

        for i, idx in enumerate(top_indices, 1):
            similarity = similarities[idx]
            chunk_id = metadata[idx]['id']
            chunk_type = metadata[idx]['metadata'].get('type', 'unknown')

            print(f"\n  {i}. Similarity: {similarity:.3f}")
            print(f"     ID: {chunk_id}")
            print(f"     Type: {chunk_type}")
            print(f"     Preview: {metadata[idx]['metadata'].get('title', 'No title')}")


# Example usage and testing
if __name__ == "__main__":
    import os

    # Get project root
    current_dir = Path(__file__).parent  # src/embeddings/
    project_root = current_dir.parent.parent  # Go up to project root

    # Set paths
    chunks_file = project_root / "data" / "processed" / "final_chunks.json"
    output_dir = project_root / "data" / "vector_store"

    print(f"📂 Chunks file: {chunks_file}")
    print(f"📂 Output directory: {output_dir}")

    # Check if chunks file exists
    if not chunks_file.exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        print("Please run the content processing pipeline first!")
        exit(1)

    # Initialize generator
    generator = EmbeddingGenerator()

    # Generate embeddings
    result = generator.generate_embeddings_from_chunks(
        chunks_file=str(chunks_file),
        output_dir=str(output_dir)
    )

    print(f"\n🎉 Embeddings generated successfully!")
    print(f"📁 Files created in: {output_dir}")

    # Test with sample queries
    print("\n" + "="*50)
    print("TESTING SIMILARITY SEARCH")
    print("="*50)

    test_queries = [
        "machine learning projects",
        "deep learning experience",
        "contact information",
        "technical skills"
    ]

    for query in test_queries:
        generator.test_similarity(query, str(output_dir), top_k=2)
        print("-" * 30)
