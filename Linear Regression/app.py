"""
Main application entry point.
Provides CLI for building index and running queries.
"""
import argparse
from pathlib import Path
from typing import Optional
import sys

from .models import Lesson, lesson_to_canonical_text
from .ingest import load_lessons_from_excel, validate_lessons
from .embeddings import create_embedding_client, EmbeddingClient
from .index import VectorIndex
from .retrieve import Retriever
from .generate import answer_question, StubLLMClient


def build_index(
    excel_path: str,
    index_dir: str,
    embedding_client: Optional[EmbeddingClient] = None
) -> None:
    """
    Build vector index from Excel file.
    
    Steps:
    1. Load lessons from Excel
    2. Validate
    3. Create embeddings
    4. Build FAISS index
    5. Save to disk
    
    Args:
        excel_path: Path to Excel file
        index_dir: Directory to save index
        embedding_client: Optional custom embedder (uses stub if None)
    """
    print("=" * 60)
    print("BUILDING RAG INDEX")
    print("=" * 60)
    
    # Load lessons
    print(f"\n[1/5] Loading lessons from {excel_path}...")
    lessons = load_lessons_from_excel(excel_path)
    
    # Validate
    print(f"\n[2/5] Validating {len(lessons)} lessons...")
    validate_lessons(lessons)
    
    # Create embeddings
    print(f"\n[3/5] Creating embeddings...")
    if embedding_client is None:
        print("  Using stub embedder (for testing only)")
        embedding_client = create_embedding_client(provider="stub")
    
    # Convert lessons to canonical text
    texts = [lesson_to_canonical_text(lesson) for lesson in lessons]
    metadatas = [lesson.to_metadata() for lesson in lessons]
    
    # Embed
    print(f"  Embedding {len(texts)} documents...")
    vectors = embedding_client.embed_texts(texts)
    print(f"  Created {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
    
    # Build index
    print(f"\n[4/5] Building vector index...")
    index = VectorIndex(dimension=embedding_client.embedding_dim)
    index.add(vectors=vectors, metadatas=metadatas, texts=texts)
    print(f"  Index contains {len(index)} documents")
    
    # Save
    print(f"\n[5/5] Saving index to {index_dir}...")
    index.save(index_dir)
    
    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"\nIndex saved to: {index_dir}")
    print(f"Documents indexed: {len(lessons)}")
    print(f"Embedding dimension: {embedding_client.embedding_dim}")
    print("\nRun queries with: python -m src.app query --index-dir {index_dir}")


def run_query(
    question: str,
    index_dir: str,
    embedding_client: Optional[EmbeddingClient] = None,
    k: int = 5,
    verbose: bool = True
) -> None:
    """
    Run a single query against the index.
    
    Args:
        question: User question
        index_dir: Path to index directory
        embedding_client: Optional embedder (must match index)
        k: Number of results
        verbose: Print detailed output
    """
    print("=" * 60)
    print("QUERYING RAG SYSTEM")
    print("=" * 60)
    
    # Load index
    print(f"\n[1/3] Loading index from {index_dir}...")
    index = VectorIndex.load(index_dir)
    
    # Create embedder (must match the one used for indexing)
    if embedding_client is None:
        print("  Using stub embedder")
        embedding_client = create_embedding_client(provider="stub")
    
    # Create retriever
    print(f"\n[2/3] Setting up retriever...")
    retriever = Retriever(
        embedder=embedding_client,
        index=index,
        default_k=k
    )
    
    # Create LLM client (stub for now)
    llm_client = StubLLMClient()
    
    # Query
    print(f"\n[3/3] Processing question...")
    print(f"\nQuestion: {question}\n")
    
    result = answer_question(
        question=question,
        retriever=retriever,
        llm_client=llm_client,
        k=k,
        verbose=verbose
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nAnswer:\n{result['answer']}\n")
    
    if result['citations']:
        print(f"Citations: {', '.join(result['citations'])}")
    
    if verbose and result['hits']:
        print(f"\nRetrieved {result['num_hits']} lessons:")
        for i, hit in enumerate(result['hits'], 1):
            metadata = hit['metadata']
            print(f"\n  [{i}] {metadata['lesson_id']} (score: {hit['score']:.3f})")
            print(f"      Discipline: {metadata['discipline']}")
            print(f"      Phase: {metadata['phase']}")
            print(f"      Project: {metadata['project_type']}")


def run_interactive(
    index_dir: str,
    embedding_client: Optional[EmbeddingClient] = None,
    k: int = 5
) -> None:
    """
    Interactive query loop.
    
    Args:
        index_dir: Path to index directory
        embedding_client: Optional embedder
        k: Number of results per query
    """
    print("=" * 60)
    print("INTERACTIVE RAG SYSTEM")
    print("=" * 60)
    
    # Load index
    print(f"\nLoading index from {index_dir}...")
    index = VectorIndex.load(index_dir)
    
    # Create embedder
    if embedding_client is None:
        print("Using stub embedder")
        embedding_client = create_embedding_client(provider="stub")
    
    # Create retriever
    retriever = Retriever(
        embedder=embedding_client,
        index=index,
        default_k=k
    )
    
    # Create LLM client
    llm_client = StubLLMClient()
    
    print(f"\nReady! Index contains {len(index)} lessons.")
    print("Type your questions below. Type 'quit' to exit.\n")
    
    # Query loop
    while True:
        try:
            question = input("Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Process query
            result = answer_question(
                question=question,
                retriever=retriever,
                llm_client=llm_client,
                k=k,
                verbose=False
            )
            
            # Display
            print(f"\nAnswer: {result['answer']}\n")
            
            if result['citations']:
                print(f"Sources: {', '.join(result['citations'][:5])}")
                if len(result['citations']) > 5:
                    print(f"         (and {len(result['citations']) - 5} more)")
            
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Construction Lessons-Learned RAG System"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build index from Excel")
    build_parser.add_argument(
        "--excel",
        required=True,
        help="Path to Excel file with lessons"
    )
    build_parser.add_argument(
        "--index-dir",
        default="./index",
        help="Directory to save index (default: ./index)"
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument(
        "--index-dir",
        default="./index",
        help="Path to index directory"
    )
    query_parser.add_argument(
        "--question",
        required=True,
        help="Question to ask"
    )
    query_parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of results to retrieve"
    )
    
    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Start interactive query session"
    )
    interactive_parser.add_argument(
        "--index-dir",
        default="./index",
        help="Path to index directory"
    )
    interactive_parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of results per query"
    )
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == "build":
        build_index(
            excel_path=args.excel,
            index_dir=args.index_dir
        )
    
    elif args.command == "query":
        run_query(
            question=args.question,
            index_dir=args.index_dir,
            k=args.k
        )
    
    elif args.command == "interactive":
        run_interactive(
            index_dir=args.index_dir,
            k=args.k
        )
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
