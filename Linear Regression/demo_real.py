#!/usr/bin/env python
"""
Demo script showing RAG system with real construction lessons data.
"""
import sys
sys.path.insert(0, '/mnt/user-data/outputs/rag_poc')

from src import *


def demo():
    """Run demonstration queries on real data."""
    
    print("="*70)
    print("CONSTRUCTION LESSONS-LEARNED RAG SYSTEM")
    print("Real Data Demo - 111 Lessons Indexed")
    print("="*70)
    
    # Load index
    print("\nLoading index...")
    index = VectorIndex.load("./index_real")
    embedder = create_embedding_client(provider="sentence-transformers")
    retriever = Retriever(embedder, index)
    llm_client = StubLLMClient()
    
    print(f"✓ Loaded {len(index)} lessons from real project data")
    
    # Example queries based on real data
    queries = [
        "Structural engineering design issues",
        "Problems with estimating",
        "Civil engineering mistakes",
        "Quality control problems",
        "Electrical design errors",
    ]
    
    print("\n" + "="*70)
    print("RUNNING EXAMPLE QUERIES")
    print("="*70)
    
    for i, question in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {question}")
        print("-"*70)
        
        result = answer_question(
            question=question,
            retriever=retriever,
            llm_client=llm_client,
            k=3,
            verbose=False
        )
        
        if result['hits']:
            print(f"\nFound {len(result['hits'])} relevant lessons:")
            for j, hit in enumerate(result['hits'], 1):
                metadata = hit['metadata']
                print(f"\n  [{j}] {metadata['lesson_id']}")
                print(f"      Operations: {metadata['operations']}")
                print(f"      Business Area: {metadata['business_area']}")
                print(f"      Project: {metadata['project']}")
                print(f"      Score: {hit['score']:.3f}")
        else:
            print("\nNo lessons found matching criteria")
        
        print()
    
    print("="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\n✓ Using sentence-transformers for semantic search (FREE)")
    print("✓ Using stub LLM (just lists lessons)")
    print("\nNext step: Add OpenAI GPT-4 for intelligent answers")
    print("See README.md for OpenAI LLM setup")
    print("\nTry interactive mode: python -m src.app interactive --index-dir ./index_real")


if __name__ == "__main__":
    demo()
