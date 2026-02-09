#!/usr/bin/env python
"""
Quick demo script to show RAG system capabilities.
Runs example queries and shows the output.
"""
from src import *


def demo():
    """Run demonstration queries."""
    
    print("="*70)
    print("CONSTRUCTION LESSONS-LEARNED RAG SYSTEM DEMO")
    print("="*70)
    
    # Load index
    print("\nLoading index...")
    index = VectorIndex.load("./index")
    embedder = create_embedding_client(provider="stub")
    retriever = Retriever(embedder, index)
    llm_client = StubLLMClient()
    
    print(f"✓ Loaded {len(index)} lessons")
    
    # Example queries
    queries = [
        "What electrical mistakes have we seen?",
        "Problems with instrumentation",
        "Issues during the design phase",
        "Coordination problems between teams",
        "Vendor-related issues",
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
        
        print(f"\nAnswer: {result['answer']}")
        
        if result['citations']:
            print(f"\nLessons Found: {', '.join(result['citations'])}")
            
            # Show brief details of top hit
            if result['hits']:
                top = result['hits'][0]
                metadata = top['metadata']
                print(f"\nTop Match: {metadata['lesson_id']}")
                print(f"  Discipline: {metadata['discipline']}")
                print(f"  Phase: {metadata['phase']}")
                print(f"  Score: {top['score']:.3f}")
        
        print()
    
    print("="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nTry interactive mode: python -m src.app interactive --index-dir ./index")
    print("Or run custom queries: python -m src.app query --question 'your question'")


if __name__ == "__main__":
    demo()
