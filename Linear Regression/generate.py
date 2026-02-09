"""
Prompt construction and LLM response generation.
Enforces grounding rules and citation requirements.
"""
from typing import List, Dict, Any, Optional


def build_grounded_prompt(question: str, hits: List[Dict[str, Any]]) -> str:
    """
    Build a prompt that enforces grounding in retrieved evidence.
    
    Structure:
    1. System instructions (grounding rules)
    2. Evidence blocks (numbered, with metadata)
    3. Question
    
    Args:
        question: User's question
        hits: Retrieved lessons with metadata and text
        
    Returns:
        Complete prompt string
    """
    if not hits:
        return ""
    
    # Build evidence blocks
    evidence_blocks = []
    for i, hit in enumerate(hits, 1):
        metadata = hit["metadata"]
        text = hit["text"]
        score = hit.get("score", 0)
        
        # Format: [N] metadata | score | text
        block = (
            f"[{i}] Lesson: {metadata.get('lesson_id', 'Unknown')} | "
            f"Discipline: {metadata.get('discipline', 'Unknown')} | "
            f"Phase: {metadata.get('phase', 'Unknown')} | "
            f"Project: {metadata.get('project_type', 'Unknown')} | "
            f"Vendor: {metadata.get('vendor', 'Unknown')} | "
            f"Date: {metadata.get('date', 'Unknown')} | "
            f"Relevance: {score:.2f}\n\n"
            f"{text}"
        )
        evidence_blocks.append(block)
    
    # System instructions
    system_rules = """You are answering questions about construction lessons learned.

CRITICAL RULES:
1. Answer ONLY using information from the EVIDENCE section below
2. If the evidence does not contain enough information, say: "No relevant lessons found for this question."
3. Do NOT invent, assume, or extrapolate beyond what is explicitly stated in the evidence
4. ALWAYS cite lesson IDs when making claims
5. When multiple lessons show a pattern, summarize the pattern and cite all relevant lesson IDs

FORMAT YOUR RESPONSE:
- Start with a direct answer to the question
- Support claims with specific evidence from lessons
- Use format: "According to lesson LESSON_XXXX, [specific detail]"
- Group related lessons when describing patterns
- End with actionable insights if applicable"""
    
    # Assemble full prompt
    prompt = (
        f"{system_rules}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n\n"
        + "\n\n---\n\n".join(evidence_blocks) +
        f"\n\nProvide your answer based only on the evidence above:"
    )
    
    return prompt


def build_system_message() -> str:
    """
    System message for the LLM chat interface.
    Sets the grounding behavior.
    """
    return """You are a construction lessons-learned assistant. You help users learn from past mistakes and successes on construction projects.

Your responses must:
- Only use information from provided evidence
- Cite specific lesson IDs for every claim
- Acknowledge when evidence is insufficient
- Summarize patterns across multiple lessons when relevant
- Be practical and actionable

Do not:
- Invent lessons or details not in evidence
- Make assumptions beyond what is stated
- Provide generic construction advice not grounded in the specific lessons"""


class LLMClient:
    """
    Abstract LLM client for response generation.
    Swappable implementation - use stub for testing, OpenAI for production.
    """
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a response.
        
        Args:
            prompt: User prompt (includes question + evidence)
            system_message: System instructions
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            Generated text
        """
        raise NotImplementedError


class StubLLMClient(LLMClient):
    """
    Stub LLM for testing the pipeline.
    Returns a canned response with citations.
    """
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """Return a basic summary of evidence."""
        # Extract lesson IDs from prompt
        import re
        lesson_ids = re.findall(r'Lesson: (LESSON_\d+)', prompt)
        
        if not lesson_ids:
            return "No relevant lessons found for this question."
        
        # Build simple response
        response = (
            f"Based on {len(lesson_ids)} retrieved lessons:\n\n"
            f"The following lessons are relevant: {', '.join(lesson_ids[:3])}"
        )
        
        if len(lesson_ids) > 3:
            response += f" (and {len(lesson_ids) - 3} more)"
        
        response += "\n\n[Note: This is a stub response. Connect a real LLM to see detailed answers.]"
        
        return response


class OpenAILLMClient(LLMClient):
    """
    OpenAI ChatGPT client.
    
    Usage when you have API key:
        client = OpenAILLMClient(api_key="sk-...")
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        max_tokens: int = 1000
    ):
        """
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4-turbo-preview, gpt-3.5-turbo, etc.)
            max_tokens: Max response length
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        
        # NOTE: Actual implementation
        # from openai import OpenAI
        # self.client = OpenAI(api_key=api_key)
        
        raise NotImplementedError(
            "OpenAI client requires openai package and API key. "
            "Use StubLLMClient for testing."
        )
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """
        Generate using OpenAI API.
        
        Implementation would be:
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
        """
        raise NotImplementedError("See class docstring")


def answer_question(
    question: str,
    retriever,
    llm_client: LLMClient,
    k: int = 5,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    End-to-end question answering.
    
    Flow:
    1. Retrieve relevant lessons
    2. Build grounded prompt
    3. Generate answer with LLM
    4. Package response with citations
    
    Args:
        question: User question
        retriever: Retriever instance
        llm_client: LLM client instance
        k: Number of lessons to retrieve
        verbose: Print debug info
        
    Returns:
        Dict with answer, citations, hits, and metadata
    """
    # Retrieve
    if verbose:
        print(f"\nRetrieving top-{k} lessons for: {question}")
    
    hits = retriever.retrieve(query=question, k=k)
    
    if verbose:
        print(f"Found {len(hits)} relevant lessons")
        if hits:
            print(f"Top score: {hits[0]['score']:.3f}")
    
    # Handle no results
    if not hits:
        return {
            "question": question,
            "answer": "No relevant lessons found for this question.",
            "citations": [],
            "hits": [],
            "num_hits": 0,
        }
    
    # Build prompt
    prompt = build_grounded_prompt(question, hits)
    system_msg = build_system_message()
    
    if verbose:
        print(f"\nPrompt length: {len(prompt)} characters")
    
    # Generate answer
    answer = llm_client.generate(
        prompt=prompt,
        system_message=system_msg,
        temperature=0.3
    )
    
    # Extract citations
    citations = [h["metadata"]["lesson_id"] for h in hits]
    
    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "hits": hits,
        "num_hits": len(hits),
    }
