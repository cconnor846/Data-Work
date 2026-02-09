# RAG System Architecture

## Overview

This is a production-oriented RAG (Retrieval-Augmented Generation) system for querying construction lessons learned. The architecture follows clean separation of concerns and uses swappable components.

## System Components

### 1. Data Model (`src/models.py`)

**Purpose**: Define the structure of a lesson and canonical text representation

**Key Classes**:
- `Lesson`: Immutable dataclass with all lesson fields
- `lesson_to_canonical_text()`: Converts lesson to searchable text

**Design Decisions**:
- Immutable (`frozen=True`) to prevent accidental modification
- Separate metadata from content for flexible indexing
- Canonical text format includes metadata for better context

### 2. Data Ingestion (`src/ingest.py`)

**Purpose**: Load and validate lessons from Excel files

**Flow**:
```
Excel File → pandas DataFrame → Validation → List[Lesson]
```

**Key Functions**:
- `load_lessons_from_excel()`: Main entry point
- `validate_lessons()`: Check for duplicates and empty fields
- `_row_to_lesson()`: Convert DataFrame row to Lesson object

**Validation Rules**:
- Required fields must be non-empty
- Lesson IDs must be unique
- Handles missing optional fields gracefully

### 3. Embeddings (`src/embeddings.py`)

**Purpose**: Convert text to vector representations

**Abstraction**:
```python
class EmbeddingClient(ABC):
    def embed_texts(self, texts: List[str]) -> np.ndarray
    
    @property
    def embedding_dim(self) -> int
```

**Implementations**:
- `StubEmbeddingClient`: Hash-based for testing (NO production use)
- `OpenAIEmbeddingClient`: Template for OpenAI integration

**Why Abstraction**:
- Easy to swap providers (OpenAI → Cohere → local models)
- Testable without API keys
- Consistent interface

### 4. Vector Index (`src/index.py`, `src/simple_index.py`)

**Purpose**: Store and search embeddings efficiently

**Two Implementations**:

**FAISSVectorIndex** (production):
- Uses Meta's FAISS library
- Exact cosine similarity via IndexFlatIP
- Scales to millions of documents
- Persistent storage

**SimpleVectorIndex** (POC):
- Pure numpy implementation
- Good for <10k documents
- No external dependencies
- Same interface as FAISS

**Storage Format**:
```
index/
├── index.pkl         # vectors + metadata (SimpleVectorIndex)
└── faiss.index       # FAISS binary (FAISSVectorIndex)
    metadata.pkl
```

**Key Methods**:
- `add()`: Add vectors with metadata
- `search()`: Cosine similarity search with threshold
- `save()`/`load()`: Persistence

### 5. Retrieval (`src/retrieve.py`)

**Purpose**: Orchestrate search with filtering

**Flow**:
```
Query → Auto-filter extraction → Embed → Vector search → 
Filter results → Apply threshold → Return hits
```

**Filtering Strategy**:
- Keyword-based (deterministic, explainable)
- Metadata fields: discipline, phase, project_type, vendor
- Case-insensitive matching

**Configurable Parameters**:
- `k`: Number of results (default 5)
- `score_threshold`: Minimum similarity (default 0.05 for stub, 0.15 for real)
- `auto_filter`: Enable/disable automatic filter extraction

**Example**:
```python
retriever.retrieve(
    query="electrical issues during construction",
    k=5
)
# Auto-detects: discipline=Electrical, phase=Construction
```

### 6. Response Generation (`src/generate.py`)

**Purpose**: Build prompts and generate grounded answers

**Prompt Structure**:
```
System Message: Grounding rules
↓
Evidence: [Lesson 1] [Lesson 2] ...
↓
Question: User's question
↓
Instruction: Answer only from evidence
```

**Grounding Rules**:
1. Answer ONLY from provided evidence
2. Cite lesson IDs for every claim
3. Say "no relevant lessons" if evidence insufficient
4. Summarize patterns when multiple lessons match
5. Do not invent or extrapolate

**LLM Abstraction**:
```python
class LLMClient(ABC):
    def generate(
        self, 
        prompt: str, 
        system_message: str,
        temperature: float
    ) -> str
```

**Why Grounded Prompts**:
- Prevents hallucination
- Ensures traceability
- Enables citation verification
- Production-safe behavior

### 7. Application (`src/app.py`)

**Purpose**: CLI interface for all operations

**Commands**:

**Build**:
```bash
python -m src.app build --excel data.xlsx --index-dir ./index
```
Creates vector index from Excel file

**Query**:
```bash
python -m src.app query --question "..." --index-dir ./index
```
Single query with full output

**Interactive**:
```bash
python -m src.app interactive --index-dir ./index
```
REPL-style querying

## Data Flow

### Indexing Flow

```
Excel File
  ↓ (pandas)
DataFrame
  ↓ (validation)
List[Lesson]
  ↓ (canonical_text)
List[str]
  ↓ (embedder)
np.ndarray (vectors)
  ↓ (index.add)
VectorIndex
  ↓ (save)
Disk
```

### Query Flow

```
User Question
  ↓ (filter extraction)
Query + Filters
  ↓ (embedder)
Query Vector
  ↓ (index.search)
Top-K Hits
  ↓ (filter + threshold)
Relevant Hits
  ↓ (build_prompt)
Grounded Prompt
  ↓ (LLM)
Answer + Citations
```

## Key Design Principles

### 1. Separation of Concerns
Each module has a single responsibility:
- `ingest`: Load data
- `embeddings`: Create vectors
- `index`: Store/search vectors
- `retrieve`: Orchestrate search
- `generate`: Create answers

### 2. Abstraction for Swappability
All external dependencies behind interfaces:
- `EmbeddingClient`: Swap OpenAI ↔ Cohere ↔ local
- `LLMClient`: Swap GPT-4 ↔ Claude ↔ local
- `VectorIndex`: Swap FAISS ↔ ChromaDB ↔ Weaviate

### 3. Explainability First
- Deterministic filtering (no NLP black box)
- Score thresholds (visible cutoffs)
- Citation requirements (traceable claims)
- Verbose logging (debuggable behavior)

### 4. Production Patterns
- Validation at ingestion (fail fast)
- Persistent storage (reproducible)
- Error handling (graceful degradation)
- Type hints (self-documenting)

## Scalability Considerations

### Current Limits (POC)
- Documents: ~10k (SimpleVectorIndex in memory)
- Query latency: <100ms
- Index size: ~150MB (10k docs × 384 dim × 4 bytes)

### Production Scale
With FAISS + proper infrastructure:
- Documents: 1M+ (FAISS supports billions)
- Query latency: <10ms (FAISS is extremely fast)
- Index size: Linear in documents × dimensions

### Bottlenecks
1. **Embedding API**: Rate limits, cost
   - Solution: Batch processing, caching
2. **LLM API**: Latency, cost
   - Solution: Streaming, caching common queries
3. **Metadata filtering**: O(n) in Python
   - Solution: Database-backed metadata store

## Extension Points

### Adding New Filters
Edit `retrieve.py`:
```python
# Add to constants
VENDORS = {"abc_electric", "cool_air", ...}

# Add to extract_filters_from_query
for vendor in VENDORS:
    if vendor in query_lower:
        filters["vendor"] = vendor
```

### Adding Chunking
Edit `ingest.py`:
```python
def chunk_lesson(lesson: Lesson) -> List[str]:
    # Split long lessons into chunks
    # Each chunk becomes separate document
    pass
```

### Adding Reranking
Edit `retrieve.py`:
```python
def retrieve_with_rerank(self, query, k):
    # 1. Get k×3 candidates from vector search
    # 2. Rerank with cross-encoder
    # 3. Return top-k
    pass
```

## Cost Analysis

**Per Query** (with real embeddings/LLM):
- Embedding: 1 query × $0.0001 = $0.0001
- Vector search: Free (local)
- LLM generation: ~500 tokens × $0.01/1k = $0.005
- **Total: ~$0.005 per query**

**Per 10k lessons indexed**:
- Embeddings: 10k × $0.0001 = $1.00
- Storage: ~150MB S3 = $0.004/month
- **Total: $1 one-time + $0.004/month**

## Testing Strategy

### Unit Tests (TODO)
- `test_ingest.py`: Excel loading, validation
- `test_embeddings.py`: Vector generation
- `test_index.py`: Add/search operations
- `test_retrieve.py`: Filtering logic
- `test_generate.py`: Prompt construction

### Integration Tests
- End-to-end: Excel → Index → Query → Answer
- Performance: Query latency, index build time
- Accuracy: Retrieval relevance, grounding compliance

### Manual Testing
- Use `demo.py` for smoke tests
- Use interactive mode for exploratory testing
- Compare results against known lessons

## Monitoring (Production)

**Metrics to Track**:
- Query latency (p50, p95, p99)
- Retrieval hit rate (% queries with results)
- LLM grounding rate (% answers citing evidence)
- Cost per query
- Top queries (for index optimization)

**Alerts**:
- Latency > 1s
- Hit rate < 80%
- Error rate > 1%
- Daily cost > budget

## Next Steps

### Short-term (Production-ready)
1. Connect OpenAI embeddings + GPT-4
2. Add unit tests (pytest)
3. Calibrate score thresholds on real data
4. Add query logging
5. Build Streamlit UI

### Medium-term (Scale)
1. Add database for metadata
2. Implement hybrid search (keyword + semantic)
3. Add reranking
4. Support multi-tenancy
5. Build analytics dashboard

### Long-term (Advanced)
1. Fine-tune embeddings on domain
2. Add feedback loop (user ratings)
3. Implement caching layer
4. Multi-modal support (images, PDFs)
5. Real-time index updates
