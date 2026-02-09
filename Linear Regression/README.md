# Construction Lessons-Learned RAG System

**Status**: ✅ Semantic search working with 111 real lessons

## What's Working Now

### ✅ Semantic Embeddings (sentence-transformers)
- **FREE** - Runs locally, no API costs
- **SEMANTIC** - Understands meaning, not just keywords
- "structural issues" matches "building framework errors"
- 384-dimensional vectors for precise matching

### ✅ Smart Retrieval
- Vector similarity search
- Auto-filtering by operations & business area
- Relevance scoring
- Citation tracking

### ⚠️ Stub LLM (Basic Output)
Currently just lists lesson IDs. 

**Next step**: Add OpenAI GPT-4 for intelligent analysis → See `OPENAI_SETUP.md`

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build index with YOUR data (already done for you!)
# Your 111 lessons are in ./index_real/

# 3. Try the demo
python demo_real.py

# 4. Interactive queries
python -m src.app interactive --index-dir ./index_real
```

---

## What's Included

### Your Data (Already Indexed)
- **111 lessons** from ll_export_020826.xlsx
- **Location**: `./index_real/`
- **Ready**: Query immediately

### Working Features
1. **Semantic Search** - Finds similar meaning
2. **Auto-Filtering** - Detects operations/business area from queries
3. **Citation Tracking** - Returns lesson IDs (LL-2023-0682, etc.)
4. **3 Interfaces**: Single query, interactive mode, Python API

---

## Your Excel Schema

**Required Columns**:
- `IDENTIFIER` → lesson_id
- `TITLE` → Brief summary
- `DESCRIPTION` → What happened
- `OPERATIONS INVOLVED` → Discipline (Structural, Electrical, etc.)
- `BUSINESS AREA INVOLVED` → Area (Engineering, Construction, etc.)
- `PROJECT` → Project name

**Optional Columns**:
- `ACTIONS TAKEN`, `PROPOSED SOLUTION`, `EVENT DATE`, `IDENTIFIED BY`

---

## Example Queries

### By Operation
```bash
python -m src.app query \
  --index-dir ./index_real \
  --question "Structural engineering design problems"

# Auto-filters to operations=Structural, business_area=Engineering
# Returns: LL-2023-0681, LL-2023-0692, LL-2018-1512
```

### By Keyword
```bash
python -m src.app query \
  --index-dir ./index_real \
  --question "Issues with quality control during construction"

# Semantic search finds relevant lessons
```

### Interactive
```bash
python -m src.app interactive --index-dir ./index_real

> Problems with estimating
> Electrical design errors  
> What went wrong on the Yellowbud Solar project?
```

---

## How It Works

### Current Pipeline

```
Your Excel (333 rows)
    ↓ [Load & Validate]
111 Complete Lessons
    ↓ [sentence-transformers - FREE, LOCAL]
Semantic Vectors (384 dim)
    ↓ [Vector Index]
Searchable Index
    ↓ [User Query]
"structural issues"
    ↓ [sentence-transformers - FREE, LOCAL]
Query Vector
    ↓ [Similarity Search]
Top 3 Relevant Lessons
    ↓ [Stub LLM]
"Found: LL-2023-0681, LL-2023-0682, LL-2023-0683"
```

### After Adding GPT-4

```
    ↓ [Top 3 Relevant Lessons]
    ↓ [OpenAI GPT-4]
"Based on these lessons, structural issues stem from:
1. Inadequate coordination (LL-2023-0681)
2. Outdated survey data (LL-2023-0682)  
3. Design-build conflicts (LL-2023-0683)
Pattern: Most occur during design phase due to schedule pressure..."
```

---

## Reindex New Data

When you get an updated Excel export:

```bash
python -m src.app build \
  --excel /path/to/new_export.xlsx \
  --index-dir ./index_updated

# Then query the new index
python -m src.app query \
  --index-dir ./index_updated \
  --question "Your question"
```

---

## Add OpenAI GPT-4 (Intelligent Answers)

**Current**: sentence-transformers (FREE) + Stub LLM (basic)  
**Goal**: sentence-transformers (FREE) + GPT-4 (intelligent)

**See `OPENAI_SETUP.md` for complete guide**

Quick version:
```bash
# 1. Install
pip install openai python-dotenv

# 2. Set API key
export OPENAI_API_KEY="sk-..."

# 3. Update generate.py (see OPENAI_SETUP.md)

# 4. Query
python -m src.app query \
  --question "What patterns do you see in structural mistakes?"
```

**Cost**: ~$0.01 per query (embeddings are FREE)

---

## Architecture

### Components
1. **Ingestion** (`src/ingest.py`) - Load Excel, validate
2. **Models** (`src/models.py`) - Lesson data structure  
3. **Embeddings** (`src/embeddings.py`) - sentence-transformers (local, free)
4. **Index** (`src/simple_index.py`) - Vector storage & search
5. **Retrieval** (`src/retrieve.py`) - Search + filtering
6. **Generation** (`src/generate.py`) - LLM prompting

### Design Principles
- ✅ **Local First** - Embeddings run on your machine (free)
- ✅ **Swappable** - Easy to change LLM or embeddings
- ✅ **Explainable** - Clear filtering, scoring, citations
- ✅ **Production-Ready** - Validation, error handling, logging

---

## Data Quality

From your ll_export_020826.xlsx:
- **Total rows**: 333
- **Complete**: 111 (33%)
- **Skipped**: 222 (missing required fields)

**Most Common Operations**:
- Structural Engineering & Design (25 lessons)
- Estimating (12 lessons)
- Civil Engineering & Design (8 lessons)

**Business Areas**:
- Engineering (102 lessons)
- Construction (28 lessons)

**Tip**: For better coverage, ensure OPERATIONS INVOLVED, TITLE, and DESCRIPTION are populated before export.

---

## Cost Analysis

### Current (sentence-transformers + Stub LLM)
- **Indexing**: FREE (runs locally)
- **Queries**: FREE (runs locally)
- **Total**: $0

### With GPT-4
- **Indexing**: FREE (sentence-transformers)
- **Queries**: ~$0.01 each (just GPT-4)
- **100 queries**: ~$1

### With GPT-3.5 Turbo (Budget)
- **Indexing**: FREE (sentence-transformers)
- **Queries**: ~$0.002 each
- **500 queries**: ~$1

---

## Files

```
rag_poc/
├── index_real/          # Your 111 lessons (ready to query)
├── src/                 # All source code
│   ├── models.py        # Lesson schema
│   ├── ingest.py        # Excel loading
│   ├── embeddings.py    # sentence-transformers
│   ├── simple_index.py  # Vector search
│   ├── retrieve.py      # Filtering
│   ├── generate.py      # LLM (stub + OpenAI)
│   └── app.py           # CLI
├── demo_real.py         # Demo script
├── README.md            # This file
├── OPENAI_SETUP.md      # How to add GPT-4
├── QUICKSTART.md        # 5-minute guide
├── CHANGES.md           # What was updated
└── requirements.txt     # Dependencies
```

---

## Next Steps

1. ✅ **Test semantic search** - Run `python demo_real.py`
2. ✅ **Try your own queries** - Use interactive mode
3. 🎯 **Add GPT-4** - See `OPENAI_SETUP.md`
4. 📊 **Scale up** - Index new exports as they come
5. 🎨 **Build UI** - Add Streamlit/Gradio (optional)

---

## Support

- **Setup**: See `QUICKSTART.md`
- **OpenAI**: See `OPENAI_SETUP.md`  
- **Architecture**: See `ARCHITECTURE.md`
- **Changes**: See `CHANGES.md`

---

## License

MIT - Use freely for your construction projects!
