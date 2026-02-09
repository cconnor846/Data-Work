# Quick Start - 5 Minutes

## Your Data is Already Indexed!

Your 111 lessons from `ll_export_020826.xlsx` are ready to query.

## Try It Now

```bash
cd rag_poc

# Run the demo
python demo_real.py

# Or try interactive mode
python -m src.app interactive --index-dir ./index_real
```

## Example Queries That Work

```bash
# By operation type
python -m src.app query \
  --index-dir ./index_real \
  --question "Structural engineering design problems"

# By keyword
python -m src.app query \
  --index-dir ./index_real \
  --question "Problems with estimating"

# General search
python -m src.app query \
  --index-dir ./index_real \
  --question "Quality control issues"
```

## Reindex New Data

When you get an updated Excel export:

```bash
python -m src.app build \
  --excel /path/to/new_export.xlsx \
  --index-dir ./index_updated
```

## What's Included

- ✅ **111 lessons indexed** from your real data
- ✅ **Working queries** with auto-filtering
- ✅ **Citation tracking** with lesson IDs
- ✅ **3 interfaces**: CLI, interactive, Python API

## Current Status

**Embeddings**: ✅ sentence-transformers (local, FREE, semantic)  
**LLM**: ⚠️ Stub (just lists lesson IDs)

**Next step**: Add OpenAI GPT-4 → See `OPENAI_SETUP.md`

## Upgrade to GPT-4 (Intelligent Answers)

**Current**: sentence-transformers (FREE) + Stub LLM  
**After**: sentence-transformers (FREE) + GPT-4

See `OPENAI_SETUP.md` for complete guide:
1. Install: `pip install openai python-dotenv`
2. Set API key: `export OPENAI_API_KEY="sk-..."`
3. Update generate.py (instructions in OPENAI_SETUP.md)
4. Done!

**Cost**: ~$0.01 per query (embeddings stay FREE)

## Files

```
rag_poc/
├── index_real/          # Your 111 lessons (ready to query)
├── demo_real.py         # Demo script
├── src/                 # All updated for your schema
└── README.md            # Full documentation
```

## Questions?

- See `README.md` for detailed docs
- See `CHANGES.md` for what was updated
- See `ARCHITECTURE.md` for system design
