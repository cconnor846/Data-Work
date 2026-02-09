# Adding OpenAI GPT-4 for Intelligent Answers

Your system now has **semantic embeddings** (sentence-transformers, local, free).

The final step is adding **OpenAI GPT-4** to generate intelligent answers instead of just listing lesson IDs.

## Current Status

✅ **Embeddings**: sentence-transformers (local, semantic, FREE)  
⚠️ **LLM**: Stub (just lists lesson IDs)

## Goal

✅ **Embeddings**: sentence-transformers (local, semantic, FREE)  
✅ **LLM**: OpenAI GPT-4 (intelligent analysis)

---

## Step 1: Install OpenAI SDK

```bash
pip install openai python-dotenv
```

## Step 2: Get API Key

1. Go to https://platform.openai.com/api-keys
2. Create new API key
3. Copy it (starts with `sk-...`)

## Step 3: Set API Key

### Option A: Environment Variable
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### Option B: .env File (Recommended)
```bash
# Create .env file in project root
echo 'OPENAI_API_KEY=sk-your-key-here' > .env
```

## Step 4: Update generate.py

Replace the OpenAILLMClient class with this working version:

```python
class OpenAILLMClient(LLMClient):
    """
    OpenAI ChatGPT client for answer generation.
    
    Usage:
        import os
        from dotenv import load_dotenv
        load_dotenv()  # Load .env file
        
        llm = OpenAILLMClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo-preview"
        )
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
            model: Model name
                - gpt-4-turbo-preview: Best quality
                - gpt-4: Stable version
                - gpt-3.5-turbo: Cheaper, faster
            max_tokens: Max response length
        """
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """
        Generate response using OpenAI API.
        """
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
```

## Step 5: Update app.py

Find the line that creates the LLM client and change it:

### Before:
```python
llm_client = StubLLMClient()
```

### After:
```python
import os
from dotenv import load_dotenv

# At top of file
load_dotenv()

# When creating LLM client
from .generate import OpenAILLMClient

llm_client = OpenAILLMClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4-turbo-preview"  # or "gpt-3.5-turbo" for cheaper
)
```

## Step 6: Test It!

```bash
# Build index with sentence-transformers embeddings
python -m src.app build \
  --excel /path/to/ll_export_020826.xlsx \
  --index-dir ./index_gpt4

# Query with GPT-4 answers
python -m src.app query \
  --index-dir ./index_gpt4 \
  --question "What patterns do you see in structural engineering mistakes?"
```

---

## Expected Output

### Before (Stub LLM):
```
Answer: Based on 3 retrieved lessons:

The following lessons are relevant: LL-2023-0681, LL-2023-0682, LL-2023-0683

[Note: This is a stub response. Connect a real LLM to see detailed answers.]
```

### After (GPT-4):
```
Answer: Based on analysis of 3 structural engineering lessons, I've identified 
several key patterns:

1. **Coordination Failures** (LL-2023-0681, LL-2023-0682)
   Both lessons show inadequate coordination between structural and other 
   disciplines during the design phase. This led to conflicts discovered late 
   in construction.

2. **Outdated Information** (LL-2023-0682)
   Design proceeded with survey data that was 6+ months old, resulting in 
   significant rework when actual site conditions differed.

3. **Common Root Cause**: Schedule pressure
   All three lessons mention compressed timelines leading teams to skip 
   verification steps that would have caught these issues.

**Recommended Actions**:
- Implement mandatory 48-hour coordination review windows
- Require fresh surveys within 30 days of design start
- Add schedule buffers for cross-discipline verification

Citations: LL-2023-0681, LL-2023-0682, LL-2023-0683
```

---

## Cost Estimate

**Per Query**:
- Embeddings: FREE (runs locally)
- Vector search: FREE (local)
- GPT-4 generation: ~$0.01 per query

**111 Lessons**:
- Index once: FREE (sentence-transformers)
- 100 queries: ~$1.00

---

## Models to Try

### GPT-4 Turbo (Recommended)
```python
model="gpt-4-turbo-preview"
```
- Best quality
- ~$0.01 per query
- Good for analysis

### GPT-3.5 Turbo (Budget)
```python
model="gpt-3.5-turbo"
```
- Good quality
- ~$0.002 per query
- 5x cheaper

### GPT-4 (Stable)
```python
model="gpt-4"
```
- Proven quality
- ~$0.03 per query
- Most expensive

---

## Alternative: Keep It Free

If you want to avoid API costs entirely:

### Option 1: Use Ollama (Free, Local)
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Download model (e.g., Llama 2)
ollama pull llama2

# Use in Python
import requests

def call_ollama(prompt):
    response = requests.post('http://localhost:11434/api/generate', 
        json={
            'model': 'llama2',
            'prompt': prompt,
            'stream': False
        }
    )
    return response.json()['response']
```

**Pros**: Completely free, private  
**Cons**: Slower, lower quality than GPT-4

### Option 2: Use Anthropic Claude
Similar to OpenAI but different pricing:

```python
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-...")
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    messages=[{"role": "user", "content": prompt}]
)
```

---

## Summary

**What you have now**:
- ✅ Real semantic search (sentence-transformers)
- ✅ Smart retrieval (finds relevant lessons)
- ⚠️ Basic output (just lists IDs)

**After adding GPT-4**:
- ✅ Real semantic search (sentence-transformers)
- ✅ Smart retrieval (finds relevant lessons)
- ✅ Intelligent analysis (patterns, recommendations, citations)

**Cost**: ~$0.01 per query (just for GPT-4, embeddings are FREE)

---

## Questions?

- See main README.md for full docs
- Test with gpt-3.5-turbo first (cheaper)
- Can switch models anytime
