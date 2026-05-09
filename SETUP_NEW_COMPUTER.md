# Setup on New Computer

## Files to Copy

### Essential (3 files)
```
langgraph_agent_boilerplate.py  ← Main code
requirements.txt                ← Dependencies
README.md                       ← Documentation
```

### Recommended (6 files total)
```
langgraph_agent_boilerplate.py
requirements.txt
README.md
example.py                      ← Working examples
test_langgraph_agent.py         ← Unit tests
QUICK_REFERENCE.md              ← Cheat sheet
```

## Quick Setup

### 1. Copy Files

**Option A: Manual Copy**
- Copy the 6 files above to new computer

**Option B: Git (Recommended)**
```bash
# On this computer (one time):
git init
git add langgraph_agent_boilerplate.py requirements.txt README.md example.py test_langgraph_agent.py QUICK_REFERENCE.md .gitignore
git commit -m "Initial commit"
git push origin main

# On new computer:
git clone <your-repo-url>
cd <repo-name>
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install packages
pip install -r requirements.txt
# or
uv pip install -r requirements.txt
```

### 3. Set API Key

```bash
# Required
export ANTHROPIC_API_KEY="sk-ant-api03-YOUR-KEY-HERE"

# Optional (for observability)
export LANGSMITH_API_KEY="your-langsmith-key"
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="interview-demo"
```

### 4. Test It Works

```bash
# Run example
python example.py

# Run tests
pytest test_langgraph_agent.py -v
```

## Files NOT to Copy

These will be recreated automatically:
- ✗ `.venv/` - Virtual environment (recreate on new machine)
- ✗ `__pycache__/` - Python cache
- ✗ `agent_graph.png` - Generated file
- ✗ `.pytest_cache/` - Test cache
- ✗ `test_langsmith.py` - Just a test file
- ✗ `cleanup.sh` - One-time cleanup script
- ✗ `visualize_graph.py` - Optional, can copy if you want visualizations

## Minimal Portable Package

For interviews or demos, you can zip just these 3 files:
```bash
zip langgraph-boilerplate.zip \
    langgraph_agent_boilerplate.py \
    requirements.txt \
    README.md
```

Then on new computer:
```bash
unzip langgraph-boilerplate.zip
pip install -r requirements.txt
# Ready to use!
```

## Git Setup (Recommended)

### Initial Setup
```bash
# Initialize repo
git init

# Add files
git add .

# First commit
git commit -m "LangGraph agent boilerplate"

# Push to GitHub/GitLab
git remote add origin <your-repo-url>
git push -u origin main
```

### On New Computer
```bash
# Clone repo
git clone <your-repo-url>
cd <repo-name>

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Add API key
export ANTHROPIC_API_KEY="your-key"

# Done!
python example.py
```

## Environment Variables File (Optional)

Create `.env` file (never commit this!):
```bash
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE
LANGSMITH_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=interview-demo
```

Load it with:
```bash
source .env  # or use python-dotenv
```

## Checklist for New Setup

- [ ] Copy 6 essential files
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Set ANTHROPIC_API_KEY
- [ ] Run `python example.py` (should work!)
- [ ] Run `pytest test_langgraph_agent.py -v` (should pass!)
- [ ] (Optional) Set LangSmith keys
- [ ] Ready for interview! 🚀
