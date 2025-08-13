# LangGraph paper summarizer
A Python tool that summarizes academic papers into well-structured sections in Traditional Chinese for easier reading and reference.
## Installation (with uv)
### 1. Clone the ropository
```
git clone https://github.com/x81u/LangGraph-paper-summarizer.git
cd LangGraph-paper-summarizer
```
### 2. Create and activate a virtual environment
```
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```
### 3. Install dependencies
From pyproject.toml
```
uv pip install -e .
```
or requirements.txt
```
uv pip install -r requirements.txt
```

## Usage
Once installed, you can run the project with:
```
uv run main.py
```