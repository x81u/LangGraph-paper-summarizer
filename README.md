# LangGraph paper summarizer
A Python tool that summarizes academic papers into well-structured sections in Traditional Chinese for easier reading and reference.
## Installation (with uv)
### 1. Clone the repository
```bash
git clone https://github.com/x81u/LangGraph-paper-summarizer.git
cd LangGraph-paper-summarizer
```
### 2. Create and activate a virtual environment
```bash
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```
### 3. Install dependencies
**Option 1: From `pyproject.toml`**
```bash
uv pip install -e .
```
**Option 2: From `requirements.txt`**
```bash
uv pip install -r requirements.txt
```
### 4. Configure API Key
1. Get a [Google API Key](https://aistudio.google.com/app/apikey)
2. Create a .env file in the project root with the following content:
```env
GOOGLE_API_KEY = your_google_api_key
```
Replace `your_google_api_key` with your actual Google API key.

## Usage
Once installed, you can run the project with:
```bash
uv run main.py
```