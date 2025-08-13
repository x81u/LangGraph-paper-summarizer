import os
import arxiv
import fitz
import time
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini small & large models
small_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
large_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=GOOGLE_API_KEY)

# Define State
class PaperState(MessagesState):
    papers: list = []
    filtered_papers: list = []
    summaries: list = []
    markdown: str = ""

# Node: Search Arxiv
def search_arxiv(state: PaperState) -> PaperState:
    query = input("Enter your research keyword: ")
    if not query:
        print("No keyword entered. Ending...")
        exit(0)
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=20,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "pdf_url": result.pdf_url,
            "published": result.published.strftime("%Y-%m-%d")
        })
    return {"papers": papers, "messages": state["messages"]}

# Node: Filter by title
def filter_by_title(state: PaperState) -> PaperState:
    filtered_by_title = []
    print("\n=== Top 20 Search Results (Titles) ===\n")
    for idx, paper in enumerate(state["papers"], 1):
        print(f"[{idx}] {paper['title']} ({paper['published']})")

    selection = input("\nEnter the paper numbers you want to keep (comma-separated), or return to search without input: ").strip()
    if not selection:
        print("No papers selected. Return to search.")
        return {"filtered_papers": [], "messages": state["messages"]}
    try:
        selected_indices = {int(i.strip()) for i in selection.split(",")}
    except ValueError:
        print("Invalid input. Return to search.")
        return {"filtered_papers": [], "messages": state["messages"]}

    for idx, paper in enumerate(state["papers"], 1):
        if idx in selected_indices:
            filtered_by_title.append(paper)
    return {"filtered_papers": filtered_by_title, "messages": state["messages"]}

# Node: Filter papers with user selection
def filter_papers(state: PaperState) -> PaperState:
    filtered = []
    print("\n=== Search Results ===\n")
    for idx, paper in enumerate(state["filtered_papers"], 1):
        # Translate abstract to Traditional Chinese
        prompt = f"Translate the following academic abstract into Traditional Chinese:\n\n{paper['abstract']}"
        res = small_model.invoke([HumanMessage(content=prompt)])
        zh_abstract = res.content.strip()
        print(f"[{idx}] {paper['title']}")
        print(f"Abstract:\n{zh_abstract}\n{'-'*60}")

    selection = input("Enter the paper numbers you want to keep (comma-separated), or return to select title without input: ").strip()
    if not selection:
        print("No papers selected. Return to title selection.")
        return {"filtered_papers": [], "messages": state["messages"]}
    try:
        selected_indices = {int(i.strip()) for i in selection.split(",")}
    except ValueError:
        print("Invalid input. Return to title selection.")
        return {"filtered_papers": [], "messages": state["messages"]}

    for idx, paper in enumerate(state["filtered_papers"], 1):
        if idx in selected_indices:
            filtered.append(paper)
    return {"filtered_papers": filtered, "messages": state["messages"]}

# Node: Download PDFs & extract text
def parse_pdfs(state: PaperState) -> PaperState:
    for paper in state["filtered_papers"]:
        os.makedirs("downloads", exist_ok=True)
        pdf_path = os.path.join("downloads", paper["title"].replace(" ", "_") + ".pdf")
        os.system(f"wget -q '{paper['pdf_url']}' -O '{pdf_path}'")
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        paper["full_text"] = text
    return {"filtered_papers": state["filtered_papers"], "messages": state["messages"]}

# Node: Summarize each paper into sections using large model
def summarize_sections(state: PaperState) -> PaperState:
    summaries = []
    for idx, paper in enumerate(state["filtered_papers"], 1):
        print(f"Processing [{idx}] {paper['title']}")
        prompt = f"""
Please summarize the following research paper into sections.
Each section should have a clear Markdown heading and a concise explanation.

Output language: Traditional Chinese.

Only return plain text with Markdown formatting.

Full text:
{paper['full_text']}
"""
        retry_count = 0
        while retry_count < 4:
            res = large_model.invoke([HumanMessage(content=prompt)])
            res_content = res.content.strip() if res.content else ""
            if res_content:
                break
            else:
                retry_count += 1
                if retry_count < 4:
                    print(f"AI reply nothing. Retry...")
                    time.sleep(5)
        if not res_content:
            res_content = "Something wrong, try again later."
        summaries.append({"title": paper["title"], "summary": res_content})
    return {"summaries": summaries, "messages": state["messages"]}

# Node: Generate Markdown report
def generate_markdown(state: PaperState) -> PaperState:
    md_content = "# Research Paper Summaries\n\n"
    for s in state["summaries"]:
        md_content += f"## {s['title']}\n\n{s['summary']}\n\n"
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    report_file_name = "report"
    counter = 1
    filename = report_file_name
    while os.path.exists(os.path.join(output_dir, filename+".md")):
        filename = f"{report_file_name}_{counter}"
        counter += 1
    with open(os.path.join(output_dir, filename+".md"), "w", encoding="utf-8") as f:
        f.write(md_content)
    return {"markdown": md_content, "messages": state["messages"]}

def check_paper_selection(state):
    return "no_selection" if not state.get("filtered_papers") else "selected"

# Build LangGraph
graph = StateGraph(PaperState)
graph.add_node("search", search_arxiv)
graph.add_node("filter_title", filter_by_title)
graph.add_node("filter", filter_papers)
graph.add_node("parse", parse_pdfs)
graph.add_node("summarize", summarize_sections)
graph.add_node("output", generate_markdown)
# search -> filter_title -> filter -> parse -> summarize -> output
graph.set_entry_point("search")
graph.add_edge("search", "filter_title")
graph.add_conditional_edges(
    "filter_title",
    check_paper_selection,
    {
        "no_selection": "search",
        "selected": "filter"
    }
)
graph.add_conditional_edges(
    "filter",
    check_paper_selection,
    {
        "no_selection": "filter_title",
        "selected": "parse"
    }
)
graph.add_edge("parse", "summarize")
graph.add_edge("summarize", "output")

app = graph.compile()

# Run
if __name__ == "__main__":
    result = app.invoke({})
    print("Done! Generated report.")
