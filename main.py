import os
import arxiv
import fitz
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
    query = state["messages"][-1].content
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
            "pdf_url": result.pdf_url
        })
    return {"papers": papers, "messages": state["messages"]}

# Node: Filter by title
def filter_by_title(state: PaperState) -> PaperState:
    filtered_by_title = []
    print("\n=== Top 20 Search Results (Titles) ===\n")
    for idx, paper in enumerate(state["papers"], 1):
        print(f"[{idx}] {paper['title']}")
    selection = input("\nEnter the paper numbers you want to keep (comma-separated), or press Enter to skip: ").strip()
    if not selection:
        print("No papers selected. Exiting...")
        exit(0)
    try:
        selected_indices = {int(i.strip()) for i in selection.split(",")}
    except ValueError:
        print("Invalid input. Exiting...")
        exit(1)
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

        # Show to user
        print(f"[{idx}] {paper['title']}")
        print(f"Abstract:\n{zh_abstract}\n{'-'*60}")

    # Ask user to select papers
    selection = input("Enter the paper numbers you want to keep (comma-separated), or press Enter to skip: ").strip()
    if not selection:
        print("No papers selected. Exiting...")
        exit(0)

    try:
        selected_indices = {int(i.strip()) for i in selection.split(",")}
    except ValueError:
        print("Invalid input. Exiting...")
        exit(1)

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

## Node: Summarize each paper into sections using large model
def summarize_sections(state: PaperState) -> PaperState:
    summaries = []
    for paper in state["filtered_papers"]:
        prompt = f"""
Please summarize the following research paper into sections.
Each section should have a clear Markdown heading and a concise explanation.

Output language: Traditional Chinese.

Full text:
{paper['full_text']}
"""
        res = large_model.invoke([HumanMessage(content=prompt)])
        summaries.append({"title": paper["title"], "summary": res.content})
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
graph.add_edge("filter_title", 'filter')
graph.add_edge("filter", "parse")
graph.add_edge("parse", "summarize")
graph.add_edge("summarize", "output")

app = graph.compile()

# Run
if __name__ == "__main__":
    query = input("Enter your research keyword: ")
    result = app.invoke({"messages": [HumanMessage(content=query)]})
    print("Done! Generated report.")
