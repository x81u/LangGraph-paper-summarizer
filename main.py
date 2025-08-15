import os
import arxiv
import fitz
import time
from dotenv import load_dotenv
from typing import List, Dict, Any, Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, BaseMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. 環境設定與模型初始化 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
agent_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_key=GOOGLE_API_KEY)
large_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=GOOGLE_API_KEY)


# --- 2. 共享的 Agent 狀態定義 ---
class PaperState(TypedDict):
    messages: Annotated[list, add_messages]
    papers: List[Dict[str, Any]]
    filtered_papers: List[Dict[str, Any]]
    summaries: List[Dict[str, Any]]
    markdown: str

# ==============================================================================
# Agent 1: Research Assistant (研究助理)
# ==============================================================================

# --- 3A. Agent 1 的工具 ---
@tool
def search_arxiv_tool(query: str) -> List[Dict[str, Any]]:
    """Searches arXiv for papers based on a query."""
    print(f"Tool (Agent 1): Searching arXiv for '{query}'...")
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=10, sort_by=arxiv.SortCriterion.Relevance)
    return [{
        "title": r.title, "abstract": r.summary, "pdf_url": r.pdf_url,
        "published": r.published.strftime("%Y-%m-%d")
    } for r in client.results(search)]

@tool
def translate_abstracts_tool(selected_indices: List[int], all_papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Selects papers from a list based on indices, then translates their abstracts.
    Args:
        selected_indices: A list of 1-based numbers for the papers to be translated.
        all_papers: The full list of papers that was originally presented to the user.
    Returns:
        A list of the selected papers, each with a new 'zh_abstract' key.
    """
    print(f"Tool (Agent 1): Translating abstracts for indices: {selected_indices}")
    if not all_papers:
        return []

    # 1. Select papers based on indices
    selected_papers = []
    for i in selected_indices:
        if 1 <= i <= len(all_papers):
            selected_papers.append(all_papers[i-1])

    # 2. Translate their abstracts
    for paper in selected_papers:
        if 'zh_abstract' not in paper: # Avoid re-translating
            prompt = f"Translate the following academic abstract into Traditional Chinese:\n\n{paper['abstract']}"
            res = agent_model.invoke(prompt)
            paper['zh_abstract'] = res.content.strip()
            
    return selected_papers

@tool
def confirm_final_selection_tool(selected_indices: List[int], all_papers: List[Dict[str, Any]]) -> str:
    """
    Confirms the user's final paper selection based on the provided indices.
    This tool should be called when the user has made their final decision.
    Args:
        selected_indices: A list of 1-based numbers corresponding to the papers the user wants.
        all_papers: The full list of papers that was originally presented to the user.
    """
    print(f"Tool (Agent 1): Confirming final selection for indices: {selected_indices}")
    if not all_papers:
        return "Error: No papers were found to select from."
    
    final_selection = []
    for i in selected_indices:
        if 1 <= i <= len(all_papers):
            final_selection.append(all_papers[i-1]) # Adjust for 0-based index

    # This is a bit of a trick: we return the selection as a string for the AI,
    # but the real update will happen in the tool_node which has access to the state.
    return f"Confirmed selection of {len(final_selection)} papers. Ready for processing."


# --- 4A. Agent 1 的核心邏輯 ---
research_system_prompt = """
You are a friendly and helpful research assistant. Your goal is to guide a user to find and select academic papers from arXiv. The user may want to refine their selection multiple times, so do not rush to the final step.

Your workflow is as follows:
1.  **Search**: Greet the user, ask for a topic, and use `search_arxiv_tool` to find papers.
2.  **Present Titles**: Show the user the numbered list of paper titles. Ask them to make a **preliminary selection** by number.
3.  **Offer Help**: After the user provides numbers, you have two options:
    - **Offer Translation**: Ask the user "您是否需要我為您翻譯這些論文的摘要，來幫助您做最終決定？" (Do you need me to translate the abstracts to help you make a final decision?). If they say yes, use the `translate_abstracts_tool`.
    - **Offer Confirmation**: If the user seems confident, you can ask them if this is their final choice.
4.  **Present Translated Abstracts**: If you used the translation tool, present the translated abstracts to the user. Then, ask them to make their **final selection** based on these abstracts. They might choose a subset of what you translated.
5.  **Final Confirmation**: When you are certain the user has made their **final, unchangeable choice**, you MUST call the `confirm_final_selection_tool`. This tool is the **very last step** of your conversation.
6.  **End Conversation**: After calling `confirm_final_selection_tool`, and only after that, end the conversation by outputting the special string `__END_OF_CONVERSATION__`.

**Crucial Rules:**
- **Memory Rule:** When the user agrees to translation (e.g., says "yes" or "好"), you MUST remember the paper numbers they selected in their PREVIOUS message and use them for the `selected_indices` parameter of the `translate_abstracts_tool`.
- **Example Interaction:**
  - User: "1,2,3"
  - You: "您是否需要我為您翻譯這些論文的摘要...?"
  - User: "好"
  - You: (MUST immediately call `translate_abstracts_tool` with `selected_indices=[1, 2, 3]`)
- **Recovery Rule:** If for any reason you are unsure which numbers to use, you MUST ask the user to confirm them again. Do not just wait silently. For example, say "好的，請問您是要我翻譯第 1, 2, 3 篇的摘要嗎？"
- **Do NOT call `confirm_final_selection_tool` too early.** Call it only when the user explicitly confirms their FINAL list of papers, especially after reviewing the abstracts if they requested them.
- The user's first selection of numbers is a **preliminary** choice, not the final one. Your job is to help them refine it.
- Always communicate in Traditional Chinese.
"""
research_agent_model_with_tools = agent_model.bind_tools([
    search_arxiv_tool, translate_abstracts_tool, confirm_final_selection_tool
])

def research_agent_node(state: PaperState) -> dict:
    print("Node (Agent 1): Research Agent")
    response = research_agent_model_with_tools.invoke([SystemMessage(content=research_system_prompt)] + state['messages'])
    return {"messages": [response]}

def research_tool_node(state: PaperState) -> dict:
    print("Node (Agent 1): Tool Executor")
    tool_calls = state["messages"][-1].tool_calls
    if not tool_calls:
        return {}
    
    tool_messages = []
    updates = {}
    papers_from_search = None
    final_selection = None

    for call in tool_calls:
        print(f"  - Executing tool: {call['name']}")
        tool_output = "Tool not found" # Default value

        if call['name'] == 'search_arxiv_tool':
            tool_output = search_arxiv_tool.invoke(call['args'])
            papers_from_search = tool_output
        
        elif call['name'] == 'translate_abstracts_tool':
            # AI is expected to provide the indices based on the conversation
            raw_indices = call['args'].get('selected_indices', [])
            selected_indices_int = [int(i) for i in raw_indices]
            
            # Prepare arguments for our newly designed tool
            args_for_tool = {
                "selected_indices": selected_indices_int,
                "all_papers": state['papers']
            }
            tool_output = translate_abstracts_tool.invoke(args_for_tool)
            # The result of this tool is our new list of filtered papers
            # which will be presented to the user for final confirmation.
            updates["filtered_papers"] = tool_output
        
        elif call['name'] == 'confirm_final_selection_tool':
            # 1. 取得從 AI 來的原始索引列表 (可能是 floats)
            raw_indices = call['args'].get('selected_indices', [])
            
            # 2. 建立一個新的、保證是整數的列表
            selected_indices_int = []
            for idx in raw_indices:
                try:
                    selected_indices_int.append(int(idx))
                except (ValueError, TypeError):
                    # 如果 AI 傳來了無法轉換的內容 (例如文字)，就忽略它
                    print(f"Warning: Could not convert index '{idx}' to an integer. Skipping.")
            
            # 3. 更新 call['args']，讓後續的工具呼叫也使用乾淨的整數列表
            call['args']['selected_indices'] = selected_indices_int
            
            # 4. 現在可以安全地呼叫工具和執行列表索引操作了
            tool_output = confirm_final_selection_tool.invoke(call['args'])
            final_selection = [state['papers'][i-1] for i in selected_indices_int if 1 <= i <= len(state['papers'])]
            
        tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=call['id']))
    
    # Return updates to the state
    updates = {"messages": tool_messages}
    if papers_from_search is not None:
        updates["papers"] = papers_from_search
    if final_selection is not None:
        updates["filtered_papers"] = final_selection
        
    return updates


# --- 5A. Agent 1 的 Graph ---
def should_continue_research(state: PaperState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and "__END_OF_CONVERSATION__" in last_message.content:
        return "end_conversation"
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tool"
    return "end_turn"

research_workflow = StateGraph(PaperState)
research_workflow.add_node("agent", research_agent_node)
research_workflow.add_node("action", research_tool_node)
research_workflow.set_entry_point("agent")
research_workflow.add_conditional_edges(
    "agent",
    should_continue_research,
    {"end_turn": END, "call_tool": "action", "end_conversation": END}
)
research_workflow.add_edge("action", "agent")


# ==============================================================================
# Agent 2: Summarizer Agent (摘要生成器)
# ==============================================================================

# --- 3B. Agent 2 的工具 ---
@tool
def download_and_parse_pdfs_tool(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Downloads PDFs for a list of papers and extracts their full text."""
    print(f"Tool (Agent 2): Downloading and parsing {len(papers)} PDFs...")
    os.makedirs("downloads", exist_ok=True)
    for paper in papers:
        safe_title = "".join(c for c in paper["title"] if c.isalnum() or c in (' ', '_')).rstrip()
        pdf_path = os.path.join("downloads", f"{safe_title}.pdf")
        try:
            os.system(f"wget -q -O '{pdf_path}' '{paper['pdf_url']}'")
            doc = fitz.open(pdf_path)
            paper["full_text"] = "".join(page.get_text() for page in doc)
        except Exception as e:
            print(f"  - Failed for {paper['title']}: {e}")
            paper["full_text"] = f"Error: Could not process PDF. {e}"
    return papers

@tool
def summarize_papers_tool(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Summarizes the full text of each paper into structured sections."""
    print(f"Tool (Agent 2): Summarizing {len(papers)} papers...")
    summaries = []
    for paper in papers:
        if "Error" in paper.get("full_text", ""):
            summary_content = "Could not generate summary because the PDF could not be processed."
        else:
            prompt = f"Summarize the following paper into sections (e.g., Introduction, Method, Results) in Traditional Chinese.\n\n{paper['full_text'][:20000]}"
            res = large_model.invoke(prompt)
            summary_content = res.content.strip()
        summaries.append({"title": paper["title"], "summary": summary_content})
    return summaries

@tool
def generate_markdown_report_tool(summaries: List[Dict[str, Any]]) -> str:
    """Generates a Markdown file from a list of summaries and returns the file path."""
    print("Tool (Agent 2): Generating Markdown report...")
    md_content = "# 研究論文摘要報告\n\n"
    for s in summaries:
        md_content += f"## {s['title']}\n\n{s['summary']}\n\n---\n\n"
    
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", "report.md")
    # Handle existing file
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join("output", f"report_{counter}.md")
        counter += 1
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    return f"Successfully generated report at: {filepath}"


# --- 4B. Agent 2 的核心邏輯 ---
summarizer_system_prompt = """
You are an automated paper processing agent. Your goal is to take a list of papers and generate a summary report.

You must use your tools in this specific order:
1.  Call `download_and_parse_pdfs_tool` with the list of papers provided.
2.  Take the output from step 1 and call `summarize_papers_tool`.
3.  Take the output from step 2 and call `generate_markdown_report_tool`.
4.  After successfully calling the final tool, announce that the process is complete and output `__SUMMARIZATION_COMPLETE__`.

Provide a brief status update to the user after each tool call.
"""
summarizer_agent_model_with_tools = agent_model.bind_tools([
    download_and_parse_pdfs_tool, summarize_papers_tool, generate_markdown_report_tool
])

def summarizer_agent_node(state: PaperState) -> dict:
    print("Node (Agent 2): Summarizer Agent")
    # The initial message will be from the orchestrator
    messages = [SystemMessage(content=summarizer_system_prompt)] + state['messages']
    response = summarizer_agent_model_with_tools.invoke(messages)
    return {"messages": [response]}

def summarizer_tool_node(state: PaperState) -> dict:
    print("Node (Agent 2): Tool Executor")
    tool_calls = state["messages"][-1].tool_calls
    if not tool_calls:
        return {}

    # This agent should only call one tool at a time
    call = tool_calls[0]
    print(f"  - Executing tool: {call['name']}")
    
    updates = {}
    # Create a clean, human-readable message for the AI.
    tool_message_content = "An error occurred."

    if call['name'] == 'download_and_parse_pdfs_tool':
        # The tool updates the state with the full data
        tool_output_data = download_and_parse_pdfs_tool.invoke({"papers": state['filtered_papers']})
        updates["filtered_papers"] = tool_output_data
        # The message for the AI is a simple confirmation
        tool_message_content = f"Successfully downloaded and parsed {len(tool_output_data)} PDFs."

    elif call['name'] == 'summarize_papers_tool':
        # The tool updates the state with the full data
        tool_output_data = summarize_papers_tool.invoke({"papers": state['filtered_papers']})
        updates["summaries"] = tool_output_data
        # The message for the AI is a simple confirmation
        tool_message_content = f"Successfully generated summaries for {len(tool_output_data)} papers."

    elif call['name'] == 'generate_markdown_report_tool':
        # The tool updates the state with the file path
        tool_output_data = generate_markdown_report_tool.invoke({"summaries": state['summaries']})
        updates["markdown"] = str(tool_output_data)
        # The message for the AI confirms the final step
        tool_message_content = f"Successfully generated report at: {tool_output_data}"
    else:
        tool_message_content = "Tool not found"

    # Pass the clean confirmation message back to the AI.
    updates["messages"] = [ToolMessage(content=tool_message_content, tool_call_id=call['id'])]
    return updates


# --- 5B. Agent 2 的 Graph ---
def should_continue_summarizer(state: PaperState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and "__SUMMARIZATION_COMPLETE__" in last_message.content:
        return END
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tool"
    # Should always call a tool or end
    return END

summarizer_workflow = StateGraph(PaperState)
summarizer_workflow.add_node("agent", summarizer_agent_node)
summarizer_workflow.add_node("action", summarizer_tool_node)
summarizer_workflow.set_entry_point("agent")
summarizer_workflow.add_conditional_edges(
    "agent",
    should_continue_summarizer,
    {
        # Path 1: If the agent needs to call a tool, go to the action node.
        "call_tool": "action",
        # Path 2: If the agent is done (returns END), then end the graph execution.
        END: END
    }
)
summarizer_workflow.add_edge("action", "agent")


# ==============================================================================
# Main Orchestrator (主程式協調者)
# ==============================================================================
if __name__ == "__main__":
    with SqliteSaver.from_conn_string(":memory:") as memory:
        # --- Part 1: Run Research Assistant ---
        print("--- 研究助理 Agent 已啟動 ---")
        research_app = research_workflow.compile(checkpointer=memory)
        config = {"configurable": {"thread_id": "research-thread"}}
        print("你好！我是你的研究助理，請問你想查詢什麼主題的論文？")

        while True:
            try:
                user_input = input("你: ")
                if user_input.lower() in ["exit", "quit", "結束"]:
                    print("好的，期待下次為您服務！")
                    break

                events = research_app.stream({"messages": [("user", user_input)]}, config)
                
                for event in events:
                    # event 的結構是 { node_name: node_output }
                    # 我們需要從 node_output 中尋找 "messages"
                    if not event:
                        continue
                    
                    # 取得節點的回傳值 (node_output)
                    node_output = list(event.values())[0]

                    # 在節點的回傳值中檢查是否有 messages
                    if "messages" in node_output and node_output["messages"]:
                        ai_message = node_output["messages"][-1]
                        if isinstance(ai_message, AIMessage) and ai_message.content:
                            if "__END_OF_CONVERSATION__" not in ai_message.content:
                                print(f"AI: {ai_message.content}")

                current_state_snapshot = research_app.get_state(config)
                current_state_values = current_state_snapshot.values
                
                if should_continue_research(current_state_values) == "end_conversation":
                    print("\n--- 研究助理任務完成，準備移交給摘要生成器 ---")
                    break
            except KeyboardInterrupt:
                print("\n操作已中斷。")
                break
        
        # --- Part 2: Run Summarizer Agent ---
        final_research_state = research_app.get_state(config)
        selected_papers = final_research_state.values.get("filtered_papers")

        if not selected_papers:
            print("沒有選擇任何論文，程式結束。")
        else:
            print(f"\n將處理以下 {len(selected_papers)} 篇論文:")
            for p in selected_papers:
                print(f"- {p['title']}")
            
            print("\n--- 摘要生成器 Agent 已啟動 ---")
            summarizer_app = summarizer_workflow.compile(checkpointer=memory)
            summarizer_config = {"configurable": {"thread_id": "summarizer-thread"}}
            # Create a detailed initial message that gives the AI context.
            paper_titles = "\n".join(f"- {p['title']}" for p in selected_papers)
            initial_content = f"""
            Please process the following list of papers. You must start by calling the `download_and_parse_pdfs_tool`.

            Papers to process:
            {paper_titles}
            """
            initial_message = HumanMessage(content=initial_content)
            summarizer_input = {"messages": [initial_message], "filtered_papers": selected_papers}

            events = summarizer_app.stream(summarizer_input, summarizer_config)
            
            for event in events:
                if not event:
                    continue
                node_output = list(event.values())[0]
                if "messages" in node_output and node_output["messages"]:
                    ai_message = node_output["messages"][-1]
                    if isinstance(ai_message, AIMessage) and ai_message.content:
                         if "__SUMMARIZATION_COMPLETE__" not in ai_message.content:
                            print(f"AI (進度更新): {ai_message.content}")

            print("\n--- 所有任務已完成！ ---")
            final_summarizer_state = summarizer_app.get_state(summarizer_config)
            final_report_path = final_summarizer_state.values.get("markdown")
            if final_report_path:
                print(final_report_path)