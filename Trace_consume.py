"""
Example of using LangGraphDetailedTracer with a LangGraph agent.
"""

import os
from typing import Dict, List, Tuple, Any, Annotated
from langgraph.graph import StateGraph, END
from operator import itemgetter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Import the tracer we just created
from langgraph_tracer import LangGraphDetailedTracer


# Define helper functions for our agent
def create_agent_with_tools(tools: List[Tuple[str, Runnable]]) -> Runnable:
    """Create an agent that can use tools."""
    
    # The tools we'll expose to the agent
    tool_map = {name: tool for name, tool in tools}
    tool_names = list(tool_map.keys())
    
    # Prompt that will be used in the final decision-making agent
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are a helpful AI assistant. You have access to the following tools:

{tool_names}

To use a tool, respond with:
```
<tool>TOOL_NAME</tool>
<input>TOOL_INPUT</input>
```

Once you no longer need to use tools to answer the human's question or when the human's request has been fulfilled, respond directly without tool invocation.
"""),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    llm = ChatOpenAI(temperature=0)
    
    # Function to parse the agent output to extract tool use
    def parse_output(message: AIMessage) -> Dict:
        import re
        
        if "<tool>" not in message.content:
            # No tool use, return the message as is
            return {"output": message.content}
        
        # Extract tool name and input
        tool_pattern = r"<tool>(.*?)</tool>"
        input_pattern = r"<input>(.*?)</input>"
        
        tool_match = re.search(tool_pattern, message.content, re.DOTALL)
        input_match = re.search(input_pattern, message.content, re.DOTALL)
        
        if not tool_match or not input_match:
            return {"output": message.content}
        
        tool_name = tool_match.group(1).strip()
        tool_input = input_match.group(1).strip()
        
        return {
            "tool": tool_name,
            "input": tool_input,
        }
    
    # Function that formats the intermediate steps for the agent
    def create_agent_scratchpad(intermediate_steps: List[Tuple[Dict, str]]) -> List[BaseMessage]:
        scratchpad = []
        for action, result in intermediate_steps:
            scratchpad.append(AIMessage(content=f"I'll use the {action['tool']} tool with input: {action['input']}"))
            scratchpad.append(SystemMessage(content=f"Tool result: {result}"))
        return scratchpad
    
    # The agent that will make decisions
    agent = (
        prompt 
        | llm 
        | (lambda x: AIMessage(content=x.content)) 
        | parse_output
    )
    
    return agent


def run_agent_with_tools(
    messages: List[BaseMessage], 
    intermediate_steps: List[Tuple[Dict, str]]
) -> Dict:
    """Run the agent with additional context from intermediate steps."""
    
    # Create the agent with tools
    tools = [
        ("Calculator", lambda x: str(eval(x))),
        ("Weather", lambda x: f"The weather in {x} is currently sunny."),
        ("Search", lambda x: f"Search results for '{x}': This is a simulated search result."),
    ]
    
    agent = create_agent_with_tools(tools)
    
    # Run the agent
    return agent.invoke({
        "messages": messages,
        "agent_scratchpad": create_agent_scratchpad(intermediate_steps),
    })


# Define the graph nodes
def should_continue(state: Dict) -> str:
    """Determine if we should continue running the agent."""
    
    # If we have a final output, we're done
    if "output" in state.get("agent_outcome", {}):
        return "end"
    
    # Otherwise, we need to run a tool
    return "continue"


def run_tool(state: Dict) -> Dict:
    """Run the specified tool and get the result."""
    agent_outcome = state["agent_outcome"]
    tool_name = agent_outcome["tool"]
    tool_input = agent_outcome["input"]
    
    # Define the tools
    tools = {
        "Calculator": lambda x: str(eval(x)),
        "Weather": lambda x: f"The weather in {x} is currently sunny.",
        "Search": lambda x: f"Search results for '{x}': This is a simulated search result.",
    }
    
    try:
        # Try to run the tool and get the result
        if tool_name in tools:
            tool_result = tools[tool_name](tool_input)
        else:
            tool_result = f"Error: Tool '{tool_name}' not found."
    except Exception as e:
        tool_result = f"Error: {str(e)}"
    
    # Update the state with the tool result
    return {
        "intermediate_steps": state["intermediate_steps"] + [(agent_outcome, tool_result)]
    }


def run_agent(state: Dict) -> Dict:
    """Run the agent with the current state."""
    messages = state["messages"]
    intermediate_steps = state["intermediate_steps"]
    
    # Run the agent
    agent_outcome = run_agent_with_tools(messages, intermediate_steps)
    
    # Return the updated state
    return {"agent_outcome": agent_outcome}


# Create the graph
def build_graph() -> StateGraph:
    """Build the agent graph."""
    # Define the graph
    graph = StateGraph(name="Agent With Tools")
    
    # Define the nodes
    graph.add_node("agent", run_agent)
    graph.add_node("tool", run_tool)
    
    # Add the conditional edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tool",
            "end": END,
        },
    )
    
    # Add the normal edges
    graph.add_edge("tool", "agent")
    
    # Set the entry point
    graph.set_entry_point("agent")
    
    # Set the initial state
    graph.set_initial_state({
        "messages": [],
        "intermediate_steps": [],
    })
    
    return graph


# Function to run the agent with tracing
def run_agent_graph_with_tracing(query: str, trace_file: str = "agent_trace.json") -> Dict:
    """Run the agent graph with detailed tracing."""
    
    # Create the tracer
    tracer = LangGraphDetailedTracer(
        trace_file=trace_file,
        auto_save=True,
        save_interval=5  # Save every 5 seconds
    )
    
    # Build the graph
    graph = build_graph().compile()
    
    # Run the graph with the tracer
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "intermediate_steps": [],
        },
        {"callbacks": [tracer]}
    )
    
    # Generate HTML report
    report_file = trace_file.replace(".json", ".html")
    tracer.generate_html_report(report_file)
    
    print(f"Trace saved to {trace_file}")
    print(f"HTML report saved to {report_file}")
    
    # Print summary stats
    stats = tracer.generate_summary_stats()
    print("\nExecution Summary:")
    print(f"Total duration: {stats['total_duration']:.2f} seconds")
    print(f"LLM calls: {stats['llm_stats']['call_count']}")
    print(f"Total tokens: {stats['llm_stats']['total_tokens']['total']}")
    print(f"Tool calls: {stats['tool_stats']['call_count']}")
    print(f"Node executions: {stats['node_stats']['node_count']}")
    print(f"State transitions: {stats['transition_stats']['transition_count']}")
    
    return result


# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Run the agent with tracing
    query = "What is the square root of 144 plus the square root of 169?"
    result = run_agent_graph_with_tracing(query)
    
    print("\nFinal Answer:")
    if "output" in result["agent_outcome"]:
        print(result["agent_outcome"]["output"])
    else:
        print("Agent did not provide a final answer.")


import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# DB setup
DB_URL = os.getenv("DATABASE_URL")  # Set this in your .env
engine = create_engine(DB_URL)

# Title and menu
st.set_page_config(page_title="AI Agent Tracing Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Data"])

# Auth (basic Streamlit login check, replace with your method if needed)
PASSWORD = os.getenv("APP_PASSWORD")
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    with st.sidebar.form("Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit and password == PASSWORD:
            st.session_state["authenticated"] = True
        elif submit:
            st.error("Invalid credentials")
    st.stop()

# Utility function
@st.cache_data(ttl=300)
def load_data():
    query = text("SELECT * FROM agent_logs ORDER BY start_time DESC LIMIT 10000")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["duration"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    return df


# Page: Dashboard
if page == "Dashboard":
    st.title("📊 AI Agent Dashboard")

    df = load_data()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        selected_user = st.selectbox("Filter by User", ["All"] + sorted(df["user_id"].unique().tolist()))
    with col2:
        selected_convo = st.selectbox("Filter by Conversation", ["All"] + sorted(df["conversation_id"].unique().tolist()))

    filtered_df = df.copy()
    if selected_user != "All":
        filtered_df = filtered_df[filtered_df["user_id"] == selected_user]
    if selected_convo != "All":
        filtered_df = filtered_df[filtered_df["conversation_id"] == selected_convo]

    # Metrics
    st.subheader("⚙️ Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Traces", len(filtered_df))
    col2.metric("Avg Latency (s)", round(filtered_df["duration"].mean(), 2))
    col3.metric("Avg Tokens", filtered_df["llm_response"].apply(lambda x: x.get("token_usage", {}).get("total_tokens", 0) if isinstance(x, dict) else 0).mean())

    # Charts
    st.subheader("📈 Activity Over Time")
    if not filtered_df.empty:
        time_data = (
            filtered_df
            .groupby(filtered_df["start_time"].dt.date)
            .agg({
                "question_id": "count",
                "duration": "mean"
            })
            .rename(columns={"question_id": "trace_count"})
            .reset_index()
        )

        fig = px.bar(time_data, x="start_time", y="trace_count", title="Traces per Day")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(time_data, x="start_time", y="duration", title="Avg Latency per Day (s)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No data available for selected filters.")

# Page: Data
elif page == "Data":
    st.title("📄 Raw Logs Table")

    df = load_data()

    # Filters
    st.sidebar.subheader("Filters")
    user_filter = st.sidebar.multiselect("User", df["user_id"].unique())
    convo_filter = st.sidebar.multiselect("Conversation ID", df["conversation_id"].unique())

    if user_filter:
        df = df[df["user_id"].isin(user_filter)]
    if convo_filter:
        df = df[df["conversation_id"].isin(convo_filter)]

    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="📥 Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="agent_logs.csv",
        mime="text/csv"
    )


