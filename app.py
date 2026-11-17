import os
import certifi
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic.agents import create_react_agent, AgentExecutor
from langsmith import Client


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TRAVILY_API_KEY = os.getenv("TRAVILY_API_KEY")


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Smart AI Assistant",
    page_icon="🤖",
    layout="centered"
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .status {
        padding: 10px;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="title">🤖 Smart AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions and let AI search the web when needed.</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Validate API Keys
# ---------------------------------------------------------

if not GEMINI_API_KEY:
    st.error("GEMINAI_API_KEY is missing from your .env file.")
    st.stop()

if not TRAVILY_API_KEY:
    st.error("TRAVILY_API_KEY is missing from your .env file.")
    st.stop()


# ---------------------------------------------------------
# Initialize Agent
# ---------------------------------------------------------

@st.cache_resource
def initialize_agent():

    # LangSmith public ReAct prompt
    client = Client()

    prompt = client.pull_prompt(
        "hwchase17/react",
        dangerously_pull_public_prompt=True
    )

    # Tavily search tool
    search_tool = TavilySearchResults(
        max_results=3
    )

    tools = [search_tool]

    # Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2
    )

    # Create ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    return executor


try:
    agent_executor = initialize_agent()

except Exception as e:
    st.error(f"Failed to initialize AI agent: {e}")
    st.stop()


# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Assistant")

    st.write(
        """
        This assistant uses:

        **LLM**
        Gemini

        **Agent**
        LangChain ReAct

        **Web Search**
        Tavily

        **UI**
        Streamlit
        """
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------------
# Display Previous Messages
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------------
# Chat Input
# ---------------------------------------------------------

user_input = st.chat_input(
    "Ask me anything..."
)


# ---------------------------------------------------------
# Process User Message
# ---------------------------------------------------------

if user_input:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = agent_executor.invoke(
                    {
                        "input": user_input
                    }
                )

                response = result["output"]

                st.markdown(response)

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

            except Exception as e:

                error_message = (
                    f"Sorry, something went wrong.\n\n"
                    f"`{str(e)}`"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )