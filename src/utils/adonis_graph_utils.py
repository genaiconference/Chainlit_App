import os
import operator
from typing import Annotated
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from src.utils.agentic_utils import get_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools.retriever import create_retriever_tool
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from src.utils.prompt_utils import(
    DG_REACT_PROMPT,
    SYSTEM_PROMPT,
    Adonis_Special_Instructions,
    WEB_Special_Instructions,
    General_Instructions,
    cypher_generation_template,
    qa_generation_template_str)

load_dotenv()

API_BASE = os.getenv("API_BASE")
API_VERSION = os.getenv("API_VERSION")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
API_KEY = os.getenv("API_KEY")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

llm = AzureChatOpenAI(
    azure_endpoint=API_BASE,
    openai_api_version=API_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
    streaming=True,
    temperature=0
)

embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=EMBEDDING_DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    azure_endpoint=API_BASE,
    openai_api_version=API_VERSION,
)


class GraphState(TypedDict):
    question: str
    final_answer: str
    conversation_history: Annotated[list[AnyMessage], operator.add] #InMemoryMessageHistory()

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
)

vectorstore = Neo4jVector.from_existing_graph(
    embedding=embedding_model,
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name="adonis_embedding_index",
    node_label="Searchable",
    text_node_properties=["name", "description"],
    embedding_node_property="embedding",
    distance_strategy="COSINE"
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "filter": {}
    }
)

adonis_graph_retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="adonis_semantic_graph_search",
    description="Use this tool to answer the questions regarding the process involved in the finance core with flow charts. **Use this tool mainly for descriptive based questions**"
)
cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=cypher_generation_template
)
qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template_str
)
cypher_chain = GraphCypherQAChain.from_llm(
    top_k=10,
    graph=graph,
    verbose=True,
    validate_cypher=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    qa_llm=llm,
    cypher_llm=llm,
    allow_dangerous_requests=True
)


@tool("graph_tool", return_direct=False)
def graph_tool(query: str) -> str:
    """Use this tool when the user asks a question that requires querying structured information
    from a Neo4j graph database. Ideal for answering questions related to relationships,
    hierarchies, dependencies, connections between entities, or insights stored in the graph.

    The graph contains entities such as:
    - Steps (which belong to process diagrams)
    - Process Diagrams (which are grouped under Subprocess Areas → Process Areas → Functions)
    - Roles (responsible or accountable for executing specific steps)
    - NFCM Controls (linked to steps via 'INVOLVED_IN' relationships)
    - Inter-diagram references (e.g., 'REFERENCED_EVENT', 'CROSS_REFERENCE', 'REFERENCED_SUBPROCESS')

    Use this tool when:
    - The user asks about who is responsible or accountable for a particular step or process
    - The question involves business process hierarchies, functional areas, or diagram structure
    - The user wants to find controls implemented in a process or step
    - The query relates to semantic or relational search across roles, functions, controls, or process flows
    - You need to traverse relationships or analyze dependencies (e.g., which diagrams a step connects to)

    Avoid using this tool for:
    - General or factual knowledge unrelated to business processes or the graph
    - Questions that don't require traversing or querying the graph database

    Always prefer this tool if the user query involves understanding structured workflows, responsibilities or control frameworks within a business process.
	"""
    response = cypher_chain.invoke(query)
    return response.get("result")


@tool(description="Use this tool to answer latest happenings from internet")
def web_tool(query):
    try:
        # Custom wrapper with region and max results
        search_tool = DuckDuckGoSearchRun(requests_kwargs={"verify": False})
        # Search for news
        return search_tool.invoke(query)
    except:
        print("Encountered Issue! Please try again!")


def web_answer_node(state):
    print(state)
    print("------ENTERING: WEB ANSWER NODE------")
    tools = [web_tool]
    generate_agent = get_react_agent(
        llm,
        tools,
        DG_REACT_PROMPT,
        verbose=True,
    )
    with get_openai_callback() as cb:
        answer = generate_agent.invoke(
            {
                "input": state["question"],
                "conversation_history": state["conversation_history"],
                "SYSTEM_PROMPT": SYSTEM_PROMPT,
                "GENERAL_INSTRUCTIONS": General_Instructions,
                "SPECIAL_INSTRUCTIONS": WEB_Special_Instructions,
            }
        )

    return {"conversation_history": [HumanMessage(content=state["question"]),
                                     AIMessage(content=answer["output"])],
            "final_answer": answer["output"]}


def adonis_answer_node(state):
    print("------ENTERING: ADONIS ANSWER NODE------")
    tools = [graph_tool, adonis_graph_retriever_tool]
    generate_agent = get_react_agent(
        llm,
        tools,
        DG_REACT_PROMPT,
        verbose=True,
    )
    with get_openai_callback() as cb:
        answer = generate_agent.invoke(
            {
                "input": state["question"],
                "conversation_history": state["conversation_history"],
                "SYSTEM_PROMPT": SYSTEM_PROMPT,
                "GENERAL_INSTRUCTIONS": General_Instructions,
                "SPECIAL_INSTRUCTIONS": Adonis_Special_Instructions,
            }
        )

    return {"conversation_history": [HumanMessage(content=state["question"]),
                                     AIMessage(content=answer["output"])],
            "final_answer": answer["output"]}


def _create_graph_builder():
    # Set up the state graph
    builder = StateGraph(GraphState)
    builder.add_node("answer_node", adonis_answer_node)
    builder.set_entry_point("answer_node")
    builder.set_finish_point("answer_node")

    return builder
