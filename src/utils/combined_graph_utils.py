import os
import operator
from typing import Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langfuse.langchain import CallbackHandler
from langchain_tavily import TavilySearch
from langfuse import get_client
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from src.utils.agentic_utils import get_react_agent
from src.utils.multivector_utils import create_retriever_pipeline
from src.utils.prompt_utils import *

load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Set env variables for Langfuse
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1",
    temperature=0,
)

embedding_model = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small"
)

# Create Retrievers
insurance_policy_retriever = create_retriever_pipeline(
    di_results_filename="./DI_output/Insurance_Policy_results.pkl",
    source_file_name="Insurance_Policy",
    pickle_path = "./parent_pickles/",
    vector_db_name="./chroma-insurance-policy",
    embeddings_model=embedding_model,
    llm_model=llm,
    vectorstore_exists=True
)

leave_policy_retriever = create_retriever_pipeline(
    di_results_filename="./DI_output/Leave_Policy_results.pkl",
    source_file_name="Leave_Policy",
pickle_path = "./parent_pickles/",
    vector_db_name="./chroma-leave-policy",
    embeddings_model=embedding_model,
    llm_model=llm,
    vectorstore_exists=True
)

MS23_retriever = create_retriever_pipeline(
    di_results_filename="./DI_output/Microsoft_2023_results.pkl",
    source_file_name="Microsoft_2023",
pickle_path = "./parent_pickles/",
    vector_db_name="./chroma-microsoft-2023",
    embeddings_model=embedding_model,
    llm_model=llm,
    vectorstore_exists=True
)

MS24_retriever = create_retriever_pipeline(
    di_results_filename="./DI_output/Microsoft_2024_results.pkl",
    source_file_name="Microsoft_2024",
pickle_path = "./parent_pickles/",
    vector_db_name="./chroma-microsoft-2024",
    embeddings_model=embedding_model,
    llm_model=llm,
    vectorstore_exists=True
)

AP23_retriever = create_retriever_pipeline(
    di_results_filename="./DI_output/Apple_2023_results.pkl",
    source_file_name="Apple_2023",
pickle_path = "./parent_pickles/",
    vector_db_name="./chroma_apple-2023",
    embeddings_model=embedding_model,
    llm_model=llm,
    vectorstore_exists=True
)

AP24_retriever = create_retriever_pipeline(
    di_results_filename="./DI_output/Apple_2024_results.pkl",
    source_file_name="Apple_2024",
pickle_path = "./parent_pickles/",
    vector_db_name="./chroma_apple-2024",
    embeddings_model=embedding_model,
    llm_model=llm,
    vectorstore_exists=True
)


# Tool creation
Insurance_Policy_tool = create_retriever_tool(retriever=insurance_policy_retriever,
                            name = 'Insurance_Policy_Retriever',
                            description="Use this tool to answer questions related to Health Insurance Policies of a company.")

Leave_Policy_tool = create_retriever_tool(retriever=leave_policy_retriever,
                            name = 'Leave_Policy_Retriever',
                            description="Use this tool to answer questions related to Leave Policies of a company.")

Microsoft_2023_tool = create_retriever_tool(retriever=MS23_retriever,
                            name = 'Microsoft_2023_Retriever',
                            description="Use this tool to answer questions related to the annual reports of Microsoft 2023.")

Microsoft_2024_tool = create_retriever_tool(retriever=MS24_retriever,
                            name = 'Microsoft_2024_Retriever',
                            description="Use this tool to answer questions related to the annual reports of Microsoft 2024.")

Apple_2023_tool = create_retriever_tool(retriever=AP23_retriever,
                            name = 'Apple_2023_Retriever',
                            description="Use this tool to answer questions related to the annual reports of Apple 2023.")

Apple_2024_tool = create_retriever_tool(retriever=AP24_retriever,
                            name = 'Apple_2024_Retriever',
                            description="Use this tool to answer questions related to the annual reports of Apple 2024.")

tavily_search_tool = TavilySearch(
            tavily_api_key=os.getenv('TAVILY_API_KEY'),
            max_results=7,
            topic="general",
        )


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    original_question: str
    query_routed_to: str
    overall_status_check: str

    question: str
    final_answer: str
    conversation_history: Annotated[list[AnyMessage], operator.add]

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


# Function to rewrite queries
def rewrite_query(_llm, prompt, conversation_history: str, user_query: str) -> str:
    """
    Rewrites the user query based on the conversation
    """
    # Prompt template
    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["question"],
    )

    # LCEL chain
    query_rewriter_chain = prompt_template | _llm

    # Invoke the chain
    result = query_rewriter_chain.invoke({
        "conversation_history": conversation_history,
        "user_query": user_query
    })
    return result.content.strip()


def query_rewriter_node(state):
    """Use this tool to rewrite the original query based on the conversation history."""
    observation = rewrite_query(llm, QUERY_REWRITER_PROMPT, state['conversation_history'], state['question'])

    print("------ENTERING: QUERY REWRITER NODE------")
    print(f"------RESULT: {observation}------")

    return {"original_question":state['question'], "question": observation}


class RerouterCheck(BaseModel):
    """Query Rerouter Output."""
    rerouter_output: str = Field(description="Given a user question, route the question to either 'INTERNAL' OR 'GENERIC' OR 'WEB'")


def query_rerouter(_llm, prompt, conversation_history, question) -> RerouterCheck:
    """Run source detection using LLM with structured output."""

    # Output parser
    parser = PydanticOutputParser(pydantic_object=RerouterCheck)

    # Prompt template
    prompt = PromptTemplate(
        template=prompt,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    query_rerouter_chain = prompt | _llm | parser

    result = query_rerouter_chain.invoke({
        "conversation_history": conversation_history,
         "question": question
        }
        )
    return result.rerouter_output.strip()


def query_rerouter_node(state):
    """Use this tool to detect top sources based on the classification rules & conversation history."""
    observation = query_rerouter(llm, QUERY_REROUTER_PROMPT, state['conversation_history'], state['question'])

    print("------ENTERING: QUERY REROUTER NODE------")
    print(f"------RESULT: {observation}------")

    return {"query_routed_to": observation}


def get_generic_answer(_llm, prompt, conversation_history, question):
    """get generic answer"""
    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["question"],
    )
    generic_answer_chain = prompt_template | _llm

    result = generic_answer_chain.invoke({
        "conversation_history": conversation_history,
        "question": question
    })
    return result.content.strip()


def generic_answer_node(state):
    """Use this tool to answer user generic questions"""
    print("------ENTERING: GENERIC ANSWER NODE------")

    final_answer = get_generic_answer(llm, GENERIC_ANSWER_PROMPT, state['conversation_history'], state['question'])

    return {"conversation_history": [HumanMessage(content=state["question"]),
                                     AIMessage(content=final_answer)],
                                     "final_answer": final_answer}


def web_answer_node(state):
    """Use this tool when you need to answer questions related to current events and latest happenings"""
    print("------ENTERING: WEB ANSWER NODE------")
    tools = [tavily_search_tool]
    generate_agent = get_react_agent(
        llm,
        tools,
        COMBINED_REACT_PROMPT,
        verbose=True,
    )
    with get_openai_callback() as cb:
        answer = generate_agent.invoke(
            {
                "input": state["question"],
                "conversation_history": state["conversation_history"],
                "SYSTEM_PROMPT": SYSTEM_PROMPT,
                "GENERAL_INSTRUCTIONS": General_Instructions,
                "ADONIS_SPECIAL_INSTRUCTIONS": Adonis_Special_Instructions,
                "HEALTH_INSURANCE_SPECIAL_INSTRUCTIONS": Health_Special_Instructions,
                "LEAVE_POLICY_SPECIAL_INSTRUCTIONS": Leave_Special_Instructions,
                "ANNUAL_REPORT_SPECIAL_INSTRUCTIONS": ANNAUAL_REPORT_SPECIAL_INSTRUCTIONS,
                "WEB_SPECIAL_INSTRUCTIONS": WEB_Special_Instructions
            }
        )

    return {"conversation_history": [HumanMessage(content=state["question"]),
                                     AIMessage(content=answer["output"])],
            "final_answer": answer["output"]}


def private_internal_data_answer_node(state):
    """Use this tool to answer any questions related to leave policies of the compny"""
    print("------ENTERING: PRIVATE INTERNAL DATA ANSWER NODE------")
    tools = [Leave_Policy_tool, Insurance_Policy_tool, Microsoft_2023_tool, Microsoft_2024_tool, Apple_2023_tool, Apple_2024_tool, tavily_search_tool]
    generate_agent = get_react_agent(
        llm,
        tools,
        COMBINED_REACT_PROMPT,
        verbose=True,
    )
    with get_openai_callback() as cb:
        answer = generate_agent.invoke(
            {
                "input": state["question"],
                "conversation_history": state["conversation_history"],
                "SYSTEM_PROMPT": SYSTEM_PROMPT,
                "GENERAL_INSTRUCTIONS": General_Instructions,
                "ADONIS_SPECIAL_INSTRUCTIONS": Adonis_Special_Instructions,
                "HEALTH_INSURANCE_SPECIAL_INSTRUCTIONS": Health_Special_Instructions,
                "LEAVE_POLICY_SPECIAL_INSTRUCTIONS": Leave_Special_Instructions,
                "ANNUAL_REPORT_SPECIAL_INSTRUCTIONS": ANNAUAL_REPORT_SPECIAL_INSTRUCTIONS,
                "WEB_SPECIAL_INSTRUCTIONS": WEB_Special_Instructions
            }
        )

    return {"conversation_history": [HumanMessage(content=state["question"]),
                                     AIMessage(content=answer["output"])],
            "final_answer": answer["output"]}


def overall_status_check_node(state):
    """Use this tool to check the overall status and update the config settings"""
    print("------ENTERING: OVERALL STATUS CHECK NODE------")
    final_answer = state['final_answer']
    observation = "Completed"
    return {"overall_status_check": observation, 'final_answer':final_answer}


def _create_graph_builder():
    # Set up the state graph
    builder = StateGraph(GraphState)
    builder.add_node("query_rewriter_node", query_rewriter_node)
    builder.add_node("query_rerouter_node", query_rerouter_node)
    builder.add_node("generic_answer_node", generic_answer_node)
    builder.add_node("web_answer_node", web_answer_node)
    builder.add_node("private_internal_data_answer_node", private_internal_data_answer_node)
    builder.add_node("overall_status_check_node", overall_status_check_node)
    builder.set_entry_point("query_rewriter_node")

    builder.add_edge("query_rewriter_node", "query_rerouter_node")

    builder.add_conditional_edges(
        "query_rerouter_node",
        lambda x: x["query_routed_to"],
        {
            "INTERNAL": "private_internal_data_answer_node",
            "WEB": "web_answer_node",
            "GENERIC": "generic_answer_node",
        },
    )

    builder.add_edge("generic_answer_node", "overall_status_check_node")
    builder.add_edge("web_answer_node", "overall_status_check_node")
    builder.add_edge("private_internal_data_answer_node", "overall_status_check_node")

    builder.set_finish_point("overall_status_check_node")

    return builder


def get_langfuse_handler():
    # langfuse config
    langfuse = get_client()
    # Verify connection
    if langfuse.auth_check():
        print("Langfuse client is authenticated and ready!")
    else:
        print("Authentication failed. Please check your credentials and host.")

    langfuse_handler = CallbackHandler()
    return langfuse_handler
