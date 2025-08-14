import os
import json
import neo4j
import operator
from typing import Annotated
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from src.utils.agentic_utils import get_react_agent
from langchain_community.callbacks import get_openai_callback
from langchain.tools import Tool
from neo4j_graphrag.schema import get_schema
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.llm import OpenAILLM

from neo4j_graphrag.retrievers import HybridCypherRetriever, Text2CypherRetriever
from neo4j_graphrag.types import RetrieverResultItem
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from src.utils.examples import examples
from src.utils.prompt_utils import(
    custom_text2cypher_prompt,
    rag_prompt,
    CYPHER_REACT_PROMPT,
    REACT_PROMPT)

load_dotenv()

API_BASE = os.getenv("API_BASE")
API_VERSION = os.getenv("API_VERSION")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
API_KEY = os.getenv("API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NEO4J_URI = os.getenv("AV_NEO4J_URI")
NEO4J_USER = os.getenv("AV_NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("AV_NEO4J_PASSWORD")

neo4j_answering_llm = OpenAILLM(
    model=DEPLOYMENT_NAME,
    api_key=OPENAI_API_KEY
)

neo4j_embedder = OpenAIEmbeddings(
    model=EMBEDDING_DEPLOYMENT_NAME
)

lang_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=DEPLOYMENT_NAME, temperature=0)

class GraphState(TypedDict):
    question: str
    final_answer: str
    conversation_history: Annotated[list[AnyMessage], operator.add]

driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

INDEX_NAME = "entity_vector_index"
FULLTEXT_INDEX_NAME = "entity_fulltext_index"

def result_formatter_dynamic(record):
    data = record.data()
    node_props = dict(list(data.values())[0]) if len(data) == 1 and isinstance(list(data.values())[0], dict) else dict(data)
    content = "\n".join(f"{k}: {v}" for k, v in node_props.items())

    return RetrieverResultItem(
        content=content.strip(),
        metadata={
            "raw_properties": node_props,
            "score": record.get("score"),
            "node_keys": list(node_props.keys())
        }
    )

@tool("text2cypher_tool", description="Convert natural language query to Cypher.")
def text2cypher_tool(query: str) -> str:
    text2cypher = Text2CypherRetriever(
        driver=driver,
        llm=neo4j_answering_llm,
        neo4j_schema=get_schema(driver),
        custom_prompt=custom_text2cypher_prompt,
        examples=examples
    )
    result = text2cypher.search(query)
    return result.metadata["cypher"]

def get_rag_for_query(query: str, cypher_query: str):
    retriever = HybridCypherRetriever(
        driver=driver,
        vector_index_name=INDEX_NAME,
        fulltext_index_name=FULLTEXT_INDEX_NAME,
        retrieval_query=cypher_query,
        embedder=neo4j_embedder,
        result_formatter=result_formatter_dynamic,
    )

    prompt_template = RagTemplate(
        template=rag_prompt,
        expected_inputs=["context", "query_text"]
    )

    rag = GraphRAG(retriever=retriever, llm=neo4j_answering_llm, prompt_template=prompt_template)

    response = rag.search(
        query,
        return_context=True,
        retriever_config={'top_k': 20},
        response_fallback="I can't answer this question without context"
    )

    for i, item in enumerate(response.retriever_result.items, 1):
        print(f"🔎 Context Item {i}:\n📄 {item.content}\n📘 {item.metadata}\n---\n")

    return response.answer


def hybrid_tool_wrapper(input_str: str) -> str:
    try:
        parsed = json.loads(input_str)
        query = parsed["query"]
        cypher_query = parsed["cypher_query"]
        return get_rag_for_query(query, cypher_query)
    except Exception as e:
        return f"Failed to parse input: {e}"


av_hybrid_tool = Tool(
    name="AVVectorRetrieval",
    func=hybrid_tool_wrapper,
    description=(
        "Use this tool to answer questions about the Analytics Vidhya DataHack Summit. "
        "Input should be a JSON object with 'query' and 'cypher_query' keys, e.g.: "
        '{"query": "List speakers", "cypher_query": "MATCH (s:Speaker)..."}'
    )
)

def av_answer_node(state):
    print("------ENTERING: AV HYBRID ANSWER NODE------")

    # Step 1: Generate Cypher
    cypher_agent = get_react_agent(
        lang_llm,
        [text2cypher_tool],
        CYPHER_REACT_PROMPT,
        verbose=True
    )

    cypher_result = cypher_agent.invoke({
        "input": state["question"],
        "schema": get_schema(driver)
    })

    cypher_query = cypher_result["output"].strip().strip("'\"")

    # Step 2: Run Hybrid Agent with Cypher
    hybrid_agent = get_react_agent(
        lang_llm,
        [av_hybrid_tool],
        REACT_PROMPT,
        verbose=True
    )

    hybrid_input = {
        "query": state["question"],
        "cypher_query": cypher_query
    }

    with get_openai_callback() as cb:
        answer = hybrid_agent.invoke({
            "input": json.dumps(hybrid_input),
            "conversation_history": state["conversation_history"],
        })

    return {
        "conversation_history": [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer["output"])
        ],
        "final_answer": answer["output"]
    }


def _create_graph_builder():
    # Set up the state graph
    builder = StateGraph(GraphState)
    builder.add_node("answer_node", av_answer_node)
    builder.set_entry_point("answer_node")
    builder.set_finish_point("answer_node")

    return builder