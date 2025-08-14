import re
import chainlit as cl
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.schema.runnable.config import RunnableConfig

from src.utils.combined_graph_utils import _create_graph_builder, get_langfuse_handler


@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    answer_prefix_tokens = ["Final", " Answer", "Final Answer:"]
    cb = cl.LangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=answer_prefix_tokens,)
    final_answer = cl.Message(content="")
    question = msg.content
    history = cl.user_session.get("history", [])
    inputs = {
        "question": question,
        "conversation_history": history
    }
    history.append({"role": "user", "content": question})
    builder = _create_graph_builder()
    langfuse_handler = get_langfuse_handler()
    buffer = ""
    streaming = False
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        graph_with_memory = builder.compile(checkpointer=memory).with_config({"callbacks": [langfuse_handler]})

        async for msg, metadata in graph_with_memory.astream(inputs,
                                                              stream_mode="messages",
                                                              config=RunnableConfig(callbacks=[cb], **config)):
            content = msg.content
            print(f"----------{content}")
            if not streaming:
                buffer += content
                print(f"############{buffer}")
                match = re.search(
                    r"Final\s+Answer\s*:\s*", buffer, re.IGNORECASE
                )
                if match:
                    streaming = True
                else:
                    continue
            if content == question:
                continue
            await final_answer.stream_token(content)
    history.append({"role": "assistant", "content": final_answer})
    await final_answer.send()
