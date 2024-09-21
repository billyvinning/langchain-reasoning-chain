import pytest
from langchain_core.messages import HumanMessage

from langchain_reasoning_chain import create_reasoning_chain_agent


def test_import():
    import langchain_reasoning_chain

    modules = set(langchain_reasoning_chain.__all__)

    assert modules == {"create_reasoning_chain_agent"}


@pytest.mark.xfail(
    reason="Must have Vertex AI and necessary permissions.",
    raises=ImportError,
)
def test_google_vertex_ai():
    from langchain_google_vertexai import ChatVertexAI  # type: ignore[import-not-found]

    llm = ChatVertexAI(model_name="gemini-1.5-flash")
    reasoning_chain = create_reasoning_chain_agent(llm, min_steps=3)

    events = reasoning_chain.stream(
        {"messages": [HumanMessage("How many Rs are in strawberry?")]},
        stream_mode="values",
    )

    for event in events:
        event["messages"][-1].pretty_print()
