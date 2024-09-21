import operator
from enum import StrEnum
from typing import Annotated, Optional

from json_repair import repair_json
from langchain_core.language_models import LLM
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

REASON_NODE_NAME = "Reason"
ANSWER_NODE_NAME = "Answer"
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert AI assistant that explains your reasoning step by step. "
    "For each step, provide a title that describes what you're doing in that step, "
    "along with the content. Decide if you need another step or if you're ready to "
    "give the final answer. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. "
    "BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. "
    "IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. "
    "CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, "
    "WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. "
    "YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, "
    "AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. "
    "USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES."
)


class ChainOfThoughtModel(BaseModel):
    title: str = Field(
        description="A summary of your reasoning in the form of a title.",
        examples=["Identifying Key Information"],
    )
    reasoning: str = Field(
        description="Your reasoning.",
        examples=[
            (
                "To begin solving this problem, we need to carefully examine the "
                "given information and identify the crucial elements that will "
                "guide our solution process. This involves..."
            ),
        ],
    )


class NextAction(StrEnum):
    CONTINUE = "continue"
    FINAL_ANSWER = "final_answer"


class ChainOfThoughtStepModel(ChainOfThoughtModel):
    next_action: NextAction = Field(description="Your next action.")


class State(BaseModel):
    messages: Annotated[list[BaseMessage], operator.add]


def create_reasoning_chain_agent(
    llm: LLM,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    min_steps: int = 0,
    max_steps: Optional[int] = None,
) -> CompiledStateGraph:
    warmup_parser: PydanticOutputParser = PydanticOutputParser(
        pydantic_object=ChainOfThoughtModel,
    )
    warmup_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=system_prompt + warmup_parser.get_format_instructions(),
            ),
            HumanMessagePromptTemplate.from_template("{user_message}"),
        ],
    )
    hot_parser: PydanticOutputParser = PydanticOutputParser(
        pydantic_object=ChainOfThoughtStepModel,
    )
    hot_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt + hot_parser.get_format_instructions()),
            HumanMessagePromptTemplate.from_template("{user_message}"),
        ],
    )

    n_steps = 0

    def route_answer(state: State) -> NextAction:
        nonlocal n_steps
        parser = warmup_parser if n_steps < min_steps else hot_parser
        assert isinstance(state.messages[-1].content, str)
        content = repair_json(state.messages[-1].content)
        assert isinstance(content, str)
        answer = parser.invoke(content)

        if n_steps < min_steps:
            out = NextAction.CONTINUE
        elif max_steps is not None and n_steps >= max_steps:
            out = NextAction.FINAL_ANSWER
        else:
            out = answer.next_action
        n_steps += 1
        return out

    def reason(state: State) -> dict[str, list[BaseMessage]]:
        nonlocal n_steps
        prompt_template = (
            warmup_prompt_template if n_steps < min_steps else hot_prompt_template
        )
        chain = prompt_template | llm
        response = chain.invoke(state.messages)  # type: ignore[arg-type]
        return {"messages": [AIMessage(response.content)]}  # type: ignore[attr-defined]

    def answer(state: State) -> dict[str, list[BaseMessage]]:
        nonlocal n_steps
        n_steps = 0
        msg = HumanMessage(
            "Please provide the final answer based on your reasoning above.",
        )
        state.messages.append(msg)
        response = llm.invoke(state.messages)
        return {"messages": [AIMessage(response.content)]}  # type: ignore[attr-defined]

    graph_builder = StateGraph(State)
    graph_builder.add_node(REASON_NODE_NAME, reason)
    graph_builder.add_node(ANSWER_NODE_NAME, answer)
    graph_builder.add_edge(START, REASON_NODE_NAME)
    graph_builder.add_conditional_edges(
        REASON_NODE_NAME,
        route_answer,
        {
            NextAction.CONTINUE: REASON_NODE_NAME,
            NextAction.FINAL_ANSWER: ANSWER_NODE_NAME,
        },
    )
    graph_builder.add_edge(ANSWER_NODE_NAME, END)

    return graph_builder.compile()
