import os
import sys
import time
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

load_dotenv()

MINIMAX_API_URL = os.getenv("MINIMAX_API_URL")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MODEL_NAME = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")
MAX_TOKENS = int(os.getenv("QUESTION_AGENT_MAX_TOKENS", "102400"))

class QuestionState(TypedDict):
    question: str
    answer: str


def create_llm() -> ChatAnthropic:
    client_kwargs = {
        "model": MODEL_NAME,
        "base_url": MINIMAX_API_URL,
        "api_key": MINIMAX_API_KEY,
        "temperature": 1.0,
        "max_tokens": MAX_TOKENS,
        "max_retries": 0,
    }

    return ChatAnthropic(**client_kwargs)


def answer_question(state: QuestionState) -> None:
    llm = create_llm()
    response = llm.invoke([
        HumanMessage(content=state["question"])
    ])
    print(response)
    for message in response.content:
        if message["type"] == "text":
            state["answer"] += message["text"]
            print(message["text"], end="\n", flush=True)

def build_agent():
    graph = StateGraph(QuestionState)
    graph.add_node("answer_question", answer_question)
    graph.add_edge(START, "answer_question")
    graph.add_edge("answer_question", END)
    return graph.compile()


agent = build_agent()


def main() -> None:
    state = QuestionState(question="你怎么评价Minimax?", answer="dummy answer")
    agent.invoke(state)

if __name__ == "__main__":
    main()


