"""
Assignment 8: Micro-Coach (On-Demand Streaming)

Goal: Provide a short plan non-streamed, and when `stream=True` deliver
encouraging guidance token-by-token via a callback.
"""

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI  # ✅ required
from langchain_core.prompts import ChatPromptTemplate  # ✅ required
from langchain_core.output_parsers import StrOutputParser  # ✅ required


class PrintTokens:
    """Minimal callback-compatible token printer."""

    # required attrs
    ignore_chain = False
    ignore_llm = False
    ignore_retry = False
    ignore_other = False
    ignore_chat_model = False
    raise_error = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="")

    # add no-op handlers to silence warnings
    def on_chain_start(self, *args, **kwargs): pass
    def on_chain_end(self, *args, **kwargs): pass
    def on_llm_end(self, *args, **kwargs): pass
    def on_chat_model_start(self, *args, **kwargs): pass


class MicroCoach:
    def __init__(self):
        """Store prompt strings and prepare placeholders.

        Provide:
        - `system_prompt` motivating but practical tone
        - `user_prompt` with variables {goal}, {time_available}
        - `self.llm_streaming` and `self.llm_plain` placeholders (None), with TODOs
        - `self.stream_prompt` and `self.plain_prompt` placeholders (None), with TODOs
        """
        self.system_prompt = (
            "You are a supportive micro-coach. Keep plans realistic and brief."
        )
        self.user_prompt = "Goal: {goal}\nTime: {time_available}\nReturn a 3-step plan."

        # TODO: Build prompts and LLMs (streaming and non-streaming)
        self.llm_streaming = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            streaming=True,
        )
        self.llm_plain = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
        )
        self.stream_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.user_prompt),
        ])
        self.plain_prompt = self.stream_prompt
        self.stream_chain = self.stream_prompt | self.llm_streaming | StrOutputParser()
        self.plain_chain = self.plain_prompt | self.llm_plain | StrOutputParser()

    def coach(self, goal: str, time_available: str, stream: bool = False) -> str:
        """Return guidance using streaming or non-streaming path.

        Implement:
        - If `stream=True`, attach a token printer callback and stream output.
        - Else, return a compact non-streamed plan string.
        """
        if stream:
            printer = PrintTokens()
            chain = self.stream_chain.with_config(callbacks=[printer])
            _ = chain.invoke({"goal": goal, "time_available": time_available})
            return ""
        else:
            result = self.plain_chain.invoke(
                {"goal": goal, "time_available": time_available}
            )
            return result


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    coach = MicroCoach()
    try:
        print("\n🏃 Micro-Coach — demo\n" + "-" * 40)
        print(coach.coach("resume drafting", "25 minutes", stream=False))
        print()
        print("\nStreaming example:")
        coach.coach("push-ups habit", "10 minutes", stream=True)
        print()
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()
