"""
Assignment 6: Reply Macro Composer — Runtime Configs

Goal: Compose short, consistent reply macros from a customer message and context.

Implement bodies according to docstrings. Prefer small, composposable helpers.
Use runtime configs (`.bind`, `.with_config`) to adjust tone and length.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class MacroComposer:
    """Compose reply macros with configurable tone and length.

    Methods here intentionally raise NotImplementedError to be implemented by students.
    """

    def __init__(self):
        """Initialize any state; prepare prompt strings.

        Requirements:
        - Define a `system_prompt` string describing style (polite, frictionless, concise).
        - Define a `user_prompt` string with variables: {message}, {context}, {style_hint}.
        - Do not build ChatPromptTemplate here; keep only strings and TODOs.
        """
        self.system_prompt = "You craft helpful, concise support macros that sound friendly and professional."
        self.user_prompt = (
            "Customer message:\n{message}\n\nContext:\n{context}\n\nStyle hint: {style_hint}\n"
            "Return a ready-to-send macro with greeting and sign-off."
        )
        # TODO: Create ChatPromptTemplate using the above strings and store as self.prompt
        # self.prompt = ChatPromptTemplate.from_messages([...])
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.user_prompt),
        ])

        # TODO: Create a base ChatOpenAI LLM (low temperature). Store as self.llm
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    def compose_macro(
        self, message: str, context: str, style_hint: str = "neutral"
    ) -> str:
        """Return a polished macro.

        Implement:
        - Bind runtime parameters (e.g., max_tokens, temperature) via `.bind` or `.with_config`.
        - Connect `self.prompt | self.llm | StrOutputParser()`.
        - Invoke with `{"message": message, "context": context, "style_hint": style_hint}`.
        - Return the string content.
        """
        # Runtime config
        configured_llm = self.llm.with_config({"temperature": 0.3}).bind(max_tokens=200)

        chain = self.prompt | configured_llm | StrOutputParser()

        result = chain.invoke({
            "message": message,
            "context": context,
            "style_hint": style_hint
        })

        return result

    def compose_bulk(
        self, items: List[Dict[str, str]], style_hint: str = "neutral"
    ) -> List[str]:
        """Batch-compose macros for many items.

        Implement:
        - Use the same chain as `compose_macro` but with `.batch` for parallelism.
        - Each item has keys: message, context.
        - Return list of strings in same order.
        """
        configured_llm = self.llm.with_config({"temperature": 0.3}).bind(max_tokens=200)
        chain = self.prompt | configured_llm | StrOutputParser()

        # Prepare input list
        payloads = [
            {
                "message": it["message"],
                "context": it["context"],
                "style_hint": style_hint
            }
            for it in items
        ]

        results = chain.batch(payloads)
        return results


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    mc = MacroComposer()
    try:
        print("\n Macro Composer — demo\n" + "-" * 40)
        print(
            mc.compose_macro(
                "My package arrived damaged. What can I do?",
                context="Order #123, policy: refund or replacement within 30 days.",
                style_hint="warm",
            )
        )
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()
