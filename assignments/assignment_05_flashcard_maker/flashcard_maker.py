"""
Assignment 5: Flashcard Maker — Structured Outputs with Pydantic

Focus: Use `with_structured_output` to coerce JSON into a Pydantic model.
"""

import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()


class Flashcard(BaseModel):
    """TODO: Define fields for a clean flashcard."""

    term: str = Field(..., description="Short term")
    definition: str = Field(..., description="One-sentence definition")


class FlashcardMaker:
    def __init__(self):
        # TODO: Create an LLM and wrap with structured output to Flashcard
        # self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        # self.structured = self.llm.with_structured_output(Flashcard)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.structured = self.llm.with_structured_output(Flashcard)

    def make_cards(self, topics: List[str]) -> List[Flashcard]:
        """TODO: Generate one card per topic with concise definitions."""
        cards: List[Flashcard] = []
        for t in topics:
            card = self.structured.invoke(
                f"Create a beginner-friendly flashcard about '{t}'."
            )
            cards.append(card)
        return cards


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    maker = FlashcardMaker()
    topics = ["positional encoding", "dropout", "precision vs recall"]
    print("\n Flashcard Maker — demo\n" + "-" * 40)
    for c in maker.make_cards(topics):
        print(f"• {c.term}: {c.definition}")


if __name__ == "__main__":
    _demo()
