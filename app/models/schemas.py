from typing import Literal
from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    question: str = Field(description="The original user question.")
    answer: str = Field(description="The final answer returned by the model.")


class FinanceAnswer(BaseModel):
    category: Literal["budgeting", "investing", "loans", "saving", "general"] = Field(
        description="The topic area this question belongs to."
    )
    answer: str = Field(description="A short, clear answer to the question.")
    key_points: list[str] = Field(description="2 to 4 short bullet points to remember.")
    needs_calculation: bool = Field(
        description="True if answering properly needs a numeric calculation."
    )