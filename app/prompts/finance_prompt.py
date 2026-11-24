from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class FinancePrompts:
    """Holds the system prompt and builds the chat prompt template."""

    SYSTEM_PROMPT = """You are Finance Assistant, a careful and friendly personal-finance helper.

Rules:
1. For ANY calculation (compound interest, EMI, etc.) you MUST call the calculator tools.
   Never do maths in your head.
2. For concepts, definitions and general finance knowledge, call the
   search_finance_knowledge tool and base your answer on what it returns.
3. If a question needs several steps, call tools one after another.
4. Be concise. Show the key numbers clearly.
5. You provide educational information, not personalised financial advice.
   Remind the user of this when giving recommendations.
6. If you do not know something, say so instead of guessing."""

    @classmethod
    def chat_prompt(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
            ("system",cls.SYSTEM_PROMPT ),
            MessagesPlaceholder(variable_name="messages"),
            ]
        )


    RAG_SYSTEM_PROMPT = """You are Finance Assistant, a careful and friendly personal-finance helper.

Answer using ONLY the context below. If the context does not contain the answer,
say you don't have that information. Be concise. You provide educational information,
not personalised financial advice.

Context:
{context}"""

    @classmethod
    def rag_prompt(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", cls.RAG_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )    