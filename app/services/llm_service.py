
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser

from app.config import Settings
from app.prompts.finance_prompt import FinancePrompts
from app.models.schemas import FinanceAnswer
from app.services.vector_store_service import VectorStoreService


class LLM:
    """Creates the ChatGroq model and builds chains for finance queries."""

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreService,
    ):
        settings.validate()

        # 1. Initialize the Groq LLM
        self.llm = ChatGroq(
            model=settings.llm_model,
            api_key=settings.api_key,
            base_url=settings.llm_base_url,
            temperature=settings.temperature,
        )

        # 2. Inject the vector store
        self.vector_store = vector_store

        # 3. Build the normal chat chain
        self.prompt = FinancePrompts.chat_prompt()
        self.chain = self.prompt | self.llm | StrOutputParser()

        # 4. Build the RAG chain
        self.rag_prompt = FinancePrompts.rag_prompt()
        self.rag_chain = (
            self.rag_prompt
            | self.llm
            | StrOutputParser()
        )

        # 5. Build the structured-output chain
        self.structured_chain = self.build_structured_chain()

    def build_structured_chain(self):
        """Build a chain that returns a FinanceAnswer object."""
        structured_llm = self.llm.with_structured_output(FinanceAnswer)
        return self.prompt | structured_llm

    def tool_chain(self, tools: list):
        """Bind tools to the LLM and return a chain."""
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.llm_with_tools = self.llm.bind_tools(tools)
        return self.prompt | self.llm_with_tools

    async def ask(
        self,
        messages: list[BaseMessage],
        k: int = 3,
    ) -> str:
        """Search Chroma, build context, and generate an answer."""

        # Step 1: Extract the latest user question
        question = next(
            (
                str(message.content)
                for message in reversed(messages)
                if message.type == "human"
            ),
            "",
        )

        if not question:
            raise ValueError("No user question found in messages.")

        # Step 2: Retrieve relevant chunks from Chroma
        documents = self.vector_store.search(question, k=k)

        # Step 3: Combine retrieved chunks into context
        context = "\n\n".join(
            (
                f"Source: {doc.metadata.get('source', 'unknown')}\n"
                f"Content: {doc.page_content}"
            )
            for doc in documents
        )

        if not context:
            context = "No relevant information was found in the knowledge base."

        # Step 4: Send context and conversation to the RAG prompt
        prompt_input = {
            "context": context,
            "messages": messages,
        }

        # Step 5: Call Groq and return the answer
        return await self.rag_chain.ainvoke(prompt_input)

    async def ask_structured_output(
        self,
        messages: list[BaseMessage],
    ) -> FinanceAnswer:
        """Generate a structured FinanceAnswer response."""
        return await self.structured_chain.ainvoke(
            {"messages": messages}
        )

    async def ask_without_retrieval(
        self,
        messages: list[BaseMessage],
    ) -> str:
        """Answer using the normal chat chain without Chroma retrieval."""
        return await self.chain.ainvoke({"messages": messages})


    async def ask_with_tools(
        self,
        messages: list[BaseMessage],
        max_tool_rounds: int = 5,
    ) -> str:
        """
        Ask the model a question and execute any tools it requests.
        The tool result is sent back to the model for a final answer.
        """
        if not hasattr(self, "llm_with_tools") or self.llm_with_tools is None:
            raise ValueError("No tool chain configured. Call tool_chain() first.")

        conversation = list(messages)

        for _ in range(max_tool_rounds):
            # 1. Format the prompt and call the model with bound tool schemas.
            prompt_value = await self.prompt.ainvoke({"messages": conversation})
            ai_message = await self.llm_with_tools.ainvoke(prompt_value)

            # 2. If the model did not request a tool, return its final response.
            if not ai_message.tool_calls:
                content = ai_message.content
                if isinstance(content, str):
                    return content
                return str(content)

            # 3. Keep the AI message containing its tool-call requests.
            conversation.append(ai_message)

            # 4. Execute each requested tool and append its result.
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                selected_tool = self.tools_by_name.get(tool_name)
                if selected_tool is None:
                    tool_result = f"Tool '{tool_name}' is not available."
                else:
                    try:
                        tool_result = await selected_tool.ainvoke(tool_args)
                    except Exception as exc:
                        tool_result = f"Tool execution failed: {exc}"

                conversation.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id,
                    )
                )

        # Safety limit: don't loop indefinitely if the model keeps requesting tools.
        return "I couldn't complete the request because the tool-call limit was reached."
