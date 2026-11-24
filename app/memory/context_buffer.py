from collections import deque
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage

class ContextBuffer:

    """Remembers the last N conversations"""

    def __init__(self, max_messages: int = 10):
        self._messages : deque[BaseMessage] = deque(maxlen=max_messages)

    def add_user_message(self, text: str) -> None:
        self._messages.append(HumanMessage(content=text))

    def add_ai_message(self, text: str) -> None:
        self._messages.append(AIMessage(content=text))

    def get_messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


class SessionStore:
    """Keeps one ContextBuffer per session_id (one per user/chat)."""

    def __init__(self, max_messages: int = 10):
        self._max_messages = max_messages
        self._buffers: dict[str, ContextBuffer] = {}

    def get(self, session_id: str) -> ContextBuffer:
        if session_id not in self._buffers:
            self._buffers[session_id] = ContextBuffer(self._max_messages)
        return self._buffers[session_id]