import hashlib
from .base import CacheKeyStrategy


class ContextAwareCacheKeyStrategy(CacheKeyStrategy):
    """Context-aware cache key strategy that factors in conversation history."""

    def __init__(self, context_window: int = 3):
        """
        Args:
            context_window: Number of recent conversation turns to include in the key.
        """
        self.context_window = context_window
        self.conversation_history = {}
        self.history_versions = {}

    def update_history(self, query: str, thread_id: str = "default", max_history: int = 10):
        """Update conversation history for a session."""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
            self.history_versions[thread_id] = 0

        self.conversation_history[thread_id].append(query)

        if len(self.conversation_history[thread_id]) > max_history:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_history:]

        # Increment version so key changes when context changes
        self.history_versions[thread_id] += 1

    def generate_key(self, query: str, **kwargs) -> str:
        """Generate a context-aware cache key."""
        thread_id = kwargs.get("thread_id", "default")

        history = self.conversation_history.get(thread_id, [])
        version = self.history_versions.get(thread_id, 0)

        context = " ".join(history[-self.context_window:] if self.context_window > 0 else [])

        combined = f"thread:{thread_id}|ctx:{context}|v{version}|{query}".strip()
        return hashlib.md5(combined.encode('utf-8')).hexdigest()


class ContextAndKeywordAwareCacheKeyStrategy(CacheKeyStrategy):
    """Cache key strategy combining conversation context and keywords."""

    def __init__(self, context_window: int = 3):
        """
        Args:
            context_window: Number of recent conversation turns to include in the key.
        """
        self.context_window = context_window
        self.conversation_history = {}
        self.history_versions = {}

    def update_history(self, query: str, thread_id: str = "default", max_history: int = 10):
        """Update conversation history for a session."""
        if thread_id not in self.conversation_history:
            self.conversation_history[thread_id] = []
            self.history_versions[thread_id] = 0

        self.conversation_history[thread_id].append(query)

        if len(self.conversation_history[thread_id]) > max_history:
            self.conversation_history[thread_id] = self.conversation_history[thread_id][-max_history:]

        self.history_versions[thread_id] += 1

    def generate_key(self, query: str, **kwargs) -> str:
        """Generate a cache key combining context and keywords."""
        thread_id = kwargs.get("thread_id", "default")
        key_parts = [f"thread:{thread_id}", query.strip()]

        history = self.conversation_history.get(thread_id, [])
        version = self.history_versions.get(thread_id, 0)

        if self.context_window > 0 and history:
            context = " ".join(history[-self.context_window:])
            key_parts.append(f"ctx:{hashlib.md5(context.encode('utf-8')).hexdigest()}")

        key_parts.append(f"v:{version}")

        low_level_keywords = kwargs.get("low_level_keywords", [])
        if low_level_keywords:
            key_parts.append("low:" + ",".join(sorted(low_level_keywords)))

        high_level_keywords = kwargs.get("high_level_keywords", [])
        if high_level_keywords:
            key_parts.append("high:" + ",".join(sorted(high_level_keywords)))

        key_str = "||".join(key_parts)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
