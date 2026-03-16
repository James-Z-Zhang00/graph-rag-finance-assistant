"""
HybridAgent pool for the search service.

One HybridAgent instance per session_id — matches the existing AgentManager
pattern from the monolith server. Thread-safe via RLock.
"""

import threading
from graphrag_agent.agents.hybrid_agent import HybridAgent


class HybridAgentPool:

    def __init__(self):
        self._instances: dict[str, HybridAgent] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> HybridAgent:
        with self._lock:
            if session_id not in self._instances:
                self._instances[session_id] = HybridAgent()
            return self._instances[session_id]

    def close_all(self):
        with self._lock:
            for agent in self._instances.values():
                try:
                    agent.close()
                except Exception as e:
                    print(f"Error closing agent: {e}")
            self._instances.clear()


agent_pool = HybridAgentPool()
