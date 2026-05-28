"""
NaiveAgent pool for the search service.

One NaiveAgent instance per session_id. Thread-safe via RLock.
"""

import threading
from graphrag_agent.agents.naive_agent import NaiveAgent


class NaiveAgentPool:

    def __init__(self):
        self._instances: dict[str, NaiveAgent] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> NaiveAgent:
        with self._lock:
            if session_id not in self._instances:
                self._instances[session_id] = NaiveAgent()
            return self._instances[session_id]

    def close_all(self):
        with self._lock:
            for agent in self._instances.values():
                try:
                    agent.close()
                except Exception as e:
                    print(f"Error closing naive agent: {e}")
            self._instances.clear()


naive_agent_pool = NaiveAgentPool()
