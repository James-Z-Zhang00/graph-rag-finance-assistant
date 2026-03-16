from typing import Annotated, Sequence, TypedDict, List, Dict, Any, AsyncGenerator, Optional
from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
import pprint
import time
import asyncio

from graphrag_agent.models.get_models import get_llm_model, get_stream_llm_model, get_embeddings_model
from graphrag_agent.cache_manager.manager import (
    CacheManager,
    ContextAwareCacheKeyStrategy,
    HybridCacheBackend
)
from graphrag_agent.cache_manager.strategies.global_strategy import GlobalCacheKeyStrategy
from graphrag_agent.config.settings import AGENT_SETTINGS

class BaseAgent(ABC):
    """Agent base class defining common functionality and interfaces"""

    def __init__(self, cache_dir="./cache", memory_only=False):
        """
        Initialize search tools

        Args:
            cache_dir: Cache directory for storing search results
        """
        # Initialize standard LLM and streaming LLM
        self.llm = get_llm_model()
        self.stream_llm = get_stream_llm_model()
        self.embeddings = get_embeddings_model()
        self.default_recursion_limit = AGENT_SETTINGS["default_recursion_limit"]
        self.stream_flush_threshold = AGENT_SETTINGS["stream_flush_threshold"]
        self.deep_stream_flush_threshold = AGENT_SETTINGS["deep_stream_flush_threshold"]
        self.fusion_stream_flush_threshold = AGENT_SETTINGS["fusion_stream_flush_threshold"]
        self.chunk_size = AGENT_SETTINGS["chunk_size"]

        self.memory = MemorySaver()
        self.execution_log = []

        # Standard context-aware cache (within session)
        self.cache_manager = CacheManager(
            key_strategy=ContextAwareCacheKeyStrategy(),
            storage_backend=HybridCacheBackend(
                cache_dir=cache_dir,
                memory_max_size=200,
                disk_max_size=2000
            ) if not memory_only else None,
            cache_dir=cache_dir,
            memory_only=memory_only
        )

        # Global cache (cross-session)
        self.global_cache_manager = CacheManager(
            key_strategy=GlobalCacheKeyStrategy(),
            storage_backend=HybridCacheBackend(
                cache_dir=f"{cache_dir}/global",
                memory_max_size=500,
                disk_max_size=5000
            ) if not memory_only else None,
            cache_dir=f"{cache_dir}/global",
            memory_only=memory_only
        )

        self.performance_metrics = {}  # Performance metrics collection

        # Initialize tools
        self.tools = self._setup_tools()

        # Set up workflow graph
        self._setup_graph()

    @abstractmethod
    def _setup_tools(self) -> List:
        """Set up tools; subclasses must implement"""
        pass

    def _setup_graph(self):
        """Set up workflow graph - base structure; subclasses can customize via _add_retrieval_edges"""
        # Define state type
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        # Create workflow graph
        workflow = StateGraph(AgentState)

        # Add nodes - consistent with original code
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("retrieve", ToolNode(self.tools))
        workflow.add_node("generate", self._generate_node)

        # Add edge from start to agent
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )

        # Add edges from retrieval to generation - logic implemented by subclasses
        self._add_retrieval_edges(workflow)

        # From generation to end
        workflow.add_edge("generate", END)

        # Compile graph
        self.graph = workflow.compile(checkpointer=self.memory)

    async def _stream_process(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Default implementation for streaming processing

        Subclasses should override this method to implement specific streaming logic

        Args:
            inputs: Input messages
            config: Configuration

        Returns:
            AsyncGenerator[str, None]: Streaming response generator
        """
        # Get messages
        messages = inputs.get("messages", [])
        query = messages[-1].content if messages else ""

        # Build state dictionary
        state = {
            "messages": messages,
            "configurable": config.get("configurable", {})
        }

        # Get generation result
        result = await self._generate_node_async(state)

        if "messages" in result and result["messages"]:
            message = result["messages"][0]
            raw = message.content if hasattr(message, "content") else None
            content = raw if raw is not None else str(message)

            # Split by sentences or paragraphs for more natural chunking
            import re
            chunks = re.split(r'([.!?。！？]\s*)', content)
            buffer = ""

            for i in range(0, len(chunks)):
                if i < len(chunks):
                    buffer += chunks[i]

                    # Output when buffer contains a complete sentence or reaches reasonable size
                    if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                        yield buffer
                        buffer = ""
                        await asyncio.sleep(0.01)  # Small delay for smooth display

            # Output any remaining content
            if buffer:
                yield buffer
        else:
            yield "Unable to generate response."


    @abstractmethod
    def _add_retrieval_edges(self, workflow):
        """Add edges from retrieval to generation; subclasses must implement"""
        pass

    def _log_execution(self, node_name: str, input_data: Any, output_data: Any):
        """Log node execution"""
        self.execution_log.append({
            "node": node_name,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data
        })

    def _log_performance(self, operation, metrics):
        """Log performance metrics"""
        self.performance_metrics[operation] = {
            "timestamp": time.time(),
            **metrics
        }

        # Output key performance metrics
        if "duration" in metrics:
            print(f"Performance metric - {operation}: {metrics['duration']:.4f}s")

    def _agent_node(self, state):
        """Agent node logic"""
        messages = state["messages"]

        # Extract keywords to optimize query
        if len(messages) > 0 and isinstance(messages[-1], HumanMessage):
            query = messages[-1].content
            keywords = self._extract_keywords(query)

            # Log keywords
            self._log_execution("extract_keywords", query, keywords)

            # Enhance message with keyword information
            if keywords:
                # Create a new message with keyword metadata
                enhanced_message = HumanMessage(
                    content=query,
                    additional_kwargs={"keywords": keywords}
                )
                # Replace original message
                messages = messages[:-1] + [enhanced_message]

        # Use tools to process request
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)

        self._log_execution("agent", messages, response)
        return {"messages": [response]}

    @abstractmethod
    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """Extract query keywords; subclasses must implement"""
        pass

    @abstractmethod
    def _generate_node(self, state):
        """Generate answer node logic; subclasses must implement"""
        pass

    async def _generate_node_stream(self, state):
        """
        Streaming version of the generate answer node logic

        Args:
            state: Current state

        Returns:
            AsyncGenerator[str, None]: Streaming response generator
        """
        # Default implementation - should be overridden by subclasses
        result = self._generate_node(state)
        if "messages" in result and result["messages"]:
            message = result["messages"][0]
            content = message.content if hasattr(message, "content") else str(message)

            # Simulate streaming output
            for i in range(0, len(content), self.chunk_size):
                yield content[i:i+self.chunk_size]
                await asyncio.sleep(0.01)

    async def _generate_node_async(self, state):
        """
        Async version of the generate answer node logic

        Args:
            state: Current state

        Returns:
            Dict: Result dictionary containing messages
        """
        # This default implementation simply calls the sync version
        # Subclasses should provide a true async implementation
        def sync_generate():
            return self._generate_node(state)

        # Run sync code in thread pool to avoid blocking the event loop
        return await asyncio.get_event_loop().run_in_executor(None, sync_generate)

    def check_fast_cache(self, query: str, thread_id: str = "default") -> str:
        """Dedicated fast cache check method for high-performance paths"""
        start_time = time.time()

        # Extract keywords to ensure they're used in the cache key
        keywords = self._extract_keywords(query)
        cache_params = {
            "thread_id": thread_id,
            "low_level_keywords": keywords.get("low_level", []),
            "high_level_keywords": keywords.get("high_level", [])
        }

        # Use cache manager's fast get method, passing relevant parameters
        result = self.cache_manager.get_fast(query, **cache_params)
        duration = time.time() - start_time
        self._log_performance("fast_cache_check", {
            "duration": duration,
            "hit": result is not None
        })

        return result

    def _check_all_caches(self, query: str, thread_id: str = "default"):
        """Consolidated cache check method"""
        cache_check_start = time.time()

        # 1. First try global cache (cross-session cache)
        global_result = self.global_cache_manager.get(query)
        if global_result:
            print(f"Global cache hit: {query[:30]}...")

            cache_time = time.time() - cache_check_start
            self._log_performance("cache_check", {
                "duration": cache_time,
                "type": "global"
            })

            return global_result

        # 2. Try fast path - high-quality cache skipping validation
        fast_result = self.check_fast_cache(query, thread_id)
        if fast_result:
            print(f"Fast path cache hit: {query[:30]}...")

            # Sync hit content to global cache
            self.global_cache_manager.set(query, fast_result)

            cache_time = time.time() - cache_check_start
            self._log_performance("cache_check", {
                "duration": cache_time,
                "type": "fast"
            })

            return fast_result

        # 3. Try standard cache path with optimized validation
        cached_response = self.cache_manager.get(query, skip_validation=True, thread_id=thread_id)
        if cached_response:
            print(f"Standard cache hit, skipping validation: {query[:30]}...")

            # Sync hit content to global cache
            self.global_cache_manager.set(query, cached_response)

            cache_time = time.time() - cache_check_start
            self._log_performance("cache_check", {
                "duration": cache_time,
                "type": "standard"
            })

            return cached_response

        # No cache hit
        cache_time = time.time() - cache_check_start
        self._log_performance("cache_check", {
            "duration": cache_time,
            "type": "miss"
        })

        return None

    def ask_with_trace(self, query: str, thread_id: str = "default", recursion_limit: Optional[int] = None) -> Dict:
        """Execute query and get answer with execution trace"""
        overall_start = time.time()
        self.execution_log = []  # Reset execution log
        recursion_limit = (
            recursion_limit
            if recursion_limit is not None
            else self.default_recursion_limit
        )

        # Ensure query string is clean
        safe_query = query.strip()

        # First try global cache (cross-session cache)
        global_cache_start = time.time()
        global_result = self.global_cache_manager.get(safe_query)
        global_cache_time = time.time() - global_cache_start

        if global_result:
            print(f"Global cache hit: {safe_query[:30]}... ({global_cache_time:.4f}s)")

            return {
                "answer": global_result,
                "execution_log": [{"node": "global_cache_hit", "timestamp": time.time(), "input": safe_query, "output": "Global cache hit"}]
            }

        # First try fast path - high-quality cache skipping validation
        fast_cache_start = time.time()
        fast_result = self.check_fast_cache(safe_query, thread_id)
        fast_cache_time = time.time() - fast_cache_start

        if fast_result:
            print(f"Fast path cache hit: {safe_query[:30]}... ({fast_cache_time:.4f}s)")

            # Sync hit content to global cache
            self.global_cache_manager.set(safe_query, fast_result)

            return {
                "answer": fast_result,
                "execution_log": [{"node": "fast_cache_hit", "timestamp": time.time(), "input": safe_query, "output": "High-quality cache hit"}]
            }

        # Try standard cache path
        cache_start = time.time()
        cached_response = self.cache_manager.get(safe_query, thread_id=thread_id)
        cache_time = time.time() - cache_start

        if cached_response:
            print(f"Full Q&A cache hit: {safe_query[:30]}... ({cache_time:.4f}s)")

            # Sync hit content to global cache
            self.global_cache_manager.set(safe_query, cached_response)

            return {
                "answer": cached_response,
                "execution_log": [{"node": "cache_hit", "timestamp": time.time(), "input": safe_query, "output": "Standard cache hit"}]
            }

        # Cache miss, execute standard flow
        process_start = time.time()

        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_limit
            }
        }

        inputs = {"messages": [HumanMessage(content=query)]}
        try:
            # Execute full processing flow
            for output in self.graph.stream(inputs, config=config):
                pprint.pprint(f"Output from node '{list(output.keys())[0]}':")
                pprint.pprint("---")
                pprint.pprint(output, indent=2, width=80, depth=None)
                pprint.pprint("\n---\n")

            chat_history = self.memory.get(config)["channel_values"]["messages"]
            answer = chat_history[-1].content

            # Cache results - update both session cache and global cache
            if answer and len(answer) > 10:
                # Update session cache
                self.cache_manager.set(safe_query, answer, thread_id=thread_id)
                # Update global cache
                self.global_cache_manager.set(safe_query, answer)

            process_time = time.time() - process_start
            print(f"Full processing time: {process_time:.4f}s")

            overall_time = time.time() - overall_start
            self._log_performance("ask_with_trace", {
                "total_duration": overall_time,
                "cache_check": cache_time,
                "processing": process_time
            })

            return {
                "answer": answer,
                "execution_log": self.execution_log
            }
        except Exception as e:
            error_time = time.time() - process_start
            print(f"Error processing query: {e} ({error_time:.4f}s)")
            return {
                "answer": f"Sorry, an error occurred while processing your question. Please try again later or rephrase your question. Error details: {str(e)}",
                "execution_log": self.execution_log + [{"node": "error", "timestamp": time.time(), "input": query, "output": str(e)}]
            }

    def ask(self, query: str, thread_id: str = "default", recursion_limit: Optional[int] = None):
        """Ask the agent a question"""
        overall_start = time.time()

        # Ensure query string is clean
        safe_query = query.strip()

        cached_result = self._check_all_caches(safe_query, thread_id)
        if cached_result:
            return cached_result

        # Cache miss, execute standard flow
        process_start = time.time()

        recursion_value = (
            recursion_limit
            if recursion_limit is not None
            else self.default_recursion_limit
        )

        # Process request normally
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_value
            }
        }

        inputs = {"messages": [HumanMessage(content=query)]}
        try:
            for output in self.graph.stream(inputs, config=config):
                pass

            chat_history = self.memory.get(config)["channel_values"]["messages"]
            answer = chat_history[-1].content

            # Cache results - update both session cache and global cache
            if answer and len(answer) > 10:
                # Update session cache
                self.cache_manager.set(safe_query, answer, thread_id=thread_id)
                # Update global cache
                self.global_cache_manager.set(safe_query, answer)

            process_time = time.time() - process_start
            overall_time = time.time() - overall_start

            self._log_performance("ask", {
                "total_duration": overall_time,
                "cache_check": 0,  # Logged by _check_all_caches
                "processing": process_time
            })

            return answer
        except Exception as e:
            error_time = time.time() - process_start
            print(f"Error processing query: {e} ({error_time:.4f}s)")
            return f"Sorry, an error occurred while processing your question. Please try again later or rephrase your question. Error details: {str(e)}"

    async def ask_stream(self, query: str, thread_id: str = "default", recursion_limit: Optional[int] = None) -> AsyncGenerator[str, None]:
        """
        Ask the agent a question and return a streaming response

        Args:
            query: User question
            thread_id: Session ID
            recursion_limit: Recursion limit

        Returns:
            AsyncGenerator[str, None]: Streaming response generator
        """
        overall_start = time.time()

        # Ensure query string is clean
        safe_query = query.strip()

        # First try global cache (cross-session cache)
        global_result = self.global_cache_manager.get(safe_query)
        if global_result and not isinstance(global_result, str):
            global_result = None
        if global_result:
            # For cached responses, return in natural language unit chunks
            import re
            chunks = re.split(r'([.!?。！？]\s*)', global_result)
            buffer = ""

            for i in range(0, len(chunks)):
                buffer += chunks[i]

                # Output when buffer contains a complete sentence or reaches reasonable size
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)

            # Output any remaining content
            if buffer:
                yield buffer
            return

        # First try fast path - high-quality cache skipping validation
        fast_result = self.check_fast_cache(safe_query, thread_id)
        if fast_result and not isinstance(fast_result, str):
            fast_result = None
        if fast_result:
            # For cached responses, return in natural language unit chunks
            import re
            chunks = re.split(r'([.!?。！？]\s*)', fast_result)
            buffer = ""

            for i in range(0, len(chunks)):
                buffer += chunks[i]

                # Output when buffer contains a complete sentence or reaches reasonable size
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)

            # Output any remaining content
            if buffer:
                yield buffer

            # Sync hit content to global cache
            self.global_cache_manager.set(safe_query, fast_result)
            return

        # Try standard cache path
        cache_start = time.time()
        cached_response = self.cache_manager.get(safe_query, thread_id=thread_id)
        cache_time = time.time() - cache_start

        if cached_response and not isinstance(cached_response, str):
            cached_response = None
        if cached_response:
            # Similarly chunk by natural language units
            import re
            chunks = re.split(r'([.!?。！？]\s*)', cached_response)
            buffer = ""

            for i in range(0, len(chunks)):
                buffer += chunks[i]

                # Output when buffer contains a complete sentence or reaches reasonable size
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)

            # Output any remaining content
            if buffer:
                yield buffer

            # Sync hit content to global cache
            self.global_cache_manager.set(safe_query, cached_response)
            return

        # Cache miss, execute standard flow
        recursion_value = (
            recursion_limit
            if recursion_limit is not None
            else self.default_recursion_limit
        )
        config = {
            "configurable": {
                "thread_id": thread_id,
                "recursion_limit": recursion_value,
                "stream_mode": True  # Indicate streaming output mode
            }
        }

        inputs = {"messages": [HumanMessage(content=query)]}
        answer = ""

        try:
            # Execute streaming processing
            async for chunk in self._stream_process(inputs, config):
                yield chunk
                answer += chunk

            # Cache complete answer - update both session cache and global cache
            if answer and len(answer) > 10:
                # Update session cache
                self.cache_manager.set(safe_query, answer, thread_id=thread_id)
                # Update global cache
                self.global_cache_manager.set(safe_query, answer)

            process_time = time.time() - overall_start
            self._log_performance("ask_stream", {
                "total_duration": process_time,
                "processing": process_time
            })

        except Exception as e:
            error_time = time.time() - overall_start
            error_msg = f"Error processing query: {str(e)} ({error_time:.4f}s)"
            print(error_msg)
            yield error_msg

    def mark_answer_quality(self, query: str, is_positive: bool, thread_id: str = "default"):
        """Mark answer quality for cache quality control"""
        start_time = time.time()

        # Extract keywords
        keywords = self._extract_keywords(query)
        cache_params = {
            "thread_id": thread_id,
            "low_level_keywords": keywords.get("low_level", []),
            "high_level_keywords": keywords.get("high_level", [])
        }

        # Call cache manager's quality marking method, passing relevant parameters
        marked = self.cache_manager.mark_quality(query.strip(), is_positive, **cache_params)

        mark_time = time.time() - start_time
        self._log_performance("mark_quality", {
            "duration": mark_time,
            "is_positive": is_positive
        })

    def clear_cache_for_query(self, query: str, thread_id: str = "default"):
        """
        Clear cache for a specific query (both session cache and global cache)

        Args:
            query: Query string
            thread_id: Session ID

        Returns:
            bool: Whether deletion was successful
        """
        # Clear session cache
        success = False

        try:
            # Try to remove possible prefix
            clean_query = query.strip()
            if ":" in clean_query:
                parts = clean_query.split(":", 1)
                if len(parts) > 1:
                    clean_query = parts[1].strip()

            # Clear session cache for original query
            session_cache_deleted = self.cache_manager.delete(query.strip(), thread_id=thread_id)
            success = session_cache_deleted

            # Clear cache for query without prefix
            if clean_query != query.strip():
                self.cache_manager.delete(clean_query, thread_id=thread_id)

            # Clear prefixed query cache variants
            prefixes = ["generate:", "deep:", "query:"]
            for prefix in prefixes:
                self.cache_manager.delete(f"{prefix}{clean_query}", thread_id=thread_id)

            # Clear global cache - using all possible variants
            if hasattr(self, 'global_cache_manager'):
                # Delete original query
                global_cache_deleted = self.global_cache_manager.delete(query.strip())
                success = success or global_cache_deleted

                # Delete cleaned query
                if clean_query != query.strip():
                    self.global_cache_manager.delete(clean_query)

                # Delete prefixed query variants
                for prefix in prefixes:
                    self.global_cache_manager.delete(f"{prefix}{clean_query}")

            # Force flush cache writes
            if hasattr(self.cache_manager.storage, '_flush_write_queue'):
                self.cache_manager.storage._flush_write_queue()

            if hasattr(self, 'global_cache_manager') and hasattr(self.global_cache_manager.storage, '_flush_write_queue'):
                self.global_cache_manager.storage._flush_write_queue()

            # Log
            print(f"Cleared query cache: {query.strip()}")

            return success
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False

    def _validate_answer(self, query: str, answer: str, thread_id: str = "default") -> bool:
        """Validate answer quality"""

        # Use cache manager's validation method
        def validator(query, answer):
            # Basic check - length
            if len(answer) < 20:
                return False

            # Check for error messages
            error_patterns = [
                "Sorry, an error occurred while processing your question",
                "Technical reason:",
                "Unable to retrieve",
                "Unable to answer this question"
            ]

            for pattern in error_patterns:
                if pattern in answer:
                    return False

            # Relevance check - check if question keywords appear in the answer
            keywords = self._extract_keywords(query)
            if keywords:
                low_level_keywords = keywords.get("low_level", [])
                if low_level_keywords:
                    # At least one low-level keyword should appear in the answer
                    keyword_found = any(keyword.lower() in answer.lower() for keyword in low_level_keywords)
                    if not keyword_found:
                        return False

            # Passed all checks
            return True

        return self.cache_manager.validate_answer(query, answer, validator, thread_id=thread_id)

    def close(self):
        """Close resources"""
        # Ensure all deferred cache writes are saved
        if hasattr(self.cache_manager.storage, '_flush_write_queue'):
            self.cache_manager.storage._flush_write_queue()

        # Also ensure global cache writes are saved
        if hasattr(self.global_cache_manager.storage, '_flush_write_queue'):
            self.global_cache_manager.storage._flush_write_queue()
