from typing import List, Dict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import tools_condition
import asyncio
import re

from graphrag_agent.config.prompts import (
    LC_SYSTEM_PROMPT,
    HYBRID_AGENT_GENERATE_PROMPT,
)
from graphrag_agent.config.settings import response_type
from graphrag_agent.search.tool.hybrid_tool import HybridSearchTool

from graphrag_agent.agents.base import BaseAgent


class HybridAgent(BaseAgent):
    """Agent implementation using hybrid search"""

    def __init__(self, enable_mcp: bool = False):
        # Initialize hybrid search tool
        self.search_tool = HybridSearchTool()

        # Whether to enable MCP tools
        self.enable_mcp = enable_mcp

        # First initialize base attributes
        self.cache_dir = "./cache/hybrid_agent"

        # Call parent constructor - using default ContextAwareCacheKeyStrategy
        super().__init__(cache_dir=self.cache_dir)

    def _setup_tools(self) -> List:
        """Set up tools"""
        tools = [
            self.search_tool.get_tool(),
            self.search_tool.get_global_tool(),
        ]
        if self.enable_mcp:
            from graphrag_agent.mcp import create_mcp_tools
            tools.extend(create_mcp_tools())
        return tools

    def _add_retrieval_edges(self, workflow):
        """Add edges from retrieval to generation"""
        # Simple direct edge from retrieval to generation
        workflow.add_edge("retrieve", "generate")

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """Extract query keywords"""
        # Check cache
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords

        try:
            # Use enhanced search tool's keyword extraction feature
            keywords = self.search_tool.extract_keywords(query)

            # Ensure valid keyword format is returned
            if not isinstance(keywords, dict):
                keywords = {}
            if "low_level" not in keywords:
                keywords["low_level"] = []
            if "high_level" not in keywords:
                keywords["high_level"] = []

            # Cache result
            self.cache_manager.set(f"keywords:{query}", keywords)

            return keywords
        except Exception as e:
            print(f"Keyword extraction failed: {e}")
            # Return default empty keywords on error
            return {"low_level": [], "high_level": []}

    def _generate_node(self, state):
        """Generate answer node logic"""
        messages = state["messages"]

        # Safely get question content
        try:
            question = messages[-3].content if len(messages) >= 3 else "Question not found"
        except Exception:
            question = "Unable to retrieve question"

        # Safely get document content
        try:
            docs = messages[-1].content if messages[-1] else "No relevant information found"
        except Exception:
            docs = "Unable to retrieve search results"

        # First try global cache
        global_result = self.global_cache_manager.get(question)
        if global_result:
            self._log_execution("generate",
                            {"question": question, "docs_length": len(docs)},
                            "Global cache hit")
            return {"messages": [AIMessage(content=global_result)]}

        # Get current session ID for context-aware caching
        thread_id = state.get("configurable", {}).get("thread_id", "default")

        # Then check session cache
        cached_result = self.cache_manager.get(question, thread_id=thread_id)
        if cached_result:
            self._log_execution("generate",
                            {"question": question, "docs_length": len(docs)},
                            "Session cache hit")
            # Sync hit content to global cache
            self.global_cache_manager.set(question, cached_result)
            return {"messages": [AIMessage(content=cached_result)]}

        prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", HYBRID_AGENT_GENERATE_PROMPT),
        ])

        rag_chain = prompt | self.llm | StrOutputParser()
        try:
            response = rag_chain.invoke({
                "context": docs,
                "question": question,
                "response_type": response_type
            })

            # Cache results - update both session cache and global cache
            if response and len(response) > 10:
                # Update session cache
                self.cache_manager.set(question, response, thread_id=thread_id)
                # Update global cache
                self.global_cache_manager.set(question, response)

            self._log_execution("generate",
                            {"question": question, "docs_length": len(docs)},
                            response)

            return {"messages": [AIMessage(content=response)]}
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            self._log_execution("generate_error",
                            {"question": question, "docs_length": len(docs)},
                            error_msg)
            return {"messages": [AIMessage(content=f"Sorry, I am unable to answer this question. Technical reason: {str(e)}")]}

    async def _generate_node_stream(self, state):
        """Streaming version of the generate answer node logic"""
        messages = state["messages"]

        # Safely get question content
        try:
            question = messages[-3].content if len(messages) >= 3 else "Question not found"
        except Exception:
            question = "Unable to retrieve question"

        # Safely get document content
        try:
            docs = messages[-1].content if messages[-1] else "No relevant information found"
        except Exception:
            docs = "Unable to retrieve search results"

        # Get current session ID
        thread_id = state.get("configurable", {}).get("thread_id", "default")

        # Check cache
        cached_result = self.cache_manager.get(f"generate:{question}", thread_id=thread_id)
        if cached_result and not isinstance(cached_result, str):
            cached_result = None
        if cached_result:
            # Output in sentence chunks
            chunks = re.split(r'([.!?。！？]\s*)', cached_result)
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

        prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", HYBRID_AGENT_GENERATE_PROMPT),
        ])

        # Use streaming model
        # Use sync model to generate complete result directly
        rag_chain = prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({
            "context": docs,
            "question": question,
            "response_type": response_type
        })

        # Output results in chunks
        if response is None:
            response = "Unable to generate a response. Please try rephrasing your question."
        sentences = re.split(r'([.!?。！？]\s*)', response)
        buffer = ""

        for i in range(len(sentences)):
            buffer += sentences[i]
            if i % 2 == 1 or len(buffer) >= self.stream_flush_threshold:
                yield buffer
                buffer = ""
                await asyncio.sleep(0.01)

        if buffer:
            yield buffer

    async def _stream_process(self, inputs, config):
        """Implement streaming processing"""
        # Implementation similar to GraphAgent but tailored for HybridAgent specifics
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        # Safely get query content
        query = ""
        if "messages" in inputs and inputs["messages"] and len(inputs["messages"]) > 0:
            last_message = inputs["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                query = last_message.content

        if not query:
            yield "Unable to get query content, please try again."
            return

        # Cache check and handling same as GraphAgent
        cached_response = self.cache_manager.get(query.strip(), thread_id=thread_id)
        if cached_response and not isinstance(cached_response, str):
            cached_response = None
        if cached_response:
            # For cached responses, return in natural language unit chunks
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
            return

        # Workflow processing same as GraphAgent but with progress prompts
        workflow_state = {"messages": [HumanMessage(content=query)]}

        # Output a processing start prompt
        yield "**Analyzing question**...\n\n"

        # Execute agent node
        agent_output = await self._agent_node_async(workflow_state)
        workflow_state = {"messages": workflow_state["messages"] + agent_output["messages"]}

        # Check if tools are needed
        tool_decision = tools_condition(workflow_state)
        if tool_decision == "tools":
            # Inform user that retrieval is in progress
            yield "**Retrieving relevant information**...\n\n"

            # Execute retrieval node
            retrieve_output = await self._retrieve_node_async(workflow_state)
            workflow_state = {"messages": workflow_state["messages"] + retrieve_output["messages"]}

            # Inform user that answer generation is in progress
            yield "**Generating answer**...\n\n"

            # Stream generate node output
            async for token in self._generate_node_stream(workflow_state):
                yield token
        else:
            # No tools needed, return agent's response directly
            final_msg = workflow_state["messages"][-1]
            raw = final_msg.content if hasattr(final_msg, "content") else None
            content = raw if raw is not None else str(final_msg)

            # Chunk by natural language units
            chunks = re.split(r'([.!?。！？]\s*)', content)
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

    async def _retrieve_node_async(self, state):
        """Async version of the retrieval node"""
        try:
            # Get last message
            last_message = state["messages"][-1]

            # Safely get tool call information
            tool_calls = []

            # Check tool_calls in additional_kwargs
            if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs:
                tool_calls = last_message.additional_kwargs.get('tool_calls', [])

            # Check direct tool_calls attribute
            if not tool_calls and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_calls = last_message.tool_calls

            # If no tool calls found
            if not tool_calls:
                return {
                    "messages": [
                        AIMessage(content="Unable to get query information, please try again.")
                    ]
                }

            # Get first tool call
            tool_call = tool_calls[0]

            # Safely get query
            query = ""
            tool_id = "tool_call_0"
            tool_name = "search_tool"

            # Extract parameters based on tool call format
            if isinstance(tool_call, dict):
                # Extract ID
                tool_id = tool_call.get("id", tool_id)

                # Extract function name
                if "function" in tool_call and isinstance(tool_call["function"], dict):
                    tool_name = tool_call["function"].get("name", tool_name)

                    # Extract arguments
                    args = tool_call["function"].get("arguments", {})
                    if isinstance(args, str):
                        # Try to parse JSON
                        try:
                            import json
                            args_dict = json.loads(args)
                            query = args_dict.get("query", "")
                        except:
                            query = args  # If parsing fails, use entire string as query
                    elif isinstance(args, dict):
                        query = args.get("query", "")
                # Check directly at root level
                elif "name" in tool_call:
                    tool_name = tool_call.get("name", tool_name)

                # Check args field
                if not query and "args" in tool_call:
                    args = tool_call["args"]
                    if isinstance(args, dict):
                        query = args.get("query", "")
                    elif isinstance(args, str):
                        query = args

            # If still no query, try simplest extraction
            if not query and hasattr(last_message, 'content'):
                query = last_message.content

            # Execute search
            tool_result = self.search_tool.search(query)

            # Return properly formatted tool message
            return {
                "messages": [
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                ]
            }
        except Exception as e:
            # Handle error
            error_msg = f"Error processing tool call: {str(e)}"
            print(error_msg)
            return {
                "messages": [
                    AIMessage(content=error_msg)
                ]
            }

    async def _agent_node_async(self, state):
        """Async version of the agent node"""
        def sync_agent():
            return self._agent_node(state)

        # Run sync code in thread pool to avoid blocking the event loop
        return await asyncio.get_event_loop().run_in_executor(None, sync_agent)

    def _get_tool_call_info(self, message):
        """
        Extract tool call information from a message

        Args:
            message: Message containing tool calls

        Returns:
            Dict: Tool call information including id, name, and args
        """
        # Check tool_calls in additional_kwargs
        if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
            tool_calls = message.additional_kwargs.get('tool_calls', [])
            if tool_calls and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                return {
                    "id": tool_call.get("id", "tool_call_0"),
                    "name": tool_call.get("function", {}).get("name", "search_tool"),
                    "args": tool_call.get("function", {}).get("arguments", {})
                }

        # Check direct tool_calls attribute
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            return {
                "id": tool_call.get("id", "tool_call_0"),
                "name": tool_call.get("name", "search_tool"),
                "args": tool_call.get("args", {})
            }

        # Default return
        return {
            "id": "tool_call_0",
            "name": "search_tool",
            "args": {"query": ""}
        }

    def close(self):
        """Close resources"""
        # First close parent resources
        super().close()

        # Then close search tool resources
        if self.search_tool:
            self.search_tool.close()
