from typing import Annotated, Sequence, TypedDict, List, Dict, Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
import re

from graphrag_agent.config.prompts import (
    LC_SYSTEM_PROMPT,
    HYBRID_AGENT_GENERATE_PROMPT,
)
from graphrag_agent.config.settings import response_type
from graphrag_agent.search.tool.hybrid_tool import HybridSearchTool
from graphrag_agent.agents.base import BaseAgent
from graphrag_agent.compliance import PIIMasker, AuditLogger, HallucinationValidator
from langgraph.graph.message import add_messages

# Grounding-enforced regeneration prompt — used when faithfulness score < threshold
GROUNDED_GENERATE_PROMPT = """\
Using ONLY the information explicitly provided in the context below, answer the question.
Do NOT include any information not directly stated in the context.
If the context does not contain sufficient information to answer the question fully,
say: "Based on the available context, I cannot provide a complete answer."

Context:
{context}

Question: {question}

Response type: {response_type}"""


class HybridAgent(BaseAgent):
    """Agent implementation using hybrid search with enterprise compliance layer."""

    def __init__(self, enable_mcp: bool = False):
        # Initialize hybrid search tool
        self.search_tool = HybridSearchTool()

        # Whether to enable MCP tools
        self.enable_mcp = enable_mcp

        # First initialize base attributes
        self.cache_dir = "./cache/hybrid_agent"

        # Call parent constructor - using default ContextAwareCacheKeyStrategy
        super().__init__(cache_dir=self.cache_dir)

        # --- Compliance components ---
        self._pii_masker = PIIMasker()
        self.audit_logger = AuditLogger()
        self._validator = HallucinationValidator(llm=self.llm)

        # Per-request compliance state (safe: each agent instance is single-threaded)
        self._last_citations: List[str] = []
        self._last_quality_score: Optional[float] = None
        self._last_audit_id: Optional[str] = None
        self._last_contexts: List[str] = []

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

    def _setup_graph(self):
        """Override base graph setup to insert the validate node after generate."""
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._agent_node)
        workflow.add_node("retrieve", ToolNode(self.tools))
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("validate", self._validate_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "retrieve", END: END},
        )
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_edge("validate", END)

        self.graph = workflow.compile(checkpointer=self.memory)

    def _add_retrieval_edges(self, workflow):
        """Not used — _setup_graph is fully overridden."""
        pass

    # ── compliance helpers ──────────────────────────────────────────────────

    def _build_citations(self) -> List[str]:
        """Build a list of citation strings from the last search tool results."""
        return [r.get_citation("default") for r in self.search_tool.get_last_results()]

    # ── LangGraph nodes ─────────────────────────────────────────────────────

    def _generate_node(self, state):
        """Generate answer node with citation collection and audit logging."""
        messages = state["messages"]
        thread_id = state.get("configurable", {}).get("thread_id", "default")

        # Safely get question content
        try:
            question = messages[-3].content if len(messages) >= 3 else "Question not found"
        except Exception:
            question = "Unable to retrieve question"

        # Safely get document content (tool message from retrieve node)
        try:
            docs = messages[-1].content if messages[-1] else "No relevant information found"
        except Exception:
            docs = "Unable to retrieve search results"

        # --- Citation collection ---
        self._last_citations = self._build_citations()

        # --- Context capture for evaluation ---
        self._last_contexts = [docs] if docs and docs != "No relevant information found" else []

        # --- Audit: retrieval done ---
        last_results = self.search_tool.get_last_results()
        local_count = sum(1 for r in last_results if r.metadata.source_type in ("entity", "relationship", "chunk", "filing_section"))
        global_count = sum(1 for r in last_results if r.metadata.source_type == "community")
        self.audit_logger.log("retrieval_done", thread_id, {
            "result_count": len(last_results),
            "local_count": local_count,
            "global_count": global_count,
        })

        # First try global cache
        global_result = self.global_cache_manager.get(question)
        if global_result:
            self._log_execution("generate",
                            {"question": question, "docs_length": len(docs)},
                            "Global cache hit")
            return {"messages": [AIMessage(content=global_result)]}

        # Then check session cache
        cached_result = self.cache_manager.get(question, thread_id=thread_id)
        if cached_result:
            self._log_execution("generate",
                            {"question": question, "docs_length": len(docs)},
                            "Session cache hit")
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

            # Cache results
            if response and len(response) > 10:
                self.cache_manager.set(question, response, thread_id=thread_id)
                self.global_cache_manager.set(question, response)

            # --- Audit: generation done ---
            self.audit_logger.log("generation_done", thread_id, {
                "answer_length": len(response) if response else 0,
            })

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

    def _validate_node(self, state):
        """
        Faithfulness validation node.

        Scores the generated answer against retrieved context. If score < 0.7,
        regenerates with a strict grounding-enforced prompt.
        Stores quality_score on self._last_quality_score for inclusion in the response.
        """
        messages = state["messages"]
        thread_id = state.get("configurable", {}).get("thread_id", "default")

        # Need at least answer + docs to validate
        if len(messages) < 2:
            return {"messages": []}

        try:
            answer = messages[-1].content if hasattr(messages[-1], "content") else ""
            docs = messages[-2].content if hasattr(messages[-2], "content") else ""
        except Exception:
            return {"messages": []}

        if not answer or not docs:
            return {"messages": []}

        validation = self._validator.validate(answer, docs)
        score = validation.get("score", 0.5)
        passed = validation.get("passed", True)

        self._last_quality_score = score

        self.audit_logger.log("validation_done", thread_id, {
            "quality_score": score,
            "passed": passed,
            "unsupported_claims_count": len(validation.get("unsupported_claims", [])),
        })
        self._log_execution("validate", {"score": score, "passed": passed}, validation)

        if passed:
            return {"messages": []}

        # --- Regenerate with grounding-enforced prompt ---
        try:
            # Get original question (first HumanMessage)
            question = next(
                (m.content for m in messages if isinstance(m, HumanMessage)),
                "Unknown question",
            )

            grounding_prompt = ChatPromptTemplate.from_messages([
                ("system", LC_SYSTEM_PROMPT),
                ("human", GROUNDED_GENERATE_PROMPT),
            ])
            rag_chain = grounding_prompt | self.llm | StrOutputParser()
            regenerated = rag_chain.invoke({
                "context": docs,
                "question": question,
                "response_type": response_type,
            })

            self._log_execution("validate_regenerate",
                                {"score": score, "question": question},
                                regenerated)
            return {"messages": [AIMessage(content=regenerated)]}
        except Exception as e:
            self._log_execution("validate_regenerate_error", {"score": score}, str(e))
            return {"messages": []}

    # ── Public entry point override ─────────────────────────────────────────

    def ask_with_trace(self, query: str, thread_id: str = "default", recursion_limit: Optional[int] = None) -> Dict:
        """
        Execute query with full compliance pipeline:
        1. PII masking on the query
        2. Audit trail: query_received
        3. Base RAG pipeline (retrieve → generate → validate)
        4. Return answer enriched with citations, quality_score, audit_id
        """
        # Reset per-request state
        self._last_citations = []
        self._last_quality_score = None
        self._last_audit_id = None
        self._last_contexts = []

        # Apply PII masking
        masked_query, pii_types = self._pii_masker.mask(query)

        if pii_types:
            self.audit_logger.log("pii_masked", thread_id, {
                "types_found": pii_types,
                "count": len(pii_types),
            })

        audit_id = self.audit_logger.log("query_received", thread_id, {
            "query_masked": masked_query,
            "session_id": thread_id,
        })
        self._last_audit_id = audit_id

        # Run base RAG pipeline with masked query
        result = super().ask_with_trace(masked_query, thread_id, recursion_limit)

        # Attach compliance fields to result
        result["citations"] = self._last_citations
        result["quality_score"] = self._last_quality_score
        result["audit_id"] = audit_id
        result["contexts"] = self._last_contexts

        return result

    # ── Streaming support (unchanged) ───────────────────────────────────────

    def _extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """Extract query keywords"""
        cached_keywords = self.cache_manager.get(f"keywords:{query}")
        if cached_keywords:
            return cached_keywords

        try:
            keywords = self.search_tool.extract_keywords(query)

            if not isinstance(keywords, dict):
                keywords = {}
            if "low_level" not in keywords:
                keywords["low_level"] = []
            if "high_level" not in keywords:
                keywords["high_level"] = []

            self.cache_manager.set(f"keywords:{query}", keywords)
            return keywords
        except Exception as e:
            print(f"Keyword extraction failed: {e}")
            return {"low_level": [], "high_level": []}

    async def _generate_node_stream(self, state):
        """Streaming version of the generate answer node logic"""
        messages = state["messages"]

        try:
            question = messages[-3].content if len(messages) >= 3 else "Question not found"
        except Exception:
            question = "Unable to retrieve question"

        try:
            docs = messages[-1].content if messages[-1] else "No relevant information found"
        except Exception:
            docs = "Unable to retrieve search results"

        thread_id = state.get("configurable", {}).get("thread_id", "default")

        cached_result = self.cache_manager.get(f"generate:{question}", thread_id=thread_id)
        if cached_result and not isinstance(cached_result, str):
            cached_result = None
        if cached_result:
            chunks = re.split(r'([.!?。！？]\s*)', cached_result)
            buffer = ""

            for i in range(0, len(chunks)):
                buffer += chunks[i]
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)

            if buffer:
                yield buffer
            return

        prompt = ChatPromptTemplate.from_messages([
            ("system", LC_SYSTEM_PROMPT),
            ("human", HYBRID_AGENT_GENERATE_PROMPT),
        ])

        rag_chain = prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({
            "context": docs,
            "question": question,
            "response_type": response_type
        })

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
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        query = ""
        if "messages" in inputs and inputs["messages"] and len(inputs["messages"]) > 0:
            last_message = inputs["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                query = last_message.content

        if not query:
            yield "Unable to get query content, please try again."
            return

        cached_response = self.cache_manager.get(query.strip(), thread_id=thread_id)
        if cached_response and not isinstance(cached_response, str):
            cached_response = None
        if cached_response:
            chunks = re.split(r'([.!?。！？]\s*)', cached_response)
            buffer = ""

            for i in range(0, len(chunks)):
                buffer += chunks[i]
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)

            if buffer:
                yield buffer
            return

        workflow_state = {"messages": [HumanMessage(content=query)]}

        yield "**Analyzing question**...\n\n"

        agent_output = await self._agent_node_async(workflow_state)
        workflow_state = {"messages": workflow_state["messages"] + agent_output["messages"]}

        tool_decision = tools_condition(workflow_state)
        if tool_decision == "tools":
            yield "**Retrieving relevant information**...\n\n"

            retrieve_output = await self._retrieve_node_async(workflow_state)
            workflow_state = {"messages": workflow_state["messages"] + retrieve_output["messages"]}

            yield "**Generating answer**...\n\n"

            async for token in self._generate_node_stream(workflow_state):
                yield token
        else:
            final_msg = workflow_state["messages"][-1]
            raw = final_msg.content if hasattr(final_msg, "content") else None
            content = raw if raw is not None else str(final_msg)

            chunks = re.split(r'([.!?。！？]\s*)', content)
            buffer = ""

            for i in range(0, len(chunks)):
                buffer += chunks[i]
                if (i % 2 == 1) or len(buffer) >= self.stream_flush_threshold:
                    yield buffer
                    buffer = ""
                    await asyncio.sleep(0.01)

            if buffer:
                yield buffer

    async def _retrieve_node_async(self, state):
        """Async version of the retrieval node"""
        try:
            last_message = state["messages"][-1]

            tool_calls = []
            if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs:
                tool_calls = last_message.additional_kwargs.get('tool_calls', [])

            if not tool_calls and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_calls = last_message.tool_calls

            if not tool_calls:
                return {
                    "messages": [
                        AIMessage(content="Unable to get query information, please try again.")
                    ]
                }

            tool_call = tool_calls[0]

            query = ""
            tool_id = "tool_call_0"
            tool_name = "search_tool"

            if isinstance(tool_call, dict):
                tool_id = tool_call.get("id", tool_id)

                if "function" in tool_call and isinstance(tool_call["function"], dict):
                    tool_name = tool_call["function"].get("name", tool_name)

                    args = tool_call["function"].get("arguments", {})
                    if isinstance(args, str):
                        try:
                            import json
                            args_dict = json.loads(args)
                            query = args_dict.get("query", "")
                        except:
                            query = args
                    elif isinstance(args, dict):
                        query = args.get("query", "")
                elif "name" in tool_call:
                    tool_name = tool_call.get("name", tool_name)

                if not query and "args" in tool_call:
                    args = tool_call["args"]
                    if isinstance(args, dict):
                        query = args.get("query", "")
                    elif isinstance(args, str):
                        query = args

            if not query and hasattr(last_message, 'content'):
                query = last_message.content

            tool_result = self.search_tool.search(query)

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

        return await asyncio.get_event_loop().run_in_executor(None, sync_agent)

    def _get_tool_call_info(self, message):
        """Extract tool call information from a message"""
        if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
            tool_calls = message.additional_kwargs.get('tool_calls', [])
            if tool_calls and len(tool_calls) > 0:
                tool_call = tool_calls[0]
                return {
                    "id": tool_call.get("id", "tool_call_0"),
                    "name": tool_call.get("function", {}).get("name", "search_tool"),
                    "args": tool_call.get("function", {}).get("arguments", {})
                }

        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            return {
                "id": tool_call.get("id", "tool_call_0"),
                "name": tool_call.get("name", "search_tool"),
                "args": tool_call.get("args", {})
            }

        return {
            "id": "tool_call_0",
            "name": "search_tool",
            "args": {"query": ""}
        }

    def close(self):
        """Close resources"""
        super().close()

        if self.search_tool:
            self.search_tool.close()
