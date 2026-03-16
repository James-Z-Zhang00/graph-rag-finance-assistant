import time
import uuid
import requests
import queue
import json
import threading
import time
import streamlit as st
from typing import Dict, Callable
from frontend_config.settings import API_URL, community_algorithm
from utils.performance import monitor_performance

@monitor_performance(endpoint="send_message")
def send_message(message: str) -> Dict:
    """Send chat message to FastAPI backend with performance monitoring"""
    start_time = time.time()
    try:
        # Build request parameters
        params = {
            "message": message,
            "session_id": st.session_state.session_id,
            "agent_type": st.session_state.agent_type
        }

        # If deep research agent, add enhanced tool parameters
        if st.session_state.agent_type == "deep_research_agent":
            params["use_deeper_tool"] = st.session_state.get("use_deeper_tool", True)
            params["show_thinking"] = st.session_state.get("show_thinking", False)

        response = requests.post(
            f"{API_URL}/chat",
            json=params,
            # timeout=120  # Increase timeout
        )
        
        # Record performance
        duration = time.time() - start_time
        print(f"Frontend API call duration: {duration:.4f}s")
        
        # Save performance data in session
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
            
        st.session_state.performance_metrics.append({
            "operation": "send_message",
            "duration": duration,
            "timestamp": time.time(),
            "message_length": len(message)
        })
        
        return response.json()
    except requests.exceptions.RequestException as e:
        # Record error performance
        duration = time.time() - start_time
        print(f"Frontend API call error: {str(e)} ({duration:.4f}s)")
        
        st.error(f"Server connection error: {str(e)}")
        return None

def send_message_stream(message: str, on_token: Callable[[str, bool], None]) -> str:
    """
    Send chat message to FastAPI backend and get streaming response
    
    Args:
        message: Message to send
        on_token: Callback function to process tokens
        
    Returns:
        str: Collected thinking content (if any)
    """
    try:
        # Build request parameters
        params = {
            "message": message,
            "session_id": st.session_state.session_id,
            "agent_type": st.session_state.agent_type
        }

        # If deep research agent, add specific parameters
        if st.session_state.agent_type == "deep_research_agent":
            params["use_deeper_tool"] = st.session_state.get("use_deeper_tool", True)
            params["show_thinking"] = st.session_state.get("show_thinking", False)

        # Setup SSE connection
        import sseclient
        import requests
        import json
        
        # Send request in non-blocking mode
        response = requests.post(
            f"{API_URL}/chat/stream",
            json=params,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        
        # Setup SSE client
        client = sseclient.SSEClient(response)
        
        # Process each event
        thinking_content = ""
        
        for event in client.events():
            try:
                # Ensure all possible exceptions are caught when parsing JSON
                try:
                    data = json.loads(event.data)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {str(e)}, raw data: {event.data[:100]}")
                    continue
                
                # Handle different event types
                if data.get("status") == "token":
                    # Model output token
                    on_token(data.get("content", ""))
                elif data.get("status") == "thinking":
                    # Thinking process chunk
                    chunk = data.get("content", "")
                    thinking_content += chunk
                    on_token(chunk, is_thinking=True)
                elif data.get("status") == "done":
                    # Completion notification
                    break
                elif data.get("status") == "error":
                    # Error notification
                    on_token(f"\n\nError: {data.get('message', 'Unknown error')}")
                    break
                else:
                    # Handle other status types
                    pass
            except Exception as e:
                # Handle any uncaught exceptions
                print(f"Error processing SSE event: {str(e)}")
                continue
        
        # Return collected thinking content for storage
        return thinking_content
    except Exception as e:
        # Handle connection errors
        on_token(f"\n\nConnection error: {str(e)}")
        print(f"Streaming API connection error: {str(e)}")
        return None

@monitor_performance(endpoint="send_feedback")
def send_feedback(message_id: str, query: str, is_positive: bool, thread_id: str, agent_type: str = "hybrid_agent"):
    """Send user feedback to backend"""
    start_time = time.time()
    try:
        # Ensure agent_type has a value
        if not agent_type:
            agent_type = "hybrid_agent"
            
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "message_id": message_id,
                "query": query,
                "is_positive": is_positive,
                "thread_id": thread_id,
                "agent_type": agent_type  # Ensure this field is included in request
            },
            timeout=10
        )
        
        # Record performance
        duration = time.time() - start_time
        print(f"Frontend feedback API call duration: {duration:.4f}s")
        
        # Save performance data in session
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
            
        st.session_state.performance_metrics.append({
            "operation": "send_feedback",
            "duration": duration,
            "timestamp": time.time(),
            "is_positive": is_positive
        })
        
        # Log and return response
        try:
            return response.json()
        except:
            return {"status": "error", "action": "Failed to parse response"}
    except requests.exceptions.RequestException as e:
        # Record error performance
        duration = time.time() - start_time
        print(f"Frontend feedback API call error: {str(e)} ({duration:.4f}s)")
        
        st.error(f"Error sending feedback: {str(e)}")
        return {"status": "error", "action": str(e)}

@monitor_performance(endpoint="get_knowledge_graph")
def get_knowledge_graph(limit: int = 100, query: str = None) -> Dict:
    """Get knowledge graph data"""
    # Generate cache key
    cache_key = f"kg:limit={limit}:query={query}"
    
    # Check cache
    if cache_key in st.session_state.cache.get('knowledge_graphs', {}):
        return st.session_state.cache['knowledge_graphs'][cache_key]
    
    try:
        params = {"limit": limit}
        if query:
            params["query"] = query
            
        response = requests.get(
            f"{API_URL}/knowledge_graph",
            params=params,
            timeout=30
        )
        result = response.json()
        
        # Cache result
        if 'knowledge_graphs' not in st.session_state.cache:
            st.session_state.cache['knowledge_graphs'] = {}
        st.session_state.cache['knowledge_graphs'][cache_key] = result
        
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting knowledge graph: {str(e)}")
        return {"nodes": [], "links": []}

def get_knowledge_graph_from_message(message: str, query: str = None):
    """Extract knowledge graph data from AI response"""
    # Generate cache key - use message hash and query combination
    import hashlib
    message_hash = hashlib.md5(message.encode()).hexdigest()
    cache_key = f"kg_msg:{message_hash}:query={query}"
    
    # Check cache
    if cache_key in st.session_state.cache.get('knowledge_graphs', {}):
        return st.session_state.cache['knowledge_graphs'][cache_key]
    
    try:
        params = {"message": message}
        if query:
            params["query"] = query
            
        response = requests.get(
            f"{API_URL}/knowledge_graph_from_message",
            params=params,
            timeout=30
        )
        result = response.json()
        
        # Cache result
        if 'knowledge_graphs' not in st.session_state.cache:
            st.session_state.cache['knowledge_graphs'] = {}
        st.session_state.cache['knowledge_graphs'][cache_key] = result
        
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"Error extracting knowledge graph from response: {str(e)}")
        return {"nodes": [], "links": []}

@monitor_performance(endpoint="get_source_content")
def get_source_content(source_id: str) -> Dict:
    """Get source content"""
    # Check cache
    cache_key = f"content:{source_id}"
    if cache_key in st.session_state.cache.get('api_responses', {}):
        return st.session_state.cache['api_responses'][cache_key]
    
    try:
        response = requests.post(
            f"{API_URL}/source",
            json={"source_id": source_id},
            timeout=30
        )
        result = response.json()
        
        # Cache result
        if 'api_responses' not in st.session_state.cache:
            st.session_state.cache['api_responses'] = {}
        st.session_state.cache['api_responses'][cache_key] = result
        
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting source content: {str(e)}")
        return None

def get_source_file_info(source_id: str) -> dict:
    """Get file information corresponding to source ID"""
    # Check cache
    if source_id in st.session_state.cache.get('source_info', {}):
        return st.session_state.cache['source_info'][source_id]
    
    try:
        response = requests.post(
            f"{API_URL}/source_info",
            json={"source_id": source_id},
            timeout=10
        )
        result = response.json()
        
        # Cache result
        if 'source_info' not in st.session_state.cache:
            st.session_state.cache['source_info'] = {}
        st.session_state.cache['source_info'][source_id] = result
        
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting source file info: {str(e)}")
        default_info = {"file_name": f"Source Text {source_id}"}
        
        # Cache default result
        if 'source_info' not in st.session_state.cache:
            st.session_state.cache['source_info'] = {}
        st.session_state.cache['source_info'][source_id] = default_info
        
        return default_info

def get_source_file_info_batch(source_ids: list) -> dict:
    """Get file information for multiple source IDs
    
    Args:
        source_ids: List of source IDs
        
    Returns:
        Dict: Mapping dictionary from ID to file information
    """
    try:
        response = requests.post(
            f"{API_URL}/source_info_batch",
            json={"source_ids": source_ids},
            timeout=10
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error batch getting source file info: {str(e)}")
        return {sid: {"file_name": f"Source Text {sid}"} for sid in source_ids}

@monitor_performance(endpoint="kg_reasoning")
def get_kg_reasoning(reasoning_type, entity_a, entity_b=None, max_depth=3, algorithm=community_algorithm):
    """Knowledge graph reasoning API call"""
    try:
        params = {
            "reasoning_type": reasoning_type,
            "entity_a": entity_a.strip() if entity_a else "",
            "max_depth": min(max(1, max_depth), 5),  # Ensure in range 1-5
            "algorithm": algorithm
        }
        
        if entity_b:
            params["entity_b"] = entity_b.strip()
        
        # print(f"Sending knowledge graph reasoning request: {params}")
        
        # Send request in JSON format
        response = requests.post(
            f"{API_URL}/kg_reasoning",
            json=params,
            timeout=60  # Community detection may require longer time
        )
        
        if response.status_code != 200:
            st.error(f"API request failed: HTTP {response.status_code}")
            try:
                error_details = response.json()
                return {"error": f"API error: {error_details.get('detail', 'Unknown error')}", "nodes": [], "links": []}
            except:
                return {"error": f"API error: HTTP {response.status_code}", "nodes": [], "links": []}
        
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Knowledge graph reasoning request failed: {str(e)}")
        return {"error": str(e), "nodes": [], "links": []}

def get_entity_types():
    """Get all entity types"""
    try:
        response = requests.get(
            f"{API_URL}/entity_types",
            timeout=10
        )
        result = response.json()
        return result.get("entity_types", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get entity types: {str(e)}")
        return []

def get_relation_types():
    """Get all relation types"""
    try:
        response = requests.get(
            f"{API_URL}/relation_types",
            timeout=10
        )
        result = response.json()
        return result.get("relation_types", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get relation types: {str(e)}")
        return []

def get_entities(filters=None):
    """Get entity list with filter support"""
    try:
        if not filters:
            filters = {}
            
        response = requests.post(
            f"{API_URL}/entities/search",
            json=filters,
            timeout=20
        )
        result = response.json()
        return result.get("entities", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get entity list: {str(e)}")
        return []

def get_relations(filters=None):
    """Get relation list with filter support"""
    try:
        if not filters:
            filters = {}
            
        response = requests.post(
            f"{API_URL}/relations/search",
            json=filters,
            timeout=20
        )
        result = response.json()
        return result.get("relations", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get relation list: {str(e)}")
        return []

def create_entity(entity_data):
    """Create new entity"""
    try:
        response = requests.post(
            f"{API_URL}/entity/create",
            json=entity_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create entity: {str(e)}")
        return {"success": False, "message": str(e)}

def update_entity(entity_data):
    """Update entity"""
    try:
        response = requests.post(
            f"{API_URL}/entity/update",
            json=entity_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to update entity: {str(e)}")
        return {"success": False, "message": str(e)}

def delete_entity(entity_id):
    """Delete entity"""
    try:
        response = requests.post(
            f"{API_URL}/entity/delete",
            json={"id": entity_id},
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to delete entity: {str(e)}")
        return {"success": False, "message": str(e)}

def create_relation(relation_data):
    """Create new relation"""
    try:
        response = requests.post(
            f"{API_URL}/relation/create",
            json=relation_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create relation: {str(e)}")
        return {"success": False, "message": str(e)}

def update_relation(relation_data):
    """Update relation"""
    try:
        response = requests.post(
            f"{API_URL}/relation/update",
            json=relation_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to update relation: {str(e)}")
        return {"success": False, "message": str(e)}

def delete_relation(relation_data):
    """Delete relation"""
    try:
        response = requests.post(
            f"{API_URL}/relation/delete",
            json=relation_data,
            timeout=15
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to delete relation: {str(e)}")
        return {"success": False, "message": str(e)}

def clear_chat():
    """Clear chat history"""
    try:
        # Clear frontend state
        st.session_state.processing_lock = False
        st.session_state.messages = []
        st.session_state.execution_log = None
        st.session_state.kg_data = None
        st.session_state.source_content = None
        
        # Important: also clear current_kg_message
        if 'current_kg_message' in st.session_state:
            del st.session_state.current_kg_message
        
        # Clear backend state
        try:
            response = requests.post(
                f"{API_URL}/clear",
                json={"session_id": st.session_state.session_id}
            )

            if response.status_code != 200:
                st.warning("Could not clear backend conversation history (backend may be offline)")
                # return  # Commented out: allow the frontend clear to complete even without backend
        except Exception as backend_err:
            st.warning(f"Could not reach backend to clear history (it may be offline): {backend_err}")

        # Always regenerate session ID and rerun, even if the backend call failed
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    except Exception as e:
        st.session_state.processing_lock = False
        st.error(f"Error clearing conversation: {str(e)}")
        # Still regenerate session and rerun so the frontend is fully reset
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

def clear_cache(cache_type=None):
    """Clear specified type or all caches"""
    if cache_type and cache_type in st.session_state.cache:
        st.session_state.cache[cache_type] = {}
    elif not cache_type:
        st.session_state.cache = {
            'source_info': {},
            'knowledge_graphs': {},
            'vector_search_results': {},
            'api_responses': {},
        }



class ApiBatchProcessor:
    """API request batch processor, merges similar requests within short time window"""
    
    def __init__(self, batch_window=0.5, max_batch_size=10):
        """
        Initialize batch processor
        
        Args:
            batch_window: Batch processing window time (seconds)
            max_batch_size: Maximum batch size
        """
        self.batch_window = batch_window
        self.max_batch_size = max_batch_size
        self.queues = {}  # Queue for each request type
        self.locks = {}   # Lock for each queue
        self.threads = {} # Processing threads
        self.running = True
    
    def add_request(self, request_type, request_data, callback):
        """
        Add request to queue
        
        Args:
            request_type: Request type (e.g. 'source_info', 'kg_data')
            request_data: Request data
            callback: Callback function to handle result return
        """
        # Initialize if this is the first time using this request type
        if request_type not in self.queues:
            self.queues[request_type] = queue.Queue()
            self.locks[request_type] = threading.Lock()
            # Start processing thread
            self.threads[request_type] = threading.Thread(
                target=self._process_queue,
                args=(request_type,),
                daemon=True
            )
            self.threads[request_type].start()
        
        # Add to queue
        self.queues[request_type].put((request_data, callback))
    
    def _process_queue(self, request_type):
        """
        Process request queue of specific type
        
        Args:
            request_type: Request type
        """
        while self.running:
            batch = []
            callbacks = []
            
            # Try to collect requests within time window
            try:
                # Get first request, block and wait
                first_request, first_callback = self.queues[request_type].get(block=True)
                batch.append(first_request)
                callbacks.append(first_callback)
                
                # Set batch processing end time
                end_time = time.time() + self.batch_window
                
                # Collect more requests until window ends or max batch size reached
                while time.time() < end_time and len(batch) < self.max_batch_size:
                    try:
                        request, callback = self.queues[request_type].get(block=False)
                        batch.append(request)
                        callbacks.append(callback)
                    except queue.Empty:
                        break
                
                # Process batch requests
                if len(batch) > 1:
                    # Execute batch processing
                    self._execute_batch(request_type, batch, callbacks)
                else:
                    # Single request, process directly
                    self._execute_single(request_type, batch[0], callbacks[0])
                    
            except Exception as e:
                print(f"Batch processing error ({request_type}): {e}")
                time.sleep(0.1)  # Avoid high CPU usage
    
    def _execute_batch(self, request_type, batch, callbacks):
        """Execute batch requests"""
        try:
            if request_type == 'source_info':
                # Batch get source information
                source_ids = batch
                results = self._batch_get_source_info(source_ids)
                
                # Process callbacks
                for i, callback in enumerate(callbacks):
                    source_id = source_ids[i]
                    if source_id in results:
                        callback(results[source_id])
                    else:
                        # Default result
                        callback({"file_name": f"Source Text {source_id}"})
                        
            elif request_type == 'content':
                # Batch get content
                chunk_ids = batch
                results = self._batch_get_content(chunk_ids)
                
                # Process callbacks
                for i, callback in enumerate(callbacks):
                    chunk_id = chunk_ids[i]
                    if chunk_id in results:
                        callback(results[chunk_id])
                    else:
                        callback(None)
                        
            # Can add other batch processing types...
            
        except Exception as e:
            print(f"Execute batch request error ({request_type}): {e}")
            # Execute each request individually on error
            for i, request in enumerate(batch):
                try:
                    self._execute_single(request_type, request, callbacks[i])
                except Exception as single_err:
                    print(f"Single request error ({request_type}): {single_err}")
    
    def _execute_single(self, request_type, request, callback):
        """Execute single request"""
        try:
            if request_type == 'source_info':
                result = get_source_file_info(request)
                callback(result)
            elif request_type == 'content':
                result = get_source_content(request)
                callback(result)
            # Can add other request types...
        except Exception as e:
            print(f"Execute single request error ({request_type}): {e}")
            callback(None)
    
    def _batch_get_source_info(self, source_ids):
        """Batch get source information"""
        try:
            response = requests.post(
                f"{API_URL}/source_info_batch",
                json={"source_ids": source_ids},
                timeout=10
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Batch get source info error: {e}")
            return {}
    
    def _batch_get_content(self, chunk_ids):
        """Batch get content"""
        try:
            response = requests.post(
                f"{API_URL}/content_batch",
                json={"chunk_ids": chunk_ids},
                timeout=30
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Batch get content error: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown batch processor"""
        self.running = False
        # Wait for all threads to complete
        for thread in self.threads.values():
            if thread.is_alive():
                thread.join(timeout=1.0)

# Initialize batch processor
def get_batch_processor():
    if 'api_batch_processor' not in st.session_state:
        st.session_state.api_batch_processor = ApiBatchProcessor()
    return st.session_state.api_batch_processor

# Example API functions using batch processor
def get_source_info_async(source_id, callback):
    """Asynchronously get source information using batch processor"""
    processor = get_batch_processor()
    processor.add_request('source_info', source_id, callback)

def get_content_async(chunk_id, callback):
    """Asynchronously get content using batch processor"""
    processor = get_batch_processor()
    processor.add_request('content', chunk_id, callback)

# Shutdown batch processor on application exit
def shutdown_batch_processor():
    if 'api_batch_processor' in st.session_state:
        st.session_state.api_batch_processor.shutdown()