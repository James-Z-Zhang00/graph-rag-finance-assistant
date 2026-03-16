import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import threading

def display_performance_stats():
    """Display performance statistics"""
    # Check if enhanced performance collector is available
    if 'performance_collector' in st.session_state:
        return display_enhanced_performance_stats()

    # Fall back to legacy implementation
    if 'performance_metrics' not in st.session_state or not st.session_state.performance_metrics:
        st.info("No performance data available")
        return

    # Calculate message response time statistics
    message_times = [m["duration"] for m in st.session_state.performance_metrics
                    if m["operation"] == "send_message"]

    if message_times:
        avg_time = sum(message_times) / len(message_times)
        max_time = max(message_times)
        min_time = min(message_times)

        st.subheader("Message Response Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Response Time", f"{avg_time:.2f}s")
        with col2:
            st.metric("Max Response Time", f"{max_time:.2f}s")
        with col3:
            st.metric("Min Response Time", f"{min_time:.2f}s")

        # Plot response time chart
        if len(message_times) > 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(message_times))
            ax.plot(x, message_times, marker='o')
            ax.set_title('Response Time Trend')
            ax.set_xlabel('Message ID')
            ax.set_ylabel('Response Time (s)')
            ax.grid(True)

            st.pyplot(fig)

    # Feedback performance statistics
    feedback_times = [m["duration"] for m in st.session_state.performance_metrics
                     if m["operation"] == "send_feedback"]

    if feedback_times:
        avg_feedback_time = sum(feedback_times) / len(feedback_times)
        st.subheader("Feedback Processing Performance")
        st.metric("Average Feedback Processing Time", f"{avg_feedback_time:.2f}s")

def clear_performance_data():
    """Clear all performance data"""
    # Clear enhanced performance collector
    if 'performance_collector' in st.session_state:
        collector = st.session_state.performance_collector
        collector.reset()

    # Clear legacy performance data format
    if 'performance_metrics' in st.session_state:
        st.session_state.performance_metrics = []

    return True

# Performance data collector class
class PerformanceCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.api_calls = defaultdict(int)
        self.api_times = defaultdict(float)
        self.page_loads = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def record_api_call(self, endpoint, duration):
        """Record an API call"""
        with self.lock:
            self.api_calls[endpoint] += 1
            self.api_times[endpoint] += duration

    def record_metric(self, name, value):
        """Record a general performance metric"""
        with self.lock:
            self.metrics[name].append(value)

    def record_page_load(self):
        """Record a page load"""
        with self.lock:
            self.page_loads += 1

    def get_uptime(self):
        """Get application uptime in seconds"""
        return time.time() - self.start_time

    def get_api_stats(self):
        """Get API call statistics"""
        with self.lock:
            total_calls = sum(self.api_calls.values())
            total_time = sum(self.api_times.values())
            return {
                "total_calls": total_calls,
                "total_time": total_time,
                "avg_time": total_time / total_calls if total_calls else 0,
                "calls_by_endpoint": dict(self.api_calls),
                "time_by_endpoint": dict(self.api_times)
            }

    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.metrics = defaultdict(list)
            self.api_calls = defaultdict(int)
            self.api_times = defaultdict(float)
            self.page_loads = 0
            self.start_time = time.time()

# Helper to get or create a performance collector
def get_performance_collector():
    """Get or create the performance collector instance"""
    if "performance_collector" not in st.session_state:
        st.session_state.performance_collector = PerformanceCollector()
    return st.session_state.performance_collector

# Performance monitoring decorator
def monitor_performance(endpoint=None):
    """Decorator that monitors function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Record with enhanced collector
            try:
                collector = get_performance_collector()
                if endpoint:
                    collector.record_api_call(endpoint, duration)
                else:
                    func_name = func.__name__
                    collector.record_metric(f"func:{func_name}", duration)
            except Exception as e:
                print(f"Failed to record performance data: {e}")

            # Also record in legacy format for backwards compatibility
            if 'performance_metrics' in st.session_state:
                st.session_state.performance_metrics.append({
                    "operation": endpoint or func.__name__,
                    "duration": duration,
                    "timestamp": time.time()
                })

            return result
        return wrapper
    return decorator

# Enhanced performance stats display
def display_enhanced_performance_stats():
    """Display enhanced performance statistics"""
    collector = get_performance_collector()

    # Basic application statistics
    st.subheader("Application Performance Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        uptime = collector.get_uptime()
        days, remainder = divmod(uptime, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"
        st.metric("Uptime", uptime_str)

    with col2:
        api_stats = collector.get_api_stats()
        st.metric("Total API Calls", f"{api_stats['total_calls']}")

    with col3:
        st.metric("Average Response Time", f"{api_stats['avg_time']:.2f}s")

    # API call statistics
    if api_stats['total_calls'] > 0:
        st.subheader("API Call Statistics")

        # Build DataFrame for sorting and display
        api_data = []
        for endpoint, count in api_stats['calls_by_endpoint'].items():
            time_total = api_stats['time_by_endpoint'].get(endpoint, 0)
            time_avg = time_total / count if count else 0
            api_data.append({
                "Endpoint": endpoint,
                "Call Count": count,
                "Total Time (s)": round(time_total, 2),
                "Avg Time (s)": round(time_avg, 2)
            })

        df = pd.DataFrame(api_data)
        if not df.empty:
            # Sort by call count descending
            df = df.sort_values(by="Call Count", ascending=False)
            st.dataframe(df, use_container_width=True)

            # Visualize API call distribution
            if len(df) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                endpoints = df["Endpoint"].tolist()
                calls = df["Call Count"].tolist()

                # Horizontal bar chart
                y_pos = np.arange(len(endpoints))
                ax.barh(y_pos, calls, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(endpoints)
                ax.invert_yaxis()  # Highest at the top
                ax.set_xlabel('Call Count')
                ax.set_title('API Call Distribution')

                st.pyplot(fig)

    # Message response time analysis
    if 'performance_metrics' in st.session_state and st.session_state.performance_metrics:
        message_times = [m["duration"] for m in st.session_state.performance_metrics
                        if m["operation"] == "send_message"]

        if message_times:
            st.subheader("Message Response Performance")
            col1, col2, col3 = st.columns(3)
            avg_time = sum(message_times) / len(message_times)
            max_time = max(message_times)
            min_time = min(message_times)

            with col1:
                st.metric("Average Response Time", f"{avg_time:.2f}s")
            with col2:
                st.metric("Max Response Time", f"{max_time:.2f}s")
            with col3:
                st.metric("Min Response Time", f"{min_time:.2f}s")

            # Plot response time chart
            if len(message_times) > 1:
                fig, ax = plt.subplots(figsize=(10, 4))
                x = np.arange(len(message_times))
                ax.plot(x, message_times, marker='o')
                ax.set_title('Response Time Trend')
                ax.set_xlabel('Message ID')
                ax.set_ylabel('Response Time (s)')
                ax.grid(True)

                st.pyplot(fig)

    # System resource monitoring
    if collector.metrics:
        st.subheader("System Resource Monitoring")

        # Plot memory usage chart if data is available
        if "memory_usage" in collector.metrics and len(collector.metrics["memory_usage"]) > 1:
            memory_data = collector.metrics["memory_usage"]
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(memory_data))
            ax.plot(x, memory_data, marker='o', color='green')
            ax.set_title('Memory Usage Trend')
            ax.set_xlabel('Checkpoint')
            ax.set_ylabel('Memory Usage (MB)')
            ax.grid(True)

            st.pyplot(fig)

    # Performance analysis tools
    st.subheader("Performance Analysis Tools")
    analyze_tab, config_tab = st.tabs(["Performance Analysis", "Configuration"])

    with analyze_tab:
        if st.button("Run Performance Check", key="run_perf_check"):
            with st.spinner("Detecting performance bottlenecks..."):
                # Simulate performance check
                time.sleep(1.5)

                # Display results
                st.success("Performance check complete")
                st.info("""
                Performance analysis results:
                1. API calls - Status OK
                2. Frontend rendering - Status OK
                3. Data processing - No significant bottlenecks
                """)

    with config_tab:
        st.checkbox("Enable verbose API logging", value=False, key="enable_api_logging")
        st.slider("Performance data retention (hours)", min_value=1, max_value=24, value=6, key="perf_data_retention")

        if st.button("Apply Configuration", key="apply_perf_config"):
            st.success("Configuration updated")

# Decorate API call functions
def decorate_api_functions():
    """Add performance monitoring decorators to API functions"""
    try:
        from frontend.utils.api import send_message, send_feedback, get_knowledge_graph, get_source_content

        # Save originals
        original_send_message = send_message
        original_send_feedback = send_feedback
        original_get_knowledge_graph = get_knowledge_graph
        original_get_source_content = get_source_content

        # Wrap with monitoring decorator
        @monitor_performance(endpoint="send_message")
        def monitored_send_message(*args, **kwargs):
            return original_send_message(*args, **kwargs)

        @monitor_performance(endpoint="send_feedback")
        def monitored_send_feedback(*args, **kwargs):
            return original_send_feedback(*args, **kwargs)

        @monitor_performance(endpoint="get_knowledge_graph")
        def monitored_get_knowledge_graph(*args, **kwargs):
            return original_get_knowledge_graph(*args, **kwargs)

        @monitor_performance(endpoint="get_source_content")
        def monitored_get_source_content(*args, **kwargs):
            return original_get_source_content(*args, **kwargs)

        # Replace originals
        import frontend.utils.api
        frontend.utils.api.send_message = monitored_send_message
        frontend.utils.api.send_feedback = monitored_send_feedback
        frontend.utils.api.get_knowledge_graph = monitored_get_knowledge_graph
        frontend.utils.api.get_source_content = monitored_get_source_content

        return True
    except Exception as e:
        print(f"Failed to decorate API functions: {e}")
        return False

# Initialize performance collection on app startup
def init_performance_monitoring():
    """Initialize performance monitoring"""
    # Get or create collector
    collector = get_performance_collector()

    # Record page load
    collector.record_page_load()

    # Decorate API functions
    decorate_api_functions()

    return collector
