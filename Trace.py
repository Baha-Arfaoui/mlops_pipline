import time
import json
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import uuid
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


class LangGraphDetailedTracer(BaseCallbackHandler):
    """
    Comprehensive tracing callback handler for LangGraph agents.
    
    This tracer captures detailed metadata about execution of LangGraph agents,
    including LLM calls, tool usage, node transitions, and more.
    """
    
    def __init__(
        self,
        trace_file: Optional[str] = None,
        auto_save: bool = False,
        save_interval: int = 30,
        include_full_prompts: bool = True
    ):
        """Initialize the callback handler with storage for traces.
        
        Args:
            trace_file: Optional file path to save traces to
            auto_save: Whether to periodically save traces
            save_interval: How often to save traces (in seconds)
            include_full_prompts: Whether to include complete prompt texts (may be large)
        """
        super().__init__()
        
        # Core trace storage
        self.execution_id = str(uuid.uuid4())
        self.execution_start = self._get_timestamp()
        self.execution_metadata = {
            "start_time": self.execution_start,
            "start_time_iso": datetime.now().isoformat(),
        }
        
        # Specific trace categories
        self.traces = []
        self.llm_calls = []
        self.tool_calls = []
        self.agent_actions = []
        self.node_traces = []
        self.state_transitions = []
        self.errors = []
        
        # Current trace context
        self.current_chain_trace = {}
        self.current_llm_trace = {}
        self.active_runs = {}  # Track nested runs by run_id
        
        # Configuration
        self.trace_file = trace_file
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.include_full_prompts = include_full_prompts
        self.last_save_time = self._get_timestamp()
        
        # Stats
        self.total_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.total_latency = 0.0
    
    # Main tracing methods for LangChain callback protocol
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Log when an LLM starts."""
        run_id = kwargs.get("run_id", str(uuid.uuid4()))
        
        # Create detailed trace entry
        llm_info = {
            "type": "llm_call",
            "llm_type": serialized.get("name", "Unknown LLM"),
            "start_time": self._get_timestamp(),
            "start_time_iso": datetime.now().isoformat(),
            "run_id": run_id,
            "metadata": kwargs.get("metadata", {}),
            "invocation_params": self._clean_serialized_inputs(serialized.get("kwargs", {})),
        }
        
        # Include prompts based on configuration
        if self.include_full_prompts:
            llm_info["prompts"] = prompts
        else:
            llm_info["prompt_summary"] = [
                f"Prompt #{i+1}: {len(p)} chars" for i, p in enumerate(prompts)
            ]
        
        # Store in active runs for later completion
        self.active_runs[run_id] = llm_info
        self.llm_calls.append(llm_info)
        self.traces.append(llm_info)
        
        # Auto-save if configured
        self._try_autosave()
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Log when an LLM ends."""
        run_id = kwargs.get("run_id", "")
        
        if run_id in self.active_runs:
            end_time = self._get_timestamp()
            duration = end_time - self.active_runs[run_id].get("start_time", end_time)
            
            # Extract token usage
            token_usage = {}
            if response.llm_output and "token_usage" in response.llm_output:
                token_usage = response.llm_output["token_usage"]
                
                # Update global stats
                self.total_tokens["prompt"] += token_usage.get("prompt_tokens", 0)
                self.total_tokens["completion"] += token_usage.get("completion_tokens", 0)
                self.total_tokens["total"] += token_usage.get("total_tokens", 0)
            
            # Update trace with completion info
            self.active_runs[run_id].update({
                "end_time": end_time,
                "end_time_iso": datetime.now().isoformat(),
                "duration": duration,
                "token_usage": token_usage,
                "response": self._clean_llm_response(response),
                "status": "completed"
            })
            
            # Update latency stats
            self.total_latency += duration
            
            # Remove from active runs
            self.active_runs.pop(run_id, None)
            
            # Auto-save if configured
            self._try_autosave()
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Log when an LLM errors."""
        run_id = kwargs.get("run_id", "")
        
        if run_id in self.active_runs:
            end_time = self._get_timestamp()
            
            # Update trace with error info
            self.active_runs[run_id].update({
                "end_time": end_time,
                "end_time_iso": datetime.now().isoformat(),
                "duration": end_time - self.active_runs[run_id].get("start_time", end_time),
                "error": str(error),
                "error_type": error.__class__.__name__,
                "status": "error"
            })
            
            # Add to errors collection
            self.errors.append({
                "timestamp": end_time,
                "timestamp_iso": datetime.now().isoformat(),
                "type": "llm_error",
                "run_id": run_id,
                "error": str(error),
                "error_type": error.__class__.__name__
            })
            
            # Remove from active runs
            self.active_runs.pop(run_id, None)
            
            # Auto-save if configured
            self._try_autosave()
    
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Log when a chain starts."""
        run_id = kwargs.get("run_id", str(uuid.uuid4()))
        
        chain_info = {
            "type": "chain",
            "chain_type": serialized.get("name", "Unknown Chain"),
            "start_time": self._get_timestamp(),
            "start_time_iso": datetime.now().isoformat(),
            "inputs": self._clean_inputs(inputs),
            "metadata": kwargs.get("metadata", {}),
            "run_id": run_id,
        }
        
        self.active_runs[run_id] = chain_info
        self.traces.append(chain_info)
        
        # Auto-save if configured
        self._try_autosave()
    
    def on_chain_end(
        self, outputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Log when a chain ends."""
        run_id = kwargs.get("run_id", "")
        
        if run_id in self.active_runs:
            end_time = self._get_timestamp()
            
            # Update trace with completion info
            self.active_runs[run_id].update({
                "end_time": end_time,
                "end_time_iso": datetime.now().isoformat(),
                "outputs": self._clean_outputs(outputs),
                "duration": end_time - self.active_runs[run_id].get("start_time", end_time),
                "status": "completed"
            })
            
            # Remove from active runs
            self.active_runs.pop(run_id, None)
            
            # Auto-save if configured
            self._try_autosave()
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Log when a chain errors."""
        run_id = kwargs.get("run_id", "")
        
        if run_id in self.active_runs:
            end_time = self._get_timestamp()
            
            # Update trace with error info
            self.active_runs[run_id].update({
                "end_time": end_time,
                "end_time_iso": datetime.now().isoformat(),
                "duration": end_time - self.active_runs[run_id].get("start_time", end_time),
                "error": str(error),
                "error_type": error.__class__.__name__,
                "status": "error"
            })
            
            # Add to errors collection
            self.errors.append({
                "timestamp": end_time,
                "timestamp_iso": datetime.now().isoformat(),
                "type": "chain_error",
                "run_id": run_id,
                "error": str(error),
                "error_type": error.__class__.__name__
            })
            
            # Remove from active runs
            self.active_runs.pop(run_id, None)
            
            # Auto-save if configured
            self._try_autosave()
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Log when a tool starts."""
        run_id = kwargs.get("run_id", str(uuid.uuid4()))
        
        tool_info = {
            "type": "tool_call",
            "tool_name": serialized.get("name", "Unknown Tool"),
            "description": serialized.get("description", ""),
            "input": input_str,
            "start_time": self._get_timestamp(),
            "start_time_iso": datetime.now().isoformat(),
            "metadata": kwargs.get("metadata", {}),
            "run_id": run_id,
        }
        
        self.active_runs[run_id] = tool_info
        self.tool_calls.append(tool_info)
        self.traces.append(tool_info)
        
        # Auto-save if configured
        self._try_autosave()
    
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Log when a tool ends."""
        run_id = kwargs.get("run_id", "")
        
        if run_id in self.active_runs:
            end_time = self._get_timestamp()
            
            # Update trace with completion info
            self.active_runs[run_id].update({
                "end_time": end_time,
                "end_time_iso": datetime.now().isoformat(),
                "output": output,
                "duration": end_time - self.active_runs[run_id].get("start_time", end_time),
                "status": "completed"
            })
            
            # Remove from active runs
            self.active_runs.pop(run_id, None)
            
            # Auto-save if configured
            self._try_autosave()
    
    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Log when a tool errors."""
        run_id = kwargs.get("run_id", "")
        
        if run_id in self.active_runs:
            end_time = self._get_timestamp()
            
            # Update trace with error info
            self.active_runs[run_id].update({
                "end_time": end_time,
                "end_time_iso": datetime.now().isoformat(),
                "duration": end_time - self.active_runs[run_id].get("start_time", end_time),
                "error": str(error),
                "error_type": error.__class__.__name__,
                "status": "error"
            })
            
            # Add to errors collection
            self.errors.append({
                "timestamp": end_time,
                "timestamp_iso": datetime.now().isoformat(),
                "type": "tool_error",
                "run_id": run_id, 
                "error": str(error),
                "error_type": error.__class__.__name__
            })
            
            # Remove from active runs
            self.active_runs.pop(run_id, None)
            
            # Auto-save if configured
            self._try_autosave()
    
    def on_agent_action(self, action: Dict[str, Any], **kwargs: Any) -> Any:
        """Log agent actions."""
        run_id = kwargs.get("run_id", str(uuid.uuid4()))
        
        action_info = {
            "type": "agent_action",
            "action": action,
            "tool": action.get("tool", ""),
            "tool_input": action.get("tool_input", ""),
            "log": action.get("log", ""),
            "timestamp": self._get_timestamp(),
            "timestamp_iso": datetime.now().isoformat(),
            "run_id": run_id,
            "metadata": kwargs.get("metadata", {})
        }
        
        self.agent_actions.append(action_info)
        self.traces.append(action_info)
        
        # Auto-save if configured
        self._try_autosave()
    
    def on_agent_finish(self, finish: Dict[str, Any], **kwargs: Any) -> Any:
        """Log agent finish."""
        run_id = kwargs.get("run_id", "")
        
        finish_info = {
            "type": "agent_finish",
            "output": finish.get("output", ""),
            "log": finish.get("log", ""),
            "return_values": finish.get("return_values", {}),
            "timestamp": self._get_timestamp(),
            "timestamp_iso": datetime.now().isoformat(),
            "run_id": run_id,
            "metadata": kwargs.get("metadata", {})
        }
        
        self.agent_actions.append(finish_info)
        self.traces.append(finish_info)
        
        # Auto-save if configured
        self._try_autosave()
    
    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Log text outputs."""
        text_info = {
            "type": "text",
            "text": text,
            "timestamp": self._get_timestamp(),
            "timestamp_iso": datetime.now().isoformat(),
            "run_id": kwargs.get("run_id", str(uuid.uuid4())),
            "metadata": kwargs.get("metadata", {})
        }
        
        self.traces.append(text_info)
        
        # Auto-save if configured
        self._try_autosave()
    
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Log when a chat model starts."""
        run_id = kwargs.get("run_id", str(uuid.uuid4()))
        
        # Create a serializable version of messages
        serializable_messages = []
        for msg_list in messages:
            serializable_messages.append([self._serialize_message(m) for m in msg_list])
        
        chat_info = {
            "type": "chat_model_call",
            "model_type": serialized.get("name", "Unknown Chat Model"),
            "start_time": self._get_timestamp(),
            "start_time_iso": datetime.now().isoformat(),
            "metadata": kwargs.get("metadata", {}),
            "run_id": run_id,
            "invocation_params": self._clean_serialized_inputs(serialized.get("kwargs", {})),
        }
        
        # Include messages based on configuration
        if self.include_full_prompts:
            chat_info["messages"] = serializable_messages
        else:
            chat_info["messages_summary"] = [
                f"Message set #{i+1}: {len(msg_list)} messages" 
                for i, msg_list in enumerate(messages)
            ]
        
        self.active_runs[run_id] = chat_info
        self.llm_calls.append(chat_info)
        self.traces.append(chat_info)
        
        # Auto-save if configured
        self._try_autosave()
    
    # LangGraph specific methods
    
    def on_node_start(
        self, node_name: str, inputs: Dict[str, Any], run_id: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Log when a graph node starts execution."""
        if run_id is None:
            run_id = str(uuid.uuid4())
            
        node_info = {
            "type": "node_execution",
            "node_name": node_name,
            "inputs": self._clean_inputs(inputs),
            "start_time": self._get_timestamp(),
            "start_time_iso": datetime.now().isoformat(),
            "metadata": kwargs.get("metadata", {}),
            "run_id": run_id,
        }
        
        self.active_runs[run_id] = node_info
        self.node_traces.append(node_info)
        self.traces.append(node_info)
        
        # Auto-save if configured
        self._try_autosave()
        
        return run_id

    def on_node_end(
        self, node_name: str, outputs: Dict[str, Any], run_id: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Log when a graph node finishes execution."""
        if run_id is None or run_id not in self.active_runs:
            # This node execution wasn't tracked on start
            return
            
        end_time = self._get_timestamp()
        
        # Update trace with completion info
        self.active_runs[run_id].update({
            "end_time": end_time,
            "end_time_iso": datetime.now().isoformat(),
            "outputs": self._clean_outputs(outputs),
            "duration": end_time - self.active_runs[run_id].get("start_time", end_time),
            "status": "completed"
        })
        
        # Remove from active runs
        self.active_runs.pop(run_id, None)
        
        # Auto-save if configured
        self._try_autosave()

    def on_node_error(
        self, node_name: str, error: Union[Exception, KeyboardInterrupt], 
        run_id: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Log when a graph node errors."""
        if run_id is None or run_id not in self.active_runs:
            # This node execution wasn't tracked on start
            return
            
        end_time = self._get_timestamp()
        
        # Update trace with error info
        self.active_runs[run_id].update({
            "end_time": end_time,
            "end_time_iso": datetime.now().isoformat(),
            "duration": end_time - self.active_runs[run_id].get("start_time", end_time),
            "error": str(error),
            "error_type": error.__class__.__name__,
            "status": "error"
        })
        
        # Add to errors collection
        self.errors.append({
            "timestamp": end_time,
            "timestamp_iso": datetime.now().isoformat(),
            "type": "node_error",
            "node_name": node_name,
            "run_id": run_id,
            "error": str(error),
            "error_type": error.__class__.__name__
        })
        
        # Remove from active runs
        self.active_runs.pop(run_id, None)
        
        # Auto-save if configured
        self._try_autosave()

    def on_state_transition(
        self, from_state: str, to_state: str, state_data: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Log state transitions in the graph."""
        transition_info = {
            "type": "state_transition",
            "from_state": from_state,
            "to_state": to_state,
            "state_data": self._clean_state_data(state_data),
            "timestamp": self._get_timestamp(),
            "timestamp_iso": datetime.now().isoformat(),
            "run_id": kwargs.get("run_id", str(uuid.uuid4())),
            "metadata": kwargs.get("metadata", {})
        }
        
        self.state_transitions.append(transition_info)
        self.traces.append(transition_info)
        
        # Auto-save if configured
        self._try_autosave()
    
    # Helper methods
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        return time.time()
    
    def _clean_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Clean inputs to make them JSON serializable."""
        if not self.include_full_prompts:
            # If not including full prompts, summarize any text input over 100 chars
            result = {}
            for k, v in inputs.items():
                if isinstance(v, str) and len(v) > 100:
                    result[k] = f"{v[:97]}... ({len(v)} chars)"
                else:
                    result[k] = v
            return result
        
        # Otherwise try to make serializable
        try:
            # Test if it's JSON serializable
            json.dumps(inputs)
            return inputs
        except (TypeError, OverflowError):
            # If not, create a simplified version
            result = {}
            for k, v in inputs.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    result[k] = v
                elif isinstance(v, (list, tuple)):
                    result[k] = f"<{type(v).__name__}> with {len(v)} items"
                elif isinstance(v, dict):
                    result[k] = f"<dict> with {len(v)} keys"
                else:
                    result[k] = f"<{type(v).__name__}>"
            return result
    
    def _clean_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Clean outputs to make them JSON serializable."""
        return self._clean_inputs(outputs)  # Reuse the same logic
    
    def _clean_state_data(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean state data to make it JSON serializable."""
        return self._clean_inputs(state_data)  # Reuse the same logic
    
    def _clean_serialized_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Clean serialized inputs to make them JSON serializable."""
        # Remove any sensitive keys
        result = inputs.copy()
        for sensitive_key in ["api_key", "credentials", "token", "password"]:
            if sensitive_key in result:
                result[sensitive_key] = "**REDACTED**"
        
        return self._clean_inputs(result)
    
    def _clean_llm_response(self, response: LLMResult) -> Dict[str, Any]:
        """Extract and clean LLM response to make it JSON serializable."""
        try:
            return response.dict()
        except Exception:
            # Fallback for when dict() method is not available
            result = {
                "generations": []
            }
            
            # Try to extract generations
            if hasattr(response, "generations"):
                for gen_list in response.generations:
                    gen_list_clean = []
                    for gen in gen_list:
                        gen_dict = {}
                        if hasattr(gen, "text"):
                            gen_dict["text"] = gen.text
                        if hasattr(gen, "message"):
                            gen_dict["message"] = self._serialize_message(gen.message)
                        gen_list_clean.append(gen_dict)
                    result["generations"].append(gen_list_clean)
            
            # Try to extract llm_output
            if hasattr(response, "llm_output") and response.llm_output:
                result["llm_output"] = response.llm_output
                
            return result
    
    def _serialize_message(self, message: BaseMessage) -> Dict[str, Any]:
        """Serialize a BaseMessage to a dict."""
        try:
            return message.dict()
        except Exception:
            # Fallback when dict() method is not available
            result = {
                "type": message.__class__.__name__,
                "content": getattr(message, "content", str(message))
            }
            
            # Try to extract common attributes
            for attr in ["role", "name", "additional_kwargs"]:
                if hasattr(message, attr):
                    result[attr] = getattr(message, attr)
                    
            return result
    
    def _try_autosave(self) -> None:
        """Autosave traces if configured and enough time has passed."""
        if not self.auto_save or not self.trace_file:
            return
            
        current_time = self._get_timestamp()
        if current_time - self.last_save_time >= self.save_interval:
            self.save_traces()
            self.last_save_time = current_time
    
    # Public API methods
    
    def get_traced_data(self) -> Dict[str, Any]:
        """Get all traced data in a structured format."""
        # Update execution metadata
        self.execution_metadata.update({
            "end_time": self._get_timestamp(),
            "end_time_iso": datetime.now().isoformat(),
            "duration": self._get_timestamp() - self.execution_start,
            "total_traces": len(self.traces),
            "total_llm_calls": len(self.llm_calls),
            "total_tool_calls": len(self.tool_calls),
            "total_agent_actions": len(self.agent_actions),
            "total_node_traces": len(self.node_traces),
            "total_state_transitions": len(self.state_transitions),
            "total_errors": len(self.errors),
            "token_usage": self.total_tokens,
            "total_latency": self.total_latency
        })
        
        return {
            "execution_id": self.execution_id,
            "execution_metadata": self.execution_metadata,
            "traces": self.traces,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "agent_actions": self.agent_actions,
            "node_traces": self.node_traces,
            "state_transitions": self.state_transitions,
            "errors": self.errors
        }
    
    def save_traces(self, file_path: Optional[str] = None) -> str:
        """Save traces to a file."""
        if file_path is None:
            if self.trace_file is None:
                # Generate a default filename if none provided
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"langgraph_trace_{timestamp}.json"
            else:
                file_path = self.trace_file
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Get trace data and save
        trace_data = self.get_traced_data()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
            
        return file_path
    
    def clear(self) -> None:
        """Clear all stored traces."""
        self.traces = []
        self.llm_calls = []
        self.tool_calls = []
        self.agent_actions = []
        self.node_traces = []
        self.state_transitions = []
        self.errors = []
        self.current_chain_trace = {}
        self.active_runs = {}
        
        # Reset stats
        self.total_tokens = {"prompt": 0, "completion": 0, "total": 0}
        self.total_latency = 0.0
    
    def to_timeline_data(self) -> List[Dict[str, Any]]:
        """Convert traces to timeline visualization format."""
        timeline_events = []
        
        for call in self.llm_calls:
            if "start_time" in call and "end_time" in call:
                timeline_events.append({
                    "id": call.get("run_id", ""),
                    "content": f"LLM: {call.get('llm_type', 'Unknown')}",
                    "start": call.get("start_time", 0) * 1000,  # Convert to milliseconds
                    "end": call.get("end_time", 0) * 1000,      # Convert to milliseconds
                    "group": "llm",
                    "className": call.get("status", "unknown"),
                    "title": f"Token usage: {call.get('token_usage', {})}"
                })
        
        for call in self.tool_calls:
            if "start_time" in call and "end_time" in call:
                timeline_events.append({
                    "id": call.get("run_id", ""),
                    "content": f"Tool: {call.get('tool_name', 'Unknown')}",
                    "start": call.get("start_time", 0) * 1000,  # Convert to milliseconds
                    "end": call.get("end_time", 0) * 1000,      # Convert to milliseconds
                    "group": "tool",
                    "className": call.get("status", "unknown"),
                    "title": f"Input: {call.get('input', '')[:50]}..."
                })
        
        for node in self.node_traces:
            if "start_time" in node and "end_time" in node:
                timeline_events.append({
                    "id": node.get("run_id", ""),
                    "content": f"Node: {node.get('node_name', 'Unknown')}",
                    "start": node.get("start_time", 0) * 1000,  # Convert to milliseconds
                    "end": node.get("end_time", 0) * 1000,      # Convert to milliseconds
                    "group": "node",
                    "className": node.get("status", "unknown"),
                })
        
        for transition in self.state_transitions:
            timeline_events.append({
                "id": transition.get("run_id", ""),
                "content": f"Transition: {transition.get('from_state', '')} → {transition.get('to_state', '')}",
                "start": transition.get("timestamp", 0) * 1000,  # Convert to milliseconds 
                "end": transition.get("timestamp", 0) * 1000 + 100,  # Add small duration for visibility
                "group": "transition",
                "type": "point",
            })
        
        return timeline_events
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics from traced data."""
        
        # Prepare end time for execution if not already set
        if "end_time" not in self.execution_metadata:
            self.execution_metadata["end_time"] = self._get_timestamp()
            self.execution_metadata["duration"] = self.execution_metadata["end_time"] - self.execution_start
        
        # Basic stats
        total_duration = self.execution_metadata["duration"]
        
        # LLM stats
        llm_stats = {
            "call_count": len(self.llm_calls),
            "total_tokens": self.total_tokens,
            "avg_latency": self.total_latency / len(self.llm_calls) if self.llm_calls else 0,
            "token_rate": self.total_tokens["total"] / self.total_latency if self.total_latency > 0 else 0,
            "total_latency": self.total_latency,
            "percent_time": (self.total_latency / total_duration * 100) if total_duration > 0 else 0,
        }
        
        # Tool stats
        tool_stats = {
            "call_count": len(self.tool_calls),
            "tools_used": {},
            "avg_latency": 0,
            "total_latency": 0,
        }
        
        # Calculate tool-specific stats
        if self.tool_calls:
            # Count tools by name
            for tool in self.tool_calls:
                name = tool.get("tool_name", "Unknown")
                if name not in tool_stats["tools_used"]:
                    tool_stats["tools_used"][name] = 0
                tool_stats["tools_used"][name] += 1
                
            # Calculate latency stats if duration is available
            tool_durations = [
                t.get("end_time", 0) - t.get("start_time", 0) 
                for t in self.tool_calls 
                if "start_time" in t and "end_time" in t
            ]
            
            if tool_durations:
                tool_stats["total_latency"] = sum(tool_durations)
                tool_stats["avg_latency"] = tool_stats["total_latency"] / len(tool_durations)
                tool_stats["percent_time"] = (tool_stats["total_latency"] / total_duration * 100) if total_duration > 0 else 0
        
        # Node stats
        node_stats = {
            "node_count": len(self.node_traces),
            "nodes_visited": {},
            "avg_latency": 0,
            "total_latency": 0,
        }
        
        # Calculate node-specific stats
        if self.node_traces:
            # Count nodes by name
            for node in self.node_traces:
                name = node.get("node_name", "Unknown")
                if name not in node_stats["nodes_visited"]:
                    node_stats["nodes_visited"][name] = 0
                node_stats["nodes_visited"][name] += 1
                
            # Calculate latency stats if duration is available
            node_durations = [
                n.get("end_time", 0) - n.get("start_time", 0) 
                for n in self.node_traces 
                if "start_time" in n and "end_time" in n
            ]
            
            if node_durations:
                node_stats["total_latency"] = sum(node_durations)
                node_stats["avg_latency"] = node_stats["total_latency"] / len(node_durations)
                node_stats["percent_time"] = (node_stats["total_latency"] / total_duration * 100) if total_duration > 0 else 0
        
        # Transition stats
        transition_stats = {
            "transition_count": len(self.state_transitions),
            "transitions": {},
        }
        
        # Calculate transition-specific stats
        if self.state_transitions:
            # Map transitions
            for transition in self.state_transitions:
                from_state = transition.get("from_state", "Unknown")
                to_state = transition.get("to_state", "Unknown")
                transition_key = f"{from_state} → {to_state}"
                
                if transition_key not in transition_stats["transitions"]:
                    transition_stats["transitions"][transition_key] = 0
                transition_stats["transitions"][transition_key] += 1
        
        # Error stats
        error_stats = {
            "error_count": len(self.errors),
            "errors_by_type": {},
        }
        
        # Calculate error-specific stats
        if self.errors:
            # Count errors by type
            for error in self.errors:
                error_type = error.get("type", "Unknown")
                if error_type not in error_stats["errors_by_type"]:
                    error_stats["errors_by_type"][error_type] = 0
                error_stats["errors_by_type"][error_type] += 1
        
        return {
            "execution_id": self.execution_id,
            "total_duration": total_duration,
            "llm_stats": llm_stats,
            "tool_stats": tool_stats,
            "node_stats": node_stats,
            "transition_stats": transition_stats,
            "error_stats": error_stats,
        }
    
    def generate_html_report(self, output_file: Optional[str] = None) -> str:
        """Generate an HTML report with visualizations of the traced data."""
        import datetime
        
        # Generate stats
        stats = self.generate_summary_stats()
        
        # Prepare timeline data
        timeline_data = self.to_timeline_data()
        timeline_data_json = json.dumps(timeline_data)
        
        # Create HTML template
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>LangGraph Execution Report - {self.execution_id}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-timeline/7.7.0/vis-timeline-graph2d.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis-timeline/7.7.0/vis-timeline-graph2d.min.css" rel="stylesheet" type="text/css" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 5px;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .stats-container {{
            display: flex;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }}
        .stat-box {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px;
            flex: 1;
            min-width: 200px;
        }}
        .timeline-container {{
            height: 400px;
            margin: 20px 0;
            border: 1px solid #ddd;
        }}
        .chart-container {{
            height: 300px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .node {{
            background-color: #d4edda;
        }}
        .llm {{
            background-color: #cce5ff;
        }}
        .tool {{
            background-color: #fff3cd;
        }}
        .transition {{
            background-color: #d6d8db;
        }}
        .error {{
            background-color: #f8d7da;
        }}
        .completed {{
            border-left: 5px solid #28a745;
        }}
        .error {{
            border-left: 5px solid #dc3545;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LangGraph Execution Report</h1>
        <p><strong>Execution ID:</strong> {self.execution_id}</p>
        <p><strong>Date:</strong> {datetime.datetime.fromtimestamp(self.execution_start).strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Duration:</strong> {stats['total_duration']:.2f} seconds</p>
        
        <h2>Execution Summary</h2>
        <div class="stats-container">
            <div class="stat-box">
                <h3>LLM Calls</h3>
                <p>Count: {stats['llm_stats']['call_count']}</p>
                <p>Total Tokens: {stats['llm_stats']['total_tokens']['total']}</p>
                <p>Avg Latency: {stats['llm_stats']['avg_latency']:.2f}s</p>
                <p>% of Time: {stats['llm_stats']['percent_time']:.1f}%</p>
            </div>
            <div class="stat-box">
                <h3>Tool Calls</h3>
                <p>Count: {stats['tool_stats']['call_count']}</p>
                <p>Unique Tools: {len(stats['tool_stats']['tools_used'])}</p>
                <p>Avg Latency: {stats['tool_stats']['avg_latency']:.2f}s</p>
                <p>% of Time: {stats['tool_stats'].get('percent_time', 0):.1f}%</p>
            </div>
            <div class="stat-box">
                <h3>Node Executions</h3>
                <p>Count: {stats['node_stats']['node_count']}</p>
                <p>Unique Nodes: {len(stats['node_stats']['nodes_visited'])}</p>
                <p>Avg Latency: {stats['node_stats']['avg_latency']:.2f}s</p>
                <p>% of Time: {stats['node_stats'].get('percent_time', 0):.1f}%</p>
            </div>
            <div class="stat-box">
                <h3>State Transitions</h3>
                <p>Count: {stats['transition_stats']['transition_count']}</p>
                <p>Unique Paths: {len(stats['transition_stats']['transitions'])}</p>
            </div>
        </div>
        
        <h2>Execution Timeline</h2>
        <div id="timeline" class="timeline-container"></div>
        
        <h2>Node Execution Distribution</h2>
        <div class="chart-container">
            <canvas id="nodeChart"></canvas>
        </div>
        
        <h2>Tool Usage Distribution</h2>
        <div class="chart-container">
            <canvas id="toolChart"></canvas>
        </div>
        
        <h2>Transition Path Analysis</h2>
        <div class="chart-container">
            <canvas id="transitionChart"></canvas>
        </div>
        
        <h2>LLM Token Usage</h2>
        <div class="chart-container">
            <canvas id="tokenChart"></canvas>
        </div>
        
        <h2>State Transitions</h2>
        <table>
            <tr>
                <th>From State</th>
                <th>To State</th>
                <th>Count</th>
            </tr>
            {"".join([f"<tr><td>{transition.split(' → ')[0]}</td><td>{transition.split(' → ')[1]}</td><td>{count}</td></tr>" for transition, count in stats['transition_stats']['transitions'].items()])}
        </table>
        
        <h2>Node Execution Details</h2>
        <table>
            <tr>
                <th>Node Name</th>
                <th>Execution Count</th>
            </tr>
            {"".join([f"<tr><td>{node}</td><td>{count}</td></tr>" for node, count in stats['node_stats']['nodes_visited'].items()])}
        </table>
        
        <h2>Tool Usage Details</h2>
        <table>
            <tr>
                <th>Tool Name</th>
                <th>Usage Count</th>
            </tr>
            {"".join([f"<tr><td>{tool}</td><td>{count}</td></tr>" for tool, count in stats['tool_stats']['tools_used'].items()])}
        </table>
        
        {"" if not stats['error_stats']['error_count'] else '''
        <h2>Errors</h2>
        <table>
            <tr>
                <th>Error Type</th>
                <th>Count</th>
            </tr>
            ''' + "".join([f"<tr><td>{error_type}</td><td>{count}</td></tr>" for error_type, count in stats['error_stats']['errors_by_type'].items()]) + "</table>"}
    </div>

    <script>
        // Timeline visualization
        const timelineContainer = document.getElementById('timeline');
        const timelineData = {timelineData_json};
        
        const groups = [
            {{id: 'llm', content: 'LLM Calls', className: 'llm'}},
            {{id: 'tool', content: 'Tool Calls', className: 'tool'}},
            {{id: 'node', content: 'Node Executions', className: 'node'}},
            {{id: 'transition', content: 'State Transitions', className: 'transition'}}
        ];
        
        const options = {{
            groupOrder: 'content',  // Group by content
            stack: true,
            stackSubgroups: true,
            horizontalScroll: true,
            zoomKey: 'ctrlKey',
            min: {self.execution_start * 1000},
            max: {(self.execution_start + stats['total_duration']) * 1000},
            format: {{
                minorLabels: {{
                    millisecond: 'SSS',
                    second: 's.SSS',
                    minute: 'HH:mm:ss',
                    hour: 'HH:mm',
                }},
                majorLabels: {{
                    millisecond: 'HH:mm:ss',
                    second: 'D MMMM HH:mm',
                    minute: 'ddd D MMMM',
                    hour: 'ddd D MMMM',
                }}
            }}
        }};
        
        const timeline = new vis.Timeline(timelineContainer, timelineData, groups, options);
        
        // Node execution chart
        const nodeCtx = document.getElementById('nodeChart').getContext('2d');
        new Chart(nodeCtx, {{
            type: 'bar',
            data: {{
                labels: Object.keys({json.dumps(stats['node_stats']['nodes_visited'])}),
                datasets: [{{
                    label: 'Node Executions',
                    data: Object.values({json.dumps(stats['node_stats']['nodes_visited'])}),
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Execution Count'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Node Name'
                        }}
                    }}
                }}
            }}
        }});
        
        // Tool usage chart
        const toolCtx = document.getElementById('toolChart').getContext('2d');
        new Chart(toolCtx, {{
            type: 'bar',
            data: {{
                labels: Object.keys({json.dumps(stats['tool_stats']['tools_used'])}),
                datasets: [{{
                    label: 'Tool Usage',
                    data: Object.values({json.dumps(stats['tool_stats']['tools_used'])}),
                    backgroundColor: 'rgba(255, 206, 86, 0.2)',
                    borderColor: 'rgba(255, 206, 86, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Usage Count'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Tool Name'
                        }}
                    }}
                }}
            }}
        }});
        
        // Transition path chart
        const transitionCtx = document.getElementById('transitionChart').getContext('2d');
        new Chart(transitionCtx, {{
            type: 'bar',
            data: {{
                labels: Object.keys({json.dumps(stats['transition_stats']['transitions'])}),
                datasets: [{{
                    label: 'Transition Count',
                    data: Object.values({json.dumps(stats['transition_stats']['transitions'])}),
                    backgroundColor: 'rgba(153, 102, 255, 0.2)',
                    borderColor: 'rgba(153, 102, 255, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {{
                    x: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Count'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Transition Path'
                        }}
                    }}
                }}
            }}
        }});
        
        // Token usage chart
        const tokenCtx = document.getElementById('tokenChart').getContext('2d');
        new Chart(tokenCtx, {{
            type: 'pie',
            data: {{
                labels: ['Prompt Tokens', 'Completion Tokens'],
                datasets: [{{
                    label: 'Token Usage',
                    data: [{stats['llm_stats']['total_tokens']['prompt']}, {stats['llm_stats']['total_tokens']['completion']}],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 99, 132, 0.2)'
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                    }},
                    title: {{
                        display: true,
                        text: 'Total Tokens: {stats['llm_stats']['total_tokens']['total']}'
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        
        # Save to file if specified
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_template)
            
        return html_template


import os
import time
import json
from langchain.schema import BaseMessage
from langchain.callbacks.base import BaseCallbackHandler

class TracingCallbackHandler(BaseCallbackHandler):
    def __init__(self, trace_id, user_id, conversation_id, request_id, log_dir="logs"):
        self.trace_id = trace_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.request_id = request_id
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.json_data = {
            "trace_id": trace_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "questions": []
        }
        
        self.current_prompt = None
        self.start_time = None
        self.current_tool = None
        self.tool_start_time = None

    def _extract_content(self, item):
        if isinstance(item, BaseMessage):
            return item.content
        elif isinstance(item, dict):
            return item.get("content")
        elif isinstance(item, str):
            return item
        else:
            return str(item)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        self.current_prompt = [self._extract_content(p) for p in prompts]

    def on_llm_end(self, response, **kwargs):
        duration = time.time() - self.start_time
        generations = getattr(response, "generations", [])
        outputs = [self._extract_content(g[0]) for g in generations if g]

        self.json_data["questions"].append({
            "request_id": self.request_id,
            "type": "llm",
            "question": self.current_prompt[0] if self.current_prompt else None,
            "response": outputs[0] if outputs else None,
            "duration": duration
        })

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_start_time = time.time()
        self.current_tool = {
            "request_id": self.request_id,
            "type": "tool",
            "tool_name": serialized.get("name"),
            "input": input_str,
            "output": None,
            "duration": None
        }

    def on_tool_end(self, output, **kwargs):
        duration = time.time() - self.tool_start_time
        if self.current_tool:
            self.current_tool["output"] = str(output)
            self.current_tool["duration"] = duration
            self.json_data["questions"].append(self.current_tool)
            self.current_tool = None

    def on_tool_error(self, error, **kwargs):
        if self.current_tool:
            self.current_tool["error"] = str(error)
            self.json_data["questions"].append(self.current_tool)
            self.current_tool = None

    def on_llm_error(self, error, **kwargs):
        self.json_data["questions"].append({
            "request_id": self.request_id,
            "type": "llm",
            "question": self.current_prompt[0] if self.current_prompt else None,
            "error": str(error)
        })

    def save_to_json(self):
        path = os.path.join(self.log_dir, f"{self.conversation_id}.json")
        with open(path, "w") as f:
            json.dump(self.json_data, f, indent=2, ensure_ascii=False)





