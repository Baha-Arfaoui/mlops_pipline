from typing import Dict, List, Any, Optional, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

class DetailedTracingCallbackHandler(BaseCallbackHandler):
    """Callback handler for tracing LangGraph agent execution."""
    
    def __init__(self):
        """Initialize the callback handler with storage for traces."""
        super().__init__()
        self.traces = []
        self.current_chain_trace = {}
        self.llm_calls = []
        self.tool_calls = []
        self.agent_actions = []
    
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Log when a chain starts."""
        chain_info = {
            "chain_type": serialized.get("name", "Unknown Chain"),
            "inputs": inputs,
            "start_time": self._get_timestamp(),
            "metadata": kwargs.get("metadata", {}),
            "run_id": kwargs.get("run_id", ""),
        }
        self.current_chain_trace = chain_info
        self.traces.append(chain_info)
    
    def on_chain_end(
        self, outputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Log when a chain ends."""
        if self.current_chain_trace:
            self.current_chain_trace.update({
                "outputs": outputs,
                "end_time": self._get_timestamp(),
                "duration": self._get_timestamp() - self.current_chain_trace.get("start_time", self._get_timestamp()),
            })
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Log when an LLM starts."""
        llm_info = {
            "llm_type": serialized.get("name", "Unknown LLM"),
            "prompts": prompts,
            "start_time": self._get_timestamp(),
            "metadata": kwargs.get("metadata", {}),
            "run_id": kwargs.get("run_id", ""),
            "invocation_params": serialized.get("kwargs", {}),
        }
        self.llm_calls.append(llm_info)
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Log when an LLM ends."""
        if self.llm_calls:
            self.llm_calls[-1].update({
                "response": response.dict(),
                "end_time": self._get_timestamp(),
                "duration": self._get_timestamp() - self.llm_calls[-1].get("start_time", self._get_timestamp()),
                "token_usage": response.llm_output.get("token_usage", {}) if response.llm_output else {},
            })
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Log when a tool starts."""
        tool_info = {
            "tool_name": serialized.get("name", "Unknown Tool"),
            "input": input_str,
            "start_time": self._get_timestamp(),
            "metadata": kwargs.get("metadata", {}),
            "run_id": kwargs.get("run_id", ""),
        }
        self.tool_calls.append(tool_info)
    
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Log when a tool ends."""
        if self.tool_calls:
            self.tool_calls[-1].update({
                "output": output,
                "end_time": self._get_timestamp(),
                "duration": self._get_timestamp() - self.tool_calls[-1].get("start_time", self._get_timestamp()),
            })
    
    def on_agent_action(
        self, action: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Log agent actions."""
        action_info = {
            "action": action,
            "timestamp": self._get_timestamp(),
            "run_id": kwargs.get("run_id", ""),
            "metadata": kwargs.get("metadata", {}),
        }
        self.agent_actions.append(action_info)
    
    def on_agent_finish(
        self, finish: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Log agent finish."""
        if self.agent_actions:
            self.agent_actions[-1].update({
                "final_result": finish,
                "finish_time": self._get_timestamp(),
            })
    
    def on_text(
        self, text: str, **kwargs: Any
    ) -> Any:
        """Log text outputs."""
        text_info = {
            "text": text,
            "timestamp": self._get_timestamp(),
            "run_id": kwargs.get("run_id", ""),
            "metadata": kwargs.get("metadata", {}),
        }
        self.traces.append(text_info)
    
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Log when a chat model starts."""
        chat_info = {
            "model_type": serialized.get("name", "Unknown Chat Model"),
            "messages": [[m.dict() for m in msg_list] for msg_list in messages],
            "start_time": self._get_timestamp(),
            "metadata": kwargs.get("metadata", {}),
            "run_id": kwargs.get("run_id", ""),
            "invocation_params": serialized.get("kwargs", {}),
        }
        self.llm_calls.append(chat_info)
    
    def _get_timestamp(self):
        """Get current timestamp."""
        import time
        return time.time()
    
    def get_traced_data(self) -> Dict[str, Any]:
        """Get all traced data in a structured format."""
        return {
            "traces": self.traces,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "agent_actions": self.agent_actions,
        }
    
    def clear(self):
        """Clear all stored traces."""
        self.traces = []
        self.current_chain_trace = {}
        self.llm_calls = []
        self.tool_calls = []
        self.agent_actions = []
