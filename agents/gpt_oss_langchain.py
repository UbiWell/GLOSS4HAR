from typing import Optional
import requests
import os

from typing import Sequence, Optional, Mapping, Any, Union, List, Type
import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import BaseModel, PrivateAttr

class Config:
    """Configuration class for the Flask LLM Chat application"""
    
    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # LLM API configuration
    LLM_API_URL = os.getenv('LLM_API_URL', 'http://localhost:11434/api/generate')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-oss:20b')
    
    # LLM API parameters
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '20000'))
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0'))
    
    # System message for the LLM
    SYSTEM_MESSAGE = os.getenv('SYSTEM_MESSAGE', "You are a helpful assistant.")
    
    # Timeout settings
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '240'))
    STATUS_CHECK_TIMEOUT = int(os.getenv('STATUS_CHECK_TIMEOUT', '10'))
    
    @classmethod
    def get_llm_headers(cls) -> dict:
        """Get headers for LLM API requests"""
        headers = {
            'Content-Type': 'application/json'
        }
            
        return headers
    
    @classmethod
    def validate_config(cls) -> list:
        """Validate configuration and return list of warnings"""
        warnings = []
        
        if cls.LLM_API_URL == 'http://localhost:8000/v1/chat/completions':
            warnings.append("Using default localhost LLM API URL. Make sure your LLM server is running.")
        
        if cls.SECRET_KEY == 'your-secret-key-change-this-in-production':
            warnings.append("Using default secret key. Change SECRET_KEY in production.")
        
        return warnings 

class GPTOSSChatModel(BaseChatModel):
    model: Optional[str] = None
    model_capabilities: Optional[dict] = None  # optional user-provided capabilities

    # Private attribute for non-Pydantic config
    _config: Config = PrivateAttr(default_factory=Config)

    def __init__(self, model: Optional[str] = None, model_capabilities: Optional[dict] = None, **kwargs):
        super().__init__(model=model or Config().LLM_MODEL, **kwargs)
        self.model_capabilities = model_capabilities or {}
        self._config = Config()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs
    ) -> ChatResult:
        # Convert LangChain messages to a prompt string
        prompt = "\n".join([f"{m.type}: {m.content}" for m in messages])

        # Build request data
        llm_request_data = {
            'model': self.model,
            'prompt': prompt,
            'stream': False
        }
        if stop:
            llm_request_data['stop'] = stop

        # Send request
        response = requests.post(
            self._config.LLM_API_URL,
            json=llm_request_data,
            headers=self._config.get_llm_headers(),
            timeout=self._config.REQUEST_TIMEOUT
        )
        response.raise_for_status()
        resp_json = response.json()

        # Wrap the response into a proper ChatResult
        ai_msg = AIMessage(content=resp_json["response"])
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])
    
    @property
    def _llm_type(self) -> str:
        return "gpt-oss-chat"

from types import SimpleNamespace
import asyncio
from autogen_core.models import CreateResult, RequestUsage, FinishReasons
from autogen_core.tools import Tool, ToolSchema
import re
from typing import Sequence, Optional, Any, Mapping, Literal, Union, Dict
from pydantic import BaseModel
import json
from autogen_core import FunctionCall

class LangChainModelClient:
    def __init__(self):
        self.lc_model = GPTOSSChatModel()
        self.model = "gpt-oss"
        self.model_info = {
            "vision": False,
            "multimodal": False,
            "chat": True
        }

    async def create(
        self,
        messages: Sequence[BaseMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[Any] = None,
        n: int = 1,
    ) -> CreateResult:
        # Convert AutoGen messages to LangChain-compatible messages
        lc_messages = self._convert_autogen_to_langchain(messages)

        # Call GPTOSS synchronously in a thread
        try:
            result = await asyncio.to_thread(self.lc_model.invoke, lc_messages)
            content = getattr(result, "content", "")
            if content is None:
                content = ""
            # print("GPTOSS response content:", content)
        except Exception as e:
            print("Error calling GPTOSS model:", e)
            content = ""

        # Wrap usage info (dummy values here)
        usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        # Wrap the response in a CreateResult object
        # For simplicity, we ignore tool calls in this example
        thought: Optional[str] = None
        finish_reason = "stop"

        if content != "":
            thought = content

        content_value: Union[str, list[FunctionCall]] = content

        response = CreateResult(
            finish_reason=self.normalize_stop_reason(finish_reason),
            content=content_value,
            usage=usage,
            cached=False,
            logprobs=None,
            thought=thought,
        )

        return response

    def message_retrieval(self, response):
        """Retrieve the messages from the response."""
        choices = response.choices
        return [choice.message.content for choice in choices]

    def cost(self, response):
        return 0.0

    def get_usage(self, response):
        return {}

    # ------------------------------
    # Private helper to convert messages
    # ------------------------------
    def _convert_autogen_to_langchain(self, messages):
        lc_messages = []
        for m in messages:
            # Already a LangChain message
            if isinstance(m, (HumanMessage, AIMessage)):
                lc_messages.append(m)
            # Convert SystemMessage to HumanMessage
            elif getattr(m, "role", None) == "system" or m.__class__.__name__ == "SystemMessage":
                lc_messages.append(HumanMessage(content=getattr(m, "content", "")))
            else:
                # fallback: treat as string
                lc_messages.append(HumanMessage(content=str(getattr(m, "content", m))))
        return lc_messages

    def normalize_name(self, name: str) -> str:
        """
        LLMs sometimes ask functions while ignoring their own format requirements, this function should be used to replace invalid characters with "_".

        Prefer _assert_valid_name for validating user configuration or input
        """
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]

    def normalize_stop_reason(self, stop_reason: str | None) -> FinishReasons:
        if stop_reason is None:
            return "unknown"

        # Convert to lower case
        stop_reason = stop_reason.lower()

        KNOWN_STOP_MAPPINGS: Dict[str, FinishReasons] = {
            "stop": "stop",
            "end_turn": "stop",
            "tool_calls": "function_calls",
        }

        return KNOWN_STOP_MAPPINGS.get(stop_reason, "unknown")
    
# test the model
if __name__ == "__main__":
    model = GPTOSSChatModel()
    GPTOSSChatModel.model_rebuild()

    messages = [
        HumanMessage(content="How are you feeling?"),
        AIMessage(content="I'm fine, thank you! How can I assist you today?")
    ]
    
    response = model.invoke(messages)
    print(response.content)  # Should print the response from the model

    # from autogen_agentchat.agents import AssistantAgent
    # import asyncio

    # # Assuming LangChainModelClient is defined
    # client = LangChainModelClient()
    # agent = AssistantAgent(
    #     name="coding_agent",
    #     model_client=client,
    #     system_message="You are a coding assistant that helps with Python code. only output one chunk of code at a time.",
    # )

    # async def main():
    #     result = await agent.run(task="can you write python code to add two numbers?")
    #     # The last message contains the final text
    #     final_text = result.messages[-1].content
    #     print(final_text)

    # asyncio.run(main())
