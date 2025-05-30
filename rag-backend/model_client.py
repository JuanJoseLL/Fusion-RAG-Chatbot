import time
from typing import List, Optional, Any
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama

import requests
from config import (
    QWEN_API_URL,
    API_KEY,
    MODEL_NAME,
    LLM_TIMEOUT,
    LLM_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_REPETITION_PENALTY,
    get_logger,
    OLLAMA_API_URL,
    OLLAMA_MODEL_NAME
)
logger = get_logger(__name__)

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun


class CustomChatQwen(SimpleChatModel):
    """Wraps the custom Qwen API endpoint."""
    model_name: str = MODEL_NAME
    api_key: str = API_KEY
    api_url: str = QWEN_API_URL
    max_tokens: int = LLM_MAX_TOKENS
    temperature: float = LLM_TEMPERATURE
    top_k: int = LLM_TOP_K
    repetition_penalty: float = LLM_REPETITION_PENALTY
    last_usage_data: Optional[dict] = None # To store usage data from the last call
    
    @property
    def _llm_type(self) -> str:
        return "custom_chat_qwen"

    def _call(self, messages: List[BaseMessage], stop: List[str] | None = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> str:
        """Makes the API call to the Qwen endpoint."""
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                 api_messages.append({"role": "assistant", "content": msg.content})

        
        payload = {
            "model": self.model_name, # Use self.model_name here
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_k": kwargs.get("top_k", self.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
        }

        
        if stop:
             payload["stop"] = stop 
       


        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        for attempt in range(LLM_RETRIES):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=LLM_TIMEOUT)

                try:
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    if run_manager:
                         run_manager.on_llm_error(e, response=response)
                    raise e 

                data = response.json()
                
                # Extract usage data
                usage = data.get("usage")
                if usage:
                    self.last_usage_data = usage
                    logger.debug(f"Captured token usage: {usage}")
                else:
                    # Reset or set to default if not present in this response
                    self.last_usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "detail": "Usage data not provided by API in this call."}
                    logger.warning("Token usage data not found in API response.")

                if data.get("choices") and data["choices"][0].get("message") and data["choices"][0]["message"].get("content"):
                    content = data["choices"][0]["message"]["content"]
                    if run_manager:
                        # run_manager.on_llm_new_token(content) # This is for streaming, not needed here
                        pass 
                    return content
                else:
                    error_msg = f"Invalid response structure from LLM API (missing content): {data}"
                    logger.error(error_msg)
                    if run_manager:
                        run_manager.on_llm_error(ValueError(error_msg), response=response)
                    raise ValueError("Invalid LLM API response structure.")
            except requests.exceptions.Timeout as e:
                 logger.error(f"LLM request timed out (attempt {attempt+1}/{LLM_RETRIES}).")
                 if run_manager:
                    run_manager.on_llm_error(e)
                 if attempt + 1 == LLM_RETRIES: raise TimeoutError("LLM request timed out.")
                 time.sleep(2 * (attempt + 1))
            except requests.exceptions.RequestException as e:
                logger.error(f"LLM request failed (attempt {attempt+1}/{LLM_RETRIES}): {e}")

                if attempt + 1 == LLM_RETRIES: raise ConnectionError(f"LLM request failed: {e}")
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                 
                 logger.error(f"Unexpected error during LLM call (attempt {attempt+1}/{LLM_RETRIES}): {e}")
                 if run_manager:
                    run_manager.on_llm_error(e)
                 if attempt + 1 == LLM_RETRIES: raise RuntimeError(f"Unexpected LLM error: {e}")
                 time.sleep(2 * (attempt + 1))

        raise RuntimeError("Failed to get LLM response after multiple retries.")

    def get_last_usage_data(self) -> Optional[dict]:
        """Returns the token usage data from the last successful API call."""
        return self.last_usage_data

class CustomChatOllama(ChatOllama):
    """Wraps the Ollama API using Langchain's ChatOllama."""
    model: str = OLLAMA_MODEL_NAME
    base_url: str = OLLAMA_API_URL
    temperature: float = LLM_TEMPERATURE
    top_k: int = LLM_TOP_K
    # repetition_penalty is not a direct parameter for ChatOllama,
    # it's often handled within the model's specific options or not supported directly.
    # We can pass it via `model_kwargs` if the specific Ollama model supports it.

    # Initialize last_usage_data
    last_usage_data: Optional[dict] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "detail": "Usage data not yet captured."}

    def __init__(self, **kwargs):
        # Ensure model_kwargs is initialized if not provided in kwargs
        if "model_kwargs" not in kwargs:
            kwargs["model_kwargs"] = {}
        
        # Update model_kwargs with LLM_REPETITION_PENALTY if it's set
        if LLM_REPETITION_PENALTY is not None:
            kwargs["model_kwargs"].update({"repeat_penalty": LLM_REPETITION_PENALTY}) # 'repeat_penalty' is common for Ollama
        
        super().__init__(**kwargs)


    @property
    def _llm_type(self) -> str:
        return "custom_chat_ollama"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override _generate to capture response_metadata."""
        chat_result = super()._generate(messages, stop, run_manager, **kwargs)
        if chat_result.generations and chat_result.generations[0].message:
            metadata = chat_result.generations[0].message.response_metadata
            if metadata:
                prompt_tokens = metadata.get('prompt_eval_count', 0)
                completion_tokens = metadata.get('eval_count', 0)
                self.last_usage_data = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "detail": "Usage data from Ollama response_metadata."
                }
                logger.debug(f"Captured Ollama token usage: {self.last_usage_data}")
            else:
                self.last_usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "detail": "response_metadata not found in Ollama AIMessage."}
                logger.warning("response_metadata not found in Ollama AIMessage to capture token usage.")
        else:
            self.last_usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "detail": "No generations found in ChatResult."}
            logger.warning("No generations found in ChatResult from Ollama.")
        return chat_result

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override _agenerate to capture response_metadata."""
        chat_result = await super()._agenerate(messages, stop, run_manager, **kwargs)
        if chat_result.generations and chat_result.generations[0].message:
            metadata = chat_result.generations[0].message.response_metadata
            if metadata:
                prompt_tokens = metadata.get('prompt_eval_count', 0)
                completion_tokens = metadata.get('eval_count', 0)
                self.last_usage_data = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "detail": "Usage data from Ollama response_metadata."
                }
                logger.debug(f"Captured Ollama token usage (async): {self.last_usage_data}")
            else:
                self.last_usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "detail": "response_metadata not found in Ollama AIMessage (async)."}
                logger.warning("response_metadata not found in Ollama AIMessage to capture token usage (async).")
        else:
            self.last_usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "detail": "No generations found in ChatResult (async)."}
            logger.warning("No generations found in ChatResult from Ollama (async).")
        return chat_result

    def get_last_usage_data(self) -> Optional[dict]:
        """Returns the captured token usage data for Ollama."""
        logger.debug(f"Ollama usage data requested. Returning: {self.last_usage_data}")
        return self.last_usage_data

    # The _call or _acall methods are handled by the parent ChatOllama class
    # by calling _generate and _agenerate respectively.
    # We only need to override if we have very specific custom logic
    # not covered by ChatOllama's parameters.
    # For basic usage, the above configuration is sufficient.