import time
from typing import List, Optional 
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

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
    LLM_REPETITION_PENALTY,get_logger)
logger = get_logger(__name__)

from langchain_core.callbacks.manager import CallbackManagerForLLMRun


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