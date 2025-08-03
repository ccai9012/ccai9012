# utils/llm_utils.py

import os
import getpass
from typing import Optional
from langchain_deepseek import ChatDeepSeek


def get_deepseek_api_key(env_var: str = "DEEPSEEK_API_KEY") -> str:
    """
    Ensure the DEEPSEEK_API_KEY is set, or prompt user to input it securely.
    """
    api_key = os.getenv(env_var)
    if not api_key:
        api_key = getpass.getpass(f"Enter your {env_var}: ")
        os.environ[env_var] = api_key
    return api_key


def initialize_llm(
    model: str = "deepseek-chat",
    temperature: float = 0.5,
    max_tokens: int = 2048,
    timeout: int = 30,
    max_retries: int = 3,
    api_key: Optional[str] = None
) -> ChatDeepSeek:
    """
    Initialize the DeepSeek Chat model with optional parameters.
    """
    if not api_key:
        api_key = get_deepseek_api_key()
    
    return ChatDeepSeek(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        api_key=api_key,
    )


def ask_llm(prompt: str, llm: Optional[ChatDeepSeek] = None):
    """
    Stream response from LLM based on a single prompt.
    """
    if llm is None:
        llm = initialize_llm()

    print(f"\n📌 Prompt:\n{prompt}\n")
    for chunk in llm.stream(prompt):
        print(chunk.text(), end="")
    print("\n")


def generate_multiple_outputs(
    prompt: str,
    num_outputs: int = 3,
    temperature: float = 1.0,
    llm_params: Optional[dict] = None
):
    """
    Generate multiple outputs with varying sampling (temperature).
    """
    llm_params = llm_params or {}
    llm = initialize_llm(temperature=temperature, **llm_params)

    for i in range(num_outputs):
        print(f"\n Output #{i + 1} — Temperature {temperature}")
        print(f"📌 Prompt:\n{prompt}\n")
        for chunk in llm.stream(prompt):
            print(chunk.text(), end="")
        print("\n")
