"""Implementação do ChatModel para a API da OpenAI."""

from .base import ChatModel


class OpenAIChatModel(ChatModel):
    """Modelo de chat usando a API da OpenAI.

    Implementação concreta de ``ChatModel`` configurada com os defaults
    da OpenAI (base URL e variável de ambiente da API key).

    Example::

        model = OpenAIChatModel(model="gpt-4o", temperature=0.7)
        response = model.invoke(messages)
    """

    default_base_url: str = "https://api.openai.com/v1"
    """URL base da API da OpenAI."""

    default_api_key_env: str = "OPENAI_API_KEY"
    """Nome da variável de ambiente contendo a API key da OpenAI."""