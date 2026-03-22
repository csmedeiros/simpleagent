"""Classes de mensagens para a OpenAI Responses API.

Este módulo define as classes que representam os diferentes tipos de items
no histórico de conversação da OpenAI Responses API. Cada classe sabe se
serializar para o formato JSON que a API espera via to_dict().

O histórico do agente é uma lista mista desses tipos:
    [SystemMessage, HumanMessage, AIMessage, ToolCall, ToolMessage, ...]

Formatos da Responses API:
    - SystemMessage/HumanMessage: {"role": "system"|"user", "content": "..."}
    - AIMessage: {"type": "message", "role": "assistant", "content": "..."}
    - ToolCall: {"type": "function_call", "call_id": "...", "name": "...", ...}
    - ToolMessage: {"type": "function_call_output", "call_id": "...", "output": "..."}
"""
from typing import Any, Literal, Dict, List
from pydantic import BaseModel


class Message(BaseModel):
    """Classe base para mensagens com role (system, user, assistant).

    Usada como base para SystemMessage, HumanMessage e AIMessage.
    Serializa para {"role": "...", "content": "..."}.
    """

    role: str
    """Papel da mensagem na conversa (system, user ou assistant)."""

    content: str | List[Dict[str, Any]] | None = None
    """Conteúdo da mensagem (texto ou lista de conteúdo multimodal)."""

    def to_dict(self) -> dict[str, Any]:
        """Serializa a mensagem para o formato da API.

        Returns:
            Dict contendo role e content (campos None são excluídos).
        """
        return self.model_dump(exclude_none=True)


class HumanMessage(Message):
    """Mensagem do usuário. Serializa para {"role": "user", "content": "..."}."""

    role: Literal["user"] = "user"
    """Papel fixo como 'user' para mensagens do usuário."""


class ToolCall(BaseModel):
    """Representa uma chamada de tool feita pelo LLM.

    NÃO herda de Message porque na Responses API é um item separado
    no histórico, com formato próprio baseado em "type" (não "role").

    Serializa para:
    {"type": "function_call", "call_id": "...", "name": "...", "arguments": "...", ...}
    """

    arguments: str
    """String JSON com os argumentos da chamada (ex: '{"query": "python"}')."""

    call_id: str
    """ID único gerado pela API para parear com o ToolMessage de resposta."""

    name: str
    """Nome da tool chamada (ex: 'search_web')."""

    type: Literal["function_call"] = "function_call"
    """Tipo do item na Responses API (sempre 'function_call')."""

    status: Literal["completed"] = "completed"
    """Status da chamada (sempre 'completed' ao reenviar no histórico)."""

    id: str
    """ID do item no response."""

    def to_dict(self) -> dict[str, Any]:
        """Serializa a chamada de tool para o formato da API.

        Returns:
            Dict contendo type, call_id, name, arguments, status e id.
        """
        return self.model_dump(exclude_none=True)


class AIMessage(Message):
    """Mensagem de resposta do LLM (assistant).

    IMPORTANTE: o to_dict() gera formato da Responses API, que exige
    "type": "message" além de "role": "assistant".
    Isso é diferente da Chat Completions API que usa apenas "role".

    Serializa para: {"type": "message", "role": "assistant", "content": "..."}
    """

    role: Literal["assistant"] = "assistant"
    """Papel fixo como 'assistant' para mensagens do LLM."""

    reasoning_content: str | None = None
    """Raciocínio do modelo (apenas modelos o-series como o3, o4-mini)."""

    def to_dict(self) -> dict[str, Any]:
        """Serializa a mensagem do assistant para o formato da Responses API.

        O reasoning_content NÃO é reenviado para a API - é apenas para
        consumo local (exibir para o usuário, logs, etc).

        Returns:
            Dict contendo type='message', role='assistant' e content (se presente).
        """
        data: dict[str, Any] = {"type": "message", "role": self.role}
        if self.content:
            data["content"] = self.content
        return data


class SystemMessage(Message):
    """Mensagem de sistema (instruções para o LLM).

    Serializa para {"role": "system", "content": "..."}.
    Na Responses API, system messages usam "role" diretamente (sem "type").
    """

    role: Literal["system"] = "system"
    """Papel fixo como 'system' para mensagens de sistema."""


class ToolMessage(BaseModel):
    """Resultado da execução de uma tool, enviado de volta para o LLM.

    NÃO herda de Message porque a Responses API usa um formato diferente
    para este tipo de item - baseado em "type" (não "role").

    Serializa para:
    {"type": "function_call_output", "call_id": "...", "output": "..."}
    """

    type: Literal["function_call_output"] = "function_call_output"
    """Tipo do item na Responses API (sempre 'function_call_output')."""

    call_id: str
    """ID que corresponde ao call_id do ToolCall que originou esta resposta."""

    output: str
    """Resultado da execução como string (dict/list devem ser convertidos para JSON)."""

    def to_dict(self) -> dict[str, Any]:
        """Serializa o resultado da tool para o formato da API.

        Returns:
            Dict contendo type='function_call_output', call_id e output.
        """
        return self.model_dump(exclude_none=True)
