"""Modelos de chat para o framework simpleagent.

Exporta ChatModel (base abstrata) e OpenAIChatModel (implementação OpenAI).
"""

from .openai import OpenAIChatModel
from .base import ChatModel

__all__ = ["OpenAIChatModel", "ChatModel"]