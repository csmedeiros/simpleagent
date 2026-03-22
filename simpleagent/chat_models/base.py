"""Módulo base para modelos de chat.

Este módulo define a classe base abstrata ChatModel que serve como interface
unificada para diferentes provedores de modelos de linguagem (OpenAI, Anthropic, etc).

A implementação usa a OpenAI Responses API como padrão de comunicação, permitindo
que diferentes provedores sejam acessados através de uma interface consistente.
Subclasses devem definir os atributos default_base_url e default_api_key_env
específicos de cada provedor.
"""
import os
from dotenv import load_dotenv
load_dotenv()
import openai
from ..messages import AIMessage, Message, ToolCall, ToolMessage
from abc import ABC
from typing import Any, Generator, AsyncGenerator


class ChatModel(ABC):
    """Classe base abstrata para modelos de linguagem.

    Usa a OpenAI Responses API (client.responses.create) para comunicação.
    Subclasses definem default_base_url e default_api_key_env para cada provedor.

    Atributos de classe (definidos nas subclasses):
    - default_base_url: URL base da API do provedor (ex: "https://api.openai.com/v1")
    - default_api_key_env: nome da variável de ambiente com a API key (ex: "OPENAI_API_KEY")
    """

    default_base_url: str
    """URL base da API do provedor (ex: 'https://api.openai.com/v1')."""

    default_api_key_env: str = "MODEL_API_KEY"
    """Nome da variável de ambiente contendo a API key (ex: 'OPENAI_API_KEY')."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, **kwargs) -> None:
        """Inicializa o modelo de chat.

        Args:
            model: Nome do modelo a ser usado (ex: 'gpt-4o', 'claude-3-5-sonnet', 'o3').
            base_url: URL base da API. Se None, usa default_base_url da subclasse.
            api_key: Chave de API. Se None, busca em os.getenv(default_api_key_env).
            **kwargs: Parâmetros adicionais específicos do modelo (ex: temperature,
                reasoning_effort, max_tokens) que serão passados em todas as chamadas à API.

        Atributos de instância criados:
            model: Nome do modelo configurado.
            model_kwargs: Dicionário com parâmetros extras capturados de **kwargs.
            client: Cliente OpenAI configurado para comunicação com a API.
        """
        # Nome do modelo (ex: "gpt-4o", "gpt-5.2", "o3")
        self.model = model

        # **kwargs captura parâmetros extras específicos do modelo
        # (ex: temperature, reasoning_effort, max_tokens)
        # que serão repassados em toda chamada à API via **self.model_kwargs
        self.model_kwargs = kwargs

        # Cria o client OpenAI.
        # Se base_url/api_key forem passados, sobrescrevem os defaults da subclasse.
        resolved_base_url = base_url or self.default_base_url
        resolved_api_key = api_key or os.getenv(self.default_api_key_env)

        self.client = openai.Client(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
        )

        # Client assíncrono para os métodos async (astream, etc).
        self.async_client = openai.AsyncClient(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
        )

    def _parse_response(self, response) -> list[AIMessage | ToolCall]:
        """Parseia o response da Responses API e retorna items para o histórico.

        A Responses API retorna response.output como lista de items tipados:
        - type="reasoning": resumo do raciocínio (modelos o-series)
        - type="message": resposta de texto do modelo
        - type="function_call": o modelo quer chamar uma tool

        Retorna uma lista de AIMessage e/ou ToolCall, na ordem:
        1. AIMessage (se teve content ou reasoning) — sempre primeiro
        2. ToolCall(s) — um para cada function_call
        """
        tool_calls = []
        reasoning = ""
        content = ""

        for item in response.output:
            match item.type:
                # Reasoning
                # item.summary é uma lista de objetos com .text contendo
                # pedaços do resumo do raciocínio.
                case "reasoning":
                    for s in item.summary:
                        reasoning += s.text + "\n"

                # Message: a resposta de texto do modelo.
                # item.content é uma lista de content blocks.
                # Cada block tem .type ("output_text", "refusal", etc).
                case "message":
                    for c in item.content:
                        if c.type == "output_text":
                            content += c.text

                # Function call: o modelo quer chamar uma tool.
                # Cada function_call vira um ToolCall separado no histórico.
                # - item.call_id: ID para parear com o ToolMessage de resposta
                # - item.name: nome da tool (ex: "search_web")
                # - item.arguments: JSON string com os argumentos
                case "function_call":
                    tool_calls.append(ToolCall(
                        call_id=item.call_id,
                        id=item.id,
                        arguments=item.arguments,
                        name=item.name,
                    ))

        # Monta a lista de items para adicionar ao histórico.
        # AIMessage vem primeiro (se existir), depois os ToolCalls.
        result: list[AIMessage | ToolCall] = []

        if content or reasoning:
            result.append(AIMessage(
                content=content or None,
                reasoning_content=reasoning or None,
            ))

        result.extend(tool_calls)

        return result

    def invoke(self, messages: list[Message | ToolCall | ToolMessage | dict], tools: list[dict] | None = None) -> list[AIMessage | ToolCall]:
        """Chama o modelo de forma síncrona e retorna os items de resposta.

        Similar ao .invoke() do LangChain.

        Args:
            messages: histórico de conversação (Message, ToolCall, ToolMessage ou dicts)
            tools: lista de tools no formato da Responses API (geradas por Tool.to_openai_tool())

        Returns:
            Lista de AIMessage e/ou ToolCall para adicionar ao histórico.
            - Se só AIMessage: o modelo deu a resposta final
            - Se ToolCall(s): o modelo quer chamar tool(s), precisa executar e continuar o loop
        """
        # Converte todos os items do histórico para dict.
        # Cada classe (Message, ToolCall, ToolMessage) tem to_dict() que gera
        # o formato correto para a Responses API.
        # Se já for dict (ex: passado manualmente), usa como está.
        input_messages = [
            m.to_dict() if isinstance(m, (Message, ToolCall, ToolMessage)) else m
            for m in messages
        ]

        response = self.client.responses.create(
            model=self.model,
            input=input_messages,
            tools=tools,
            **self.model_kwargs,
        )

        return self._parse_response(response)

    def stream(self, messages: list[Message | ToolCall | ToolMessage | dict], tools: list[dict] | None = None) -> Generator[dict, None, None]:
        """Faz streaming da resposta do modelo, yielding chunks conforme chegam.

        Similar ao .stream() do LangChain.

        Args:
            messages: histórico de conversação
            tools: lista de tools no formato da Responses API

        Yields dicts com tipos diferentes conforme os eventos chegam:
            {"type": "text", "content": "pedaço"}         -> chunk de texto da resposta
            {"type": "reasoning", "content": "pedaço"}     -> chunk do raciocínio (modelos o-series)
            {"type": "tool_call", "tool_call": ToolCall}   -> tool call completa (quando args terminam)
            {"type": "end", "items": [...]}                -> fim do stream, com items finais completos

        Uso:
            for chunk in model.stream(messages):
                if chunk["type"] == "text":
                    print(chunk["content"], end="", flush=True)
                elif chunk["type"] == "end":
                    final_items = chunk["items"]
        """
        input_messages = [
            m.to_dict() if isinstance(m, (Message, ToolCall, ToolMessage)) else m
            for m in messages
        ]

        stream = self.client.responses.create(
            model=self.model,
            input=input_messages,
            tools=tools,
            stream=True,
            **self.model_kwargs,
        )

        # Acumuladores para montar os items finais quando o stream terminar
        content = ""
        reasoning = ""
        tool_calls = []

        # Mapa de output_index -> ResponseFunctionToolCall.
        # O streaming da Responses API emite os dados de uma function_call em 2 etapas:
        #   1. "response.output_item.added" — traz o item com metadados (call_id, id, name)
        #      mas com arguments vazio ("")
        #   2. "response.function_call_arguments.done" — traz os arguments completos
        #      mas NÃO repete call_id (e name pode vir None)
        #
        # A estratégia é guardar o item da etapa 1 e, na etapa 2, apenas preencher
        # os arguments nele. Assim o item pendente é a única fonte de verdade.
        pending_tool_items: dict[int, Any] = {}

        # A Responses API emite eventos tipados via Server-Sent Events.
        # Cada evento tem event.type indicando o que aconteceu.
        for event in stream:
            match event.type:
                # Delta de texto: um pedaço da resposta chegou.
                # event.delta contém o texto incremental.
                case "response.output_text.delta":
                    content += event.delta
                    yield {"type": "text", "content": event.delta}

                # Delta de reasoning: um pedaço do raciocínio chegou.
                # Só emitido por modelos o-series com reasoning_effort configurado.
                case "response.reasoning_summary_text.delta":
                    reasoning += event.delta
                    yield {"type": "reasoning", "content": event.delta}

                # Novo item de output adicionado ao response.
                # Para function_calls, o item (ResponseFunctionToolCall) já traz
                # os metadados completos: call_id, id, name.
                # Guardamos indexado por output_index para parear com arguments.done.
                case "response.output_item.added":
                    if event.item.type == "function_call":
                        pending_tool_items[event.output_index] = event.item

                # Arguments da function call completos.
                # Buscamos o item pendente (que tem call_id, id, name) e construímos
                # o ToolCall com os arguments que acabaram de chegar.
                case "response.function_call_arguments.done":
                    item = pending_tool_items[event.output_index]
                    tc = ToolCall(
                        call_id=item.call_id,
                        id=item.id,
                        name=item.name,
                        arguments=event.arguments,
                    )
                    tool_calls.append(tc)
                    yield {"type": "tool_call", "tool_call": tc}

                # Resposta completa: o stream terminou.
                # Montamos os items finais (igual ao _parse_response) e yielding.
                case "response.completed":
                    items: list[AIMessage | ToolCall] = []
                    if content or reasoning:
                        items.append(AIMessage(
                            content=content or None,
                            reasoning_content=reasoning or None,
                        ))
                    items.extend(tool_calls)
                    yield {"type": "end", "items": items}

    async def astream(self, messages: list[Message | ToolCall | ToolMessage | dict], tools: list[dict] | None = None) -> AsyncGenerator[dict, None]:
        """Versão assíncrona do stream(). Faz streaming da resposta do modelo.

        Args:
            messages: histórico de conversação
            tools: lista de tools no formato da Responses API

        Yields:
            Dicts com os mesmos tipos do stream() síncrono:
            ``{"type": "text", "content": "..."}``
            ``{"type": "reasoning", "content": "..."}``
            ``{"type": "tool_call", "tool_call": ToolCall}``
            ``{"type": "end", "items": [...]}``
        """
        # Converte todos os items do histórico para dict (igual ao stream síncrono).
        input_messages = [
            m.to_dict() if isinstance(m, (Message, ToolCall, ToolMessage)) else m
            for m in messages
        ]

        # Usa o async_client para criar o stream.
        # O await retorna um async iterator que podemos consumir com async for.
        # A diferença do síncrono é que aqui cada chunk chega sem bloquear o event loop,
        # permitindo que outras coroutines rodem enquanto esperamos a próxima resposta da API.
        stream = await self.async_client.responses.create(
            model=self.model,
            input=input_messages,
            tools=tools,
            stream=True,
            **self.model_kwargs,
        )

        # Acumuladores para montar os items finais quando o stream terminar.
        # São idênticos aos do stream() síncrono.
        content = ""
        reasoning = ""
        tool_calls = []

        # Mapa de output_index -> ResponseFunctionToolCall.
        # Mesmo mecanismo do stream() síncrono — guarda o item completo
        # (call_id, id, name) e depois preenche os arguments quando chegam.
        pending_tool_items: dict[int, Any] = {}

        # async for: consome o stream de forma assíncrona.
        # Cada event é um Server-Sent Event da Responses API.
        # A lógica de parsing é idêntica ao stream() síncrono — a única diferença
        # é o uso de async for ao invés de for, e yield dentro de async generator.
        async for event in stream:
            match event.type:
                # Delta de texto: um pedaço da resposta chegou.
                # event.delta contém o texto incremental.
                case "response.output_text.delta":
                    content += event.delta
                    yield {"type": "text", "content": event.delta}

                # Delta de reasoning: um pedaço do raciocínio chegou.
                # Só emitido por modelos com reasoning_effort configurado.
                case "response.reasoning_summary_text.delta":
                    reasoning += event.delta
                    yield {"type": "reasoning", "content": event.delta}

                # Novo item de output adicionado ao response.
                # Para function_calls, o item (ResponseFunctionToolCall) já traz
                # os metadados completos: call_id, id, name.
                # Guardamos indexado por output_index para parear com arguments.done.
                case "response.output_item.added":
                    if event.item.type == "function_call":
                        pending_tool_items[event.output_index] = event.item

                # Arguments da function call completos.
                # Buscamos o item pendente (que tem call_id, id, name) e construímos
                # o ToolCall com os arguments que acabaram de chegar.
                case "response.function_call_arguments.done":
                    item = pending_tool_items[event.output_index]
                    tc = ToolCall(
                        call_id=item.call_id,
                        id=item.id,
                        name=item.name,
                        arguments=event.arguments,
                    )
                    tool_calls.append(tc)
                    yield {"type": "tool_call", "tool_call": tc}

                # Resposta completa: o stream terminou.
                # Montamos os items finais (igual ao _parse_response) e yielding.
                case "response.completed":
                    items: list[AIMessage | ToolCall] = []
                    if content or reasoning:
                        items.append(AIMessage(
                            content=content or None,
                            reasoning_content=reasoning or None,
                        ))
                    items.extend(tool_calls)
                    yield {"type": "end", "items": items}
