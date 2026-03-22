"""Implementação base de um agente ReAct (Reasoning and Acting).

Este módulo fornece a classe Agent, que implementa o padrão ReAct loop:
um ciclo iterativo onde o LLM alterna entre raciocínio (gerar respostas) e
ação (chamar ferramentas/tools) até resolver a tarefa do usuário.

O agente usa a Responses API da OpenAI, onde ToolCall e ToolMessage são
items separados no histórico de mensagens (diferente da Chat Completions API).
"""

import json
from simpleagent.chat_models.base import ChatModel
from simpleagent.agents.tools.base import Tool
from simpleagent.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, ToolCall, Message
from typing import Callable, List, AsyncGenerator, Any

# ============================================================================
# COMO UM AGENTE FUNCIONA (ReAct Loop)
# ============================================================================
#
# Um agente é um loop que alterna entre PENSAR e AGIR usando a Responses API:
#
#   1. O usuário envia uma mensagem
#   2. O LLM recebe o histórico + tools disponíveis
#   3. O LLM decide:
#      - Responder diretamente (retorna AIMessage) -> fim do loop
#      - Chamar tool(s) (retorna ToolCall) -> continua o loop
#   4. Se chamou tool(s): executa cada uma, cria ToolMessage com o resultado
#   5. Adiciona tudo ao histórico e volta ao passo 2
#
# O histórico é uma lista mista de tipos (formato Responses API):
#   [SystemMessage, HumanMessage, AIMessage, ToolCall, ToolMessage, ...]
#
# Cada tipo se serializa para o formato correto da API via to_dict().
# Na Responses API, ToolCall e ToolMessage são items separados no histórico
# (diferente da Chat Completions API onde tool_calls era um campo do AIMessage).
# ============================================================================


class Agent:
    """Agente autônomo que executa tarefas usando LLMs e ferramentas (ReAct loop).

    O Agent implementa o padrão ReAct (Reasoning and Acting), alternando entre:
    - Raciocínio: o LLM analisa a situação e decide o próximo passo
    - Ação: o LLM chama ferramentas para obter informações ou executar tarefas

    O ciclo continua até o LLM produzir uma resposta final ou atingir o limite
    de iterações. O histórico completo de mensagens é mantido usando os tipos
    da Responses API (SystemMessage, HumanMessage, AIMessage, ToolCall, ToolMessage).

    Attributes:
        model: O modelo de linguagem usado pelo agente para raciocínio e decisões.
        tools: Lista de ferramentas disponíveis para o agente executar.
        tools_map: Dicionário mapeando nome -> Tool para lookup rápido durante execução.
        openai_tools: Lista de tools no formato JSON da Responses API para passar ao LLM.
        system_prompt: Instruções/personalidade do agente, enviadas como primeira mensagem.
        max_iterations: Limite de iterações do loop para prevenir execução infinita.

    Example:
        >>> from chat_models.openai import OpenAIChatModel
        >>> from agents.base import Agent
        >>>
        >>> def search_web(query: str) -> str:
        ...     '''Busca informações na web.'''
        ...     return f"Resultados para: {query}"
        >>>
        >>> model = OpenAIChatModel(model="gpt-4")
        >>> agent = Agent(
        ...     model=model,
        ...     tools=[search_web],
        ...     system_prompt="Você é um assistente prestativo.",
        ...     max_iterations=5
        ... )
        >>>
        >>> messages = agent.run("Qual a capital do Brasil?")
        >>> print(messages[-1].content)  # Última mensagem é a resposta final
    """

    model: ChatModel
    """O modelo de linguagem usado pelo agente."""

    tools: list[Tool]
    """Lista de ferramentas disponíveis para o agente."""

    tools_map: dict[str, Tool]
    """Mapa de nome -> Tool para lookup rápido durante execução."""

    openai_tools: list[dict] | None
    """Lista de tools no formato JSON da Responses API (ou None se sem tools)."""

    system_prompt: str
    """Instruções e personalidade do agente, enviadas como primeira mensagem."""

    max_iterations: int
    """Limite de iterações do loop para prevenir execução infinita."""

    def __init__(
        self,
        model: ChatModel,
        tools: list[Tool | Callable] | None = None,
        system_prompt: str = "",
        max_iterations: int = 10,
    ) -> None:
        """Inicializa o agente com modelo, ferramentas e configurações.

        Args:
            model: Instância de ChatModel (ex: OpenAIChatModel) usada para raciocínio.
            tools: Lista de ferramentas disponíveis. Aceita objetos Tool ou funções Python.
                Funções são automaticamente encapsuladas em Tool.
            system_prompt: Instruções iniciais que definem comportamento e personalidade.
                Enviado como SystemMessage no início do histórico.
            max_iterations: Número máximo de iterações do loop antes de forçar parada.
                Previne loops infinitos se o LLM não convergir para resposta final.
        """
        # O modelo de linguagem que o agente usa para "pensar".
        # É uma instância de ChatModel (ex: OpenAIChatModel).
        # O agente delega todas as chamadas de LLM para este objeto.
        self.model = model

        # Aceita tanto Tool já instanciada quanto função Python pura.
        # Se receber uma Callable (função), encapsula automaticamente em Tool.
        # Tool cuida de extrair nome, docstring e schema JSON dos parâmetros.
        self.tools = [Tool(func=tool) if isinstance(tool, Callable) else tool for tool in tools] if tools else []

        # tools_map: dicionário de nome -> Tool para lookup rápido.
        # Quando o LLM retorna um ToolCall com name="search_web",
        # usamos este mapa para encontrar a Tool correspondente e executar.
        # Ex: {"search_web": Tool(search_web), "calculator": Tool(calculator)}
        self.tools_map: dict[str, Tool] = {tool.name: tool for tool in self.tools}

        # Converte cada Tool para o formato JSON da Responses API.
        # É passado no parâmetro tools= de toda chamada ao LLM.
        # Formato: [{"type": "function", "name": "...", "description": "...", "parameters": {...}}, ...]
        # Se não tiver tools, passa None (o LLM não vai tentar chamar nenhuma).
        self.openai_tools = [tool.to_openai_tool() for tool in self.tools] or None

        # O system prompt define a "personalidade" e instruções do agente.
        # É a primeira mensagem no histórico e guia o comportamento do LLM.
        self.system_prompt = system_prompt

        # Limite de segurança para evitar loops infinitos.
        # Se o LLM ficar chamando tools sem nunca dar uma resposta final,
        # o loop para após max_iterations.
        self.max_iterations = max_iterations

    def run(self, user_input: str) -> List[Message | ToolCall | ToolMessage]:
        """Executa o agente com a mensagem do usuário e retorna o histórico completo.

        Este é o ponto de entrada principal. Ele:
        1. Monta o histórico inicial (system + user message)
        2. Entra no loop do agente
        3. Retorna o histórico completo com todas as mensagens trocadas
        """
        # Monta o histórico inicial.
        # Começa com o system prompt (instruções) + mensagem do usuário.
        messages = []

        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        messages.append(HumanMessage(content=user_input))

        # Entra no loop principal do agente
        return self._loop(messages)

    def _loop(self, messages: list) -> List[Message | ToolCall | ToolMessage]:
        """O loop principal do agente (ReAct loop).

        A cada iteração:
        1. Chama o LLM com o histórico atual + tools disponíveis
        2. Recebe uma lista de items (AIMessage e/ou ToolCall)
        3. Se tem ToolCall(s): executa cada tool e adiciona tudo ao histórico
        4. Se NÃO tem ToolCall: o LLM respondeu, retorna o histórico completo
        """
        for _ in range(self.max_iterations):

            # Chama o LLM passando todo o histórico e as tools disponíveis.
            # Retorna uma lista de AIMessage e/ou ToolCall.
            # Ex sem tool: [AIMessage(content="Olá!")]
            # Ex com tool: [ToolCall(name="search_web", arguments='{"query": "..."}')]
            # Ex misto:    [AIMessage(content="Vou buscar..."), ToolCall(...)]
            new = self.model.invoke(messages, tools=self.openai_tools)

            # Verifica se o LLM chamou alguma tool
            if any(isinstance(m, ToolCall) for m in new):
                # Filtra apenas os ToolCalls da resposta
                tool_calls = [m for m in new if isinstance(m, ToolCall)]

                # Executa cada tool e adiciona o ToolMessage (resultado) à lista
                for tc in tool_calls:
                    tool_response = self._execute_tool(tool_call=tc)
                    new.append(tool_response)

                # Adiciona tudo ao histórico: AIMessage (se teve), ToolCalls e ToolMessages.
                # O histórico fica: [..., AIMessage?, ToolCall, ToolMessage, ToolCall, ToolMessage, ...]
                # Na próxima iteração o LLM vê os resultados das tools e decide o que fazer.
                messages.extend(new)
            else:
                # Sem ToolCalls = o LLM deu a resposta final.
                # Adiciona ao histórico e retorna tudo.
                messages.extend(new)
                return messages

        # Se chegou aqui, atingiu o limite de iterações sem resposta final.
        # Isso é uma proteção contra loops infinitos.
        return [AIMessage(content="Limite de iterações atingido.")]

    def _execute_tool(self, tool_call: ToolCall) -> ToolMessage:
        """Executa uma tool chamada pelo LLM e retorna o resultado como ToolMessage.

        Fluxo:
        1. Extrai nome e argumentos do ToolCall
        2. Parseia os argumentos de JSON string para dict Python
        3. Busca a Tool correspondente no tools_map
        4. Executa a função com os argumentos
        5. Retorna ToolMessage com o resultado (string)

        O ToolMessage retornado precisa ter:
        - call_id: mesmo call_id do ToolCall original.
          A Responses API usa isso para parear chamada -> resultado.
        - output: resultado da execução como string
        """
        # Extrai nome e argumentos do ToolCall.
        # name: nome da tool (ex: "search_web")
        # arguments: JSON string com os args (ex: '{"query": "python"}')
        tool_name = tool_call.name
        tool_args_json = tool_call.arguments

        # Faz parse dos argumentos de JSON string para dict Python.
        # Ex: '{"query": "python", "limit": 5}' -> {"query": "python", "limit": 5}
        tool_args: dict = json.loads(tool_args_json)

        # Busca a Tool pelo nome no mapa.
        tool = self.tools_map.get(tool_name)

        if tool is None:
            # Se o LLM alucionou um nome de tool que não existe,
            # retorna erro como resultado. O LLM verá isso na próxima
            # iteração e pode tentar corrigir.
            result = f"Erro: tool '{tool_name}' não encontrada."
        else:
            # Executa a tool com os argumentos via keyword unpacking.
            # Ex: tool(query="python", limit=5) -> search_web(query="python", limit=5)
            # O __call__ do Tool delega para self.func(**kwargs).
            try:
                result = tool(**tool_args)
            except Exception as e:
                # Se a tool falhar, captura o erro e retorna como resultado.
                # Isso permite que o LLM veja o erro e tente uma abordagem diferente.
                result = f"Erro ao executar '{tool_name}': {e}"

        # Converte o resultado para string se necessário.
        # A Responses API espera output como string.
        # Se a tool retornou dict/list, converte para JSON string.
        result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return ToolMessage(
            call_id=tool_call.call_id,
            output=result_str,
        )

    async def astream_events(self, user_input: str) -> AsyncGenerator[dict[str, Any], None]:
        """Executa o agente com streaming assíncrono, emitindo eventos a cada etapa.

        Versão assíncrona do agente que faz streaming dos chunks do LLM em tempo
        real e emite eventos estruturados para cada etapa do ReAct loop.

        Args:
            user_input: Mensagem do usuário para iniciar a conversação.

        Yields:
            Dicts com campo ``event`` indicando o tipo do evento:

            - ``{"event": "on_llm_start", "iteration": int}``
                Início de uma chamada ao LLM.

            - ``{"event": "on_llm_text_delta", "content": str, "iteration": int}``
                Chunk de texto da resposta chegando via stream.

            - ``{"event": "on_llm_reasoning_delta", "content": str, "iteration": int}``
                Chunk do raciocínio do modelo (modelos o-series).

            - ``{"event": "on_tool_call_start", "tool_call": ToolCall, "iteration": int}``
                O LLM decidiu chamar uma tool (argumentos completos).

            - ``{"event": "on_tool_call_end", "tool_call": ToolCall, "tool_message": ToolMessage, "iteration": int}``
                Resultado da execução de uma tool.

            - ``{"event": "on_llm_end", "message": AIMessage, "iteration": int}``
                Resposta final do LLM (sem tool calls).

            - ``{"event": "on_agent_end", "messages": list}``
                Agente terminou. Contém o histórico completo de mensagens.

        Example::

            async for event in agent.astream_events("Pesquise sobre IA"):
                match event["event"]:
                    case "on_llm_text_delta":
                        print(event["content"], end="", flush=True)
                    case "on_tool_call_start":
                        print(f"\\nChamando: {event['tool_call'].name}")
                    case "on_tool_call_end":
                        print(f"Resultado: {event['tool_message'].output[:100]}")
        """
        # Monta o histórico inicial, igual ao run() síncrono.
        # Começa com o system prompt (instruções) + mensagem do usuário.
        messages: list = []

        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        messages.append(HumanMessage(content=user_input))

        # ====================================================================
        # ReAct Loop com Streaming Assíncrono
        # ====================================================================
        #
        # Funciona igual ao _loop() síncrono, mas com duas diferenças:
        #   1. Usa model.astream() ao invés de model.invoke() — os chunks
        #      do LLM chegam em tempo real e são emitidos como eventos
        #   2. Cada etapa do loop emite eventos estruturados para que o
        #      consumidor (ex: frontend, WebSocket) possa reagir em tempo real
        #
        # Fluxo de eventos por iteração:
        #   on_llm_start -> on_llm_text_delta* / on_llm_reasoning_delta*
        #   -> on_tool_call_start* -> on_tool_call_end* -> (próxima iteração)
        #   OU
        #   on_llm_start -> on_llm_text_delta* -> on_llm_end -> on_agent_end
        # ====================================================================
        for iteration in range(1, self.max_iterations + 1):

            # Notifica que uma nova chamada ao LLM vai começar.
            # Útil para mostrar um spinner/loading no frontend.
            yield {"event": "on_llm_start", "iteration": iteration}

            # items vai acumular os items finais do LLM nesta iteração
            # (AIMessage e/ou ToolCalls), preenchido quando chegar o evento "end".
            items: list[AIMessage | ToolCall | ToolMessage] = []

            # Consome o stream assíncrono do LLM.
            # model.astream() é um async generator que yield dicts conforme
            # os Server-Sent Events chegam da Responses API.
            # Para cada chunk, re-emitimos como evento do agente com nome padronizado.
            async for chunk in self.model.astream(messages, tools=self.openai_tools):
                match chunk["type"]:
                    # Chunk de texto da resposta.
                    # Re-emitimos como on_llm_text_delta para o consumidor
                    # poder printar/exibir em tempo real.
                    case "text":
                        yield {"event": "on_llm_text_delta", "content": chunk["content"], "iteration": iteration}

                    # Chunk do raciocínio (modelos o-series: o3, o4-mini).
                    # Re-emitimos como on_llm_reasoning_delta.
                    case "reasoning":
                        yield {"event": "on_llm_reasoning_delta", "content": chunk["content"], "iteration": iteration}

                    # Tool call completa: o LLM terminou de gerar os argumentos
                    # de uma chamada de tool. Emitimos on_tool_call_start para
                    # notificar que a tool será executada.
                    case "tool_call":
                        yield {"event": "on_tool_call_start", "tool_call": chunk["tool_call"], "iteration": iteration}

                    # Stream terminou. chunk["items"] contém a lista final de
                    # AIMessage e/ou ToolCall(s), igual ao retorno do invoke().
                    case "end":
                        items = chunk["items"]

            # A partir daqui a lógica é igual ao _loop() síncrono:
            # verifica se teve tool calls e decide se continua ou para.

            # Filtra apenas os ToolCalls da resposta
            tool_calls = [m for m in items if isinstance(m, ToolCall)]

            if tool_calls:
                # O LLM quer chamar tool(s). Executa cada uma e emite
                # on_tool_call_end com o ToolCall original + o resultado.
                for tc in tool_calls:
                    tool_response = self._execute_tool(tool_call=tc)
                    items.append(tool_response)

                    # Emite o resultado da execução da tool.
                    # O consumidor pode usar isso para exibir o output da tool.
                    yield {"event": "on_tool_call_end", "tool_call": tc, "tool_message": tool_response, "iteration": iteration}

                # Adiciona tudo ao histórico (AIMessage?, ToolCalls, ToolMessages)
                # e volta para a próxima iteração do loop.
                messages.extend(items)
            else:
                # Sem ToolCalls = o LLM deu a resposta final.
                # Adiciona ao histórico.
                messages.extend(items)

                # Emite on_llm_end com a AIMessage final (se existir).
                # Isso marca que o LLM terminou de responder.
                ai_messages = [m for m in items if isinstance(m, AIMessage)]
                if ai_messages:
                    yield {"event": "on_llm_end", "message": ai_messages[0], "iteration": iteration}

                # Emite on_agent_end com o histórico completo.
                # Isso marca que o agente terminou toda a execução.
                yield {"event": "on_agent_end", "messages": messages}
                return

        # Se chegou aqui, atingiu o limite de iterações sem resposta final.
        # Isso é uma proteção contra loops infinitos (igual ao _loop síncrono).
        yield {"event": "on_agent_end", "messages": messages}
