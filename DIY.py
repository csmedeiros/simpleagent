"""
DIY.py — Demonstrativo das funcionalidades da biblioteca simpleagent.

Este script mostra como usar os 3 pilares da biblioteca:
  1. Chat Models — interação direta com LLMs (invoke, stream)
  2. Tools — converter funções Python em tools para a API
  3. Agents — agente autônomo com ReAct loop (run, astream_events)

Para rodar:
    cd simpleagent/
    python DIY.py
"""

import os
import sys
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# SETUP — Configuração do modelo
# ============================================================================
#
# OpenAIChatModel é a implementação concreta de ChatModel para a API da OpenAI.
# Qualquer provedor compatível com a OpenAI API pode ser usado (Azure, OpenRouter, etc)
# bastando passar base_url e api_key customizados.

from simpleagent.chat_models.openai import OpenAIChatModel
from simpleagent.messages import SystemMessage, HumanMessage, AIMessage, ToolCall, ToolMessage
from simpleagent.agents.tools import Tool
from simpleagent.agents import Agent

# Cria o modelo apontando para a API configurada no .env.
# **kwargs extras (como temperature) são repassados em toda chamada à API.
llm = OpenAIChatModel(
    model="gpt-4o",
    base_url=os.getenv("MODEL_BASE_URL"),
    api_key=os.getenv("MODEL_API_KEY"),
)


# ============================================================================
# 1. CHAT MODELS — Interação direta com o LLM
# ============================================================================
#
# O ChatModel oferece 3 modos de interação:
#   - invoke()  : síncrono, retorna a resposta completa de uma vez
#   - stream()  : síncrono, retorna chunks em tempo real via generator
#   - astream() : assíncrono, retorna chunks via async generator

def demo_invoke():
    """Demonstra o uso do invoke() — chamada síncrona ao LLM.

    invoke() recebe uma lista de mensagens (histórico) e retorna
    uma lista de AIMessage e/ou ToolCall.
    Sem tools, sempre retorna [AIMessage(content="...")].
    """
    print("=" * 60)
    print("1a. ChatModel.invoke() — Chamada síncrona")
    print("=" * 60)

    # O histórico é uma lista de mensagens tipadas.
    # SystemMessage define o comportamento, HumanMessage é a pergunta do usuário.
    messages = [
        SystemMessage(content="Você é um assistente conciso. Responda em no máximo 2 frases."),
        HumanMessage(content="O que é o padrão ReAct em agentes de IA?"),
    ]

    # invoke() chama a Responses API e parseia o retorno em AIMessage/ToolCall.
    # Sem tools, o LLM sempre responde diretamente com AIMessage.
    result = llm.invoke(messages)

    # result é uma lista — normalmente [AIMessage(content="...")]
    ai_message = result[0]
    print(f"\nResposta: {ai_message.content}")
    print()


def demo_stream():
    """Demonstra o uso do stream() — streaming síncrono.

    stream() é um generator que yield dicts conforme os chunks chegam:
      {"type": "text", "content": "pedaço"}      -> texto incremental
      {"type": "reasoning", "content": "pedaço"}  -> raciocínio (modelos o-series)
      {"type": "end", "items": [...]}             -> fim, com items completos
    """
    print("=" * 60)
    print("1b. ChatModel.stream() — Streaming síncrono")
    print("=" * 60)

    messages = [
        SystemMessage(content="Você é um assistente conciso."),
        HumanMessage(content="Explique em 3 bullet points o que é tool calling em LLMs."),
    ]

    print("\nResposta (streaming): ", end="")

    # stream() retorna um generator — cada iteração yield um chunk.
    # Os chunks de texto chegam incrementalmente conforme a API gera.
    for chunk in llm.stream(messages):
        match chunk["type"]:
            # Texto incremental — printamos sem newline para efeito de "digitação"
            case "text":
                print(chunk["content"], end="", flush=True)

            # Fim do stream — items contém [AIMessage] com o conteúdo completo
            case "end":
                pass

    print("\n")


# ============================================================================
# 2. TOOLS — Converter funções Python em tools para a API
# ============================================================================
#
# A classe Tool faz introspecção de uma função Python e gera automaticamente
# o JSON Schema que a OpenAI Responses API espera.
# Ela extrai: nome (func.__name__), descrição (func.__doc__),
# parâmetros (type hints + defaults → JSON Schema).

def demo_tools():
    """Demonstra a conversão de funções Python em tools.

    Tool(func) analisa a assinatura da função e gera o schema JSON
    automaticamente. Depois, to_openai_tool() formata para a API.
    """
    print("=" * 60)
    print("2. Tools — Introspecção de funções Python")
    print("=" * 60)

    # Definimos uma função Python normal com type hints e docstring.
    # A Tool vai extrair tudo automaticamente:
    #   - nome: "calcular" (de func.__name__)
    #   - descrição: "Faz um cálculo..." (de func.__doc__)
    #   - parâmetros: operacao (str, obrigatório), a (float, obrigatório), b (float, obrigatório)
    def calcular(operacao: str, a: float, b: float) -> float:
        """Faz um cálculo matemático entre dois números."""
        ops = {"soma": a + b, "subtracao": a - b, "multiplicacao": a * b, "divisao": a / b}
        return ops.get(operacao, "Operação inválida")

    # Encapsula a função em Tool — isso faz toda a introspecção.
    tool = Tool(calcular)

    print(f"\nNome:       {tool.name}")
    print(f"Descrição:  {tool.description}")
    print(f"Parâmetros: {json.dumps(tool.parameters, indent=2, ensure_ascii=False)}")

    # to_openai_tool() gera o dict pronto para passar na API.
    # Formato: {"type": "function", "name": "...", "description": "...", "parameters": {...}}
    print(f"\nFormato API:\n{json.dumps(tool.to_openai_tool(), indent=2, ensure_ascii=False)}")

    # Tool é callable — chamar tool(...) executa a função original.
    resultado = tool(operacao="soma", a=10, b=5)
    print(f"\ntool(operacao='soma', a=10, b=5) = {resultado}")
    print()


# ============================================================================
# 3. AGENT — Agente autônomo com ReAct loop
# ============================================================================
#
# O Agent combina ChatModel + Tools em um loop ReAct:
#   1. Recebe mensagem do usuário
#   2. Chama o LLM com histórico + tools disponíveis
#   3. Se o LLM pedir tools → executa e volta ao passo 2
#   4. Se o LLM responder → retorna o histórico completo
#
# Oferece 2 modos:
#   - run()             : síncrono, retorna histórico completo
#   - astream_events()  : assíncrono, emite eventos em tempo real

# Definimos as tools que o agente pode usar.
# O Agent aceita tanto Tool() quanto funções Python puras (encapsula automaticamente).

def calcular(operacao: str, a: float, b: float) -> float:
    """Faz um cálculo matemático entre dois números. Operações: soma, subtracao, multiplicacao, divisao."""
    ops = {"soma": a + b, "subtracao": a - b, "multiplicacao": a * b, "divisao": a / b}
    return ops.get(operacao, "Operação inválida")


def buscar_informacao(topico: str) -> str:
    """Busca informações sobre um tópico. Use para perguntas que requerem conhecimento factual."""
    # Simulação — em produção, conectaria a uma API real (DuckDuckGo, Google, etc)
    dados = {
        "python": "Python é uma linguagem de programação criada por Guido van Rossum em 1991.",
        "react": "ReAct é um padrão onde agentes alternam entre raciocínio e ação usando LLMs.",
        "openai": "OpenAI é uma empresa de IA que criou o GPT-4, ChatGPT e a Responses API.",
    }
    # Busca parcial pelo tópico
    for chave, valor in dados.items():
        if chave in topico.lower():
            return valor
    return f"Nenhuma informação encontrada sobre '{topico}'."


def demo_agent_run():
    """Demonstra o Agent.run() — execução síncrona do ReAct loop.

    O agente recebe a pergunta, decide se precisa de tools,
    executa-as se necessário, e retorna o histórico completo.
    """
    print("=" * 60)
    print("3a. Agent.run() — ReAct loop síncrono")
    print("=" * 60)

    # Cria o agente com modelo, tools e system prompt.
    # Funções Python são automaticamente encapsuladas em Tool.
    agent = Agent(
        model=llm,
        tools=[calcular, buscar_informacao],
        system_prompt="Você é um assistente que usa ferramentas quando necessário. Seja conciso.",
        max_iterations=5,
    )

    # run() executa o loop completo e retorna o histórico de mensagens.
    # O histórico contém todos os tipos: SystemMessage, HumanMessage, AIMessage, ToolCall, ToolMessage.
    pergunta = "Quanto é 42 * 17? E me diga o que é Python."
    print(f"\nPergunta: {pergunta}\n")

    messages = agent.run(pergunta)

    # Percorre o histórico mostrando cada etapa do agente
    print("--- Histórico do agente ---")
    for msg in messages:
        match msg:
            case SystemMessage():
                print(f"  [SYSTEM]  {msg.content[:80]}...")
            case HumanMessage():
                print(f"  [USER]    {msg.content}")
            case AIMessage():
                if msg.content:
                    print(f"  [AI]      {msg.content[:120]}...")
            case ToolCall():
                print(f"  [CALL]    {msg.name}({msg.arguments})")
            case ToolMessage():
                print(f"  [RESULT]  {msg.output[:100]}")

    # A última mensagem é a resposta final do agente
    resposta_final = [m for m in messages if isinstance(m, AIMessage) and m.content]
    if resposta_final:
        print(f"\nResposta final: {resposta_final[-1].content}")
    print()


async def demo_agent_astream_events():
    """Demonstra o Agent.astream_events() — streaming assíncrono de eventos.

    Emite eventos estruturados em tempo real para cada etapa do ReAct loop:
      on_llm_start         → nova chamada ao LLM
      on_llm_text_delta    → chunk de texto chegando
      on_tool_call_start   → LLM decidiu chamar uma tool
      on_tool_call_end     → resultado da execução da tool
      on_llm_end           → resposta final do LLM
      on_agent_end         → agente terminou
    """
    print("=" * 60)
    print("3b. Agent.astream_events() — Streaming assíncrono")
    print("=" * 60)

    agent = Agent(
        model=llm,
        tools=[calcular, buscar_informacao],
        system_prompt="Você é um assistente que usa ferramentas quando necessário. Seja conciso.",
        max_iterations=5,
    )

    pergunta = "Qual é o resultado de 128 / 4? E busque informações sobre OpenAI."
    print(f"\nPergunta: {pergunta}\n")

    # astream_events() é um async generator — consome com async for.
    # Cada evento tem um campo "event" indicando o tipo.
    async for event in agent.astream_events(pergunta):
        match event["event"]:

            # Nova iteração do loop — o LLM vai ser chamado.
            case "on_llm_start":
                print(f"--- Iteração {event['iteration']} ---")

            # Chunk de texto da resposta — printamos em tempo real.
            case "on_llm_text_delta":
                print(event["content"], end="", flush=True)

            # Chunk de raciocínio (modelos o-series como o3).
            case "on_llm_reasoning_delta":
                print(f"  [REASONING] {event['content']}", end="", flush=True)

            # O LLM decidiu chamar uma tool.
            case "on_tool_call_start":
                tc = event["tool_call"]
                print(f"  [TOOL CALL] {tc.name}({tc.arguments})")

            # A tool foi executada — mostra o resultado.
            case "on_tool_call_end":
                tm = event["tool_message"]
                print(f"  [TOOL RESULT] {tm.output[:100]}")

            # Resposta final do LLM.
            case "on_llm_end":
                print()  # newline após o streaming de texto

            # Agente terminou completamente.
            case "on_agent_end":
                print(f"\n[AGENT END] Total de mensagens no histórico: {len(event['messages'])}")

    print()


# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    print()
    print("simpleagent — Demonstrativo de Funcionalidades")
    print("=" * 60)
    print()

    # 1a. invoke() — chamada síncrona
    demo_invoke()

    # 1b. stream() — streaming síncrono
    demo_stream()

    # 2. Tools — introspecção de funções
    demo_tools()

    # 3a. Agent.run() — ReAct loop síncrono
    demo_agent_run()

    # 3b. Agent.astream_events() — streaming assíncrono
    asyncio.run(demo_agent_astream_events())

    print("=" * 60)
    print("Fim do demonstrativo!")
    print("=" * 60)
