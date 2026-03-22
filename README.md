# simpleagent — Agente de IA do zero

Implementação didática de um agente de IA com tool calling, construído do zero em Python usando apenas a OpenAI Responses API e Pydantic.

O objetivo é entender **como frameworks como LangChain funcionam por dentro**, sem depender deles.

## Arquitetura

```
simpleagent/
├── messages/          # Tipos de mensagem (SystemMessage, HumanMessage, AIMessage, ToolCall, ToolMessage)
│   ├── messages.py    # Classes Pydantic que serializam para o formato da Responses API
│   └── __init__.py
├── chat_models/       # Interface com LLMs
│   ├── base.py        # ChatModel (ABC) — invoke, stream, astream
│   ├── openai.py      # OpenAIChatModel — implementação para OpenAI/Azure
│   └── __init__.py
├── agents/            # Agente autônomo
│   ├── base.py        # Agent — ReAct loop (run, astream_events)
│   ├── tools/
│   │   ├── base.py    # Tool — converte funções Python em tools via introspecção
│   │   └── __init__.py
│   └── __init__.py
DIY.py                 # Script demonstrativo de todas as funcionalidades
```

## Como funciona

### 1. Chat Models — Interação com LLMs

`ChatModel` é a interface para chamar modelos de linguagem. Suporta 3 modos:

```python
from simpleagent.chat_models.openai import OpenAIChatModel
from simpleagent.messages import SystemMessage, HumanMessage

llm = OpenAIChatModel(model="gpt-4o", api_key="...")

messages = [
    SystemMessage(content="Você é um assistente conciso."),
    HumanMessage(content="O que é Python?"),
]

# Síncrono — retorna resposta completa
result = llm.invoke(messages)
print(result[0].content)

# Streaming síncrono — chunks em tempo real
for chunk in llm.stream(messages):
    if chunk["type"] == "text":
        print(chunk["content"], end="", flush=True)

# Streaming assíncrono
async for chunk in llm.astream(messages):
    if chunk["type"] == "text":
        print(chunk["content"], end="", flush=True)
```

### 2. Tools — Funções Python como ferramentas

`Tool` faz introspecção de uma função Python (nome, docstring, type hints) e gera automaticamente o JSON Schema que a API espera:

```python
from simpleagent.agents.tools import Tool

def calcular(operacao: str, a: float, b: float) -> float:
    """Faz um cálculo matemático entre dois números."""
    ops = {"soma": a + b, "subtracao": a - b}
    return ops.get(operacao)

tool = Tool(calcular)
tool.to_openai_tool()  # dict pronto para a API
tool(operacao="soma", a=10, b=5)  # executa a função: 15.0
```

### 3. Agent — ReAct Loop

`Agent` combina ChatModel + Tools em um loop autônomo:

1. Recebe mensagem do usuário
2. Chama o LLM com histórico + tools
3. Se o LLM pedir tools → executa e volta ao passo 2
4. Se o LLM responder → retorna

```python
from simpleagent.agents import Agent

agent = Agent(
    model=llm,
    tools=[calcular],  # aceita funções Python direto
    system_prompt="Você é um assistente prestativo.",
)

# Síncrono
messages = agent.run("Quanto é 42 * 17?")
print(messages[-1].content)

# Streaming assíncrono com eventos
async for event in agent.astream_events("Quanto é 42 * 17?"):
    match event["event"]:
        case "on_llm_text_delta":
            print(event["content"], end="", flush=True)
        case "on_tool_call_start":
            print(f"\nChamando: {event['tool_call'].name}")
        case "on_tool_call_end":
            print(f"Resultado: {event['tool_message'].output}")
```

#### Eventos do `astream_events`

| Evento | Descrição |
|---|---|
| `on_llm_start` | Início de uma chamada ao LLM |
| `on_llm_text_delta` | Chunk de texto da resposta |
| `on_llm_reasoning_delta` | Chunk de raciocínio (modelos o-series) |
| `on_tool_call_start` | LLM decidiu chamar uma tool |
| `on_tool_call_end` | Resultado da execução da tool |
| `on_llm_end` | Resposta final do LLM |
| `on_agent_end` | Agente terminou (histórico completo) |

## Setup

```bash
# 1. Clone e entre no diretório
git clone <repo-url>
cd agent_from_scratch

# 2. Crie o ambiente virtual e instale dependências
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env com sua API key e URL

# 4. Rode o demonstrativo
python DIY.py
```

## Dependências

- **openai** — Client para a Responses API
- **pydantic** — Validação e serialização das mensagens
- **python-dotenv** — Carregamento de variáveis de ambiente

## Conceitos implementados

- **Responses API** — formato moderno da OpenAI (diferente da Chat Completions API)
- **Tool calling** — o LLM decide quando e quais ferramentas usar
- **ReAct loop** — ciclo de raciocínio e ação até resolver a tarefa
- **Streaming** — resposta em tempo real via Server-Sent Events
- **Async generators** — streaming assíncrono sem bloquear o event loop
- **Introspecção Python** — `inspect.signature`, `get_type_hints`, `__doc__` para gerar schemas
