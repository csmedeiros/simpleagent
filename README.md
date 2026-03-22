# simpleagent — Agente de IA do zero, sem frameworks

Implementação didática de um agente de IA com tool calling, construído do zero em Python.

O objetivo não é criar mais um framework — é **abrir a caixa preta** e entender o que acontece por baixo quando você usa LangChain, CrewAI, ou qualquer outra abstração. Cada arquivo tem comentários detalhados explicando linha a linha o que está acontecendo e por quê.

Dependências mínimas: `openai` (client HTTP), `pydantic` (serialização) e `python-dotenv`.

---

## Guia de leitura

O código foi escrito para ser lido em sequência. Cada módulo constrói sobre o anterior, e os comentários explicam não só o *quê*, mas o *porquê* de cada decisão.

### Passo 1: Mensagens (`simpleagent/messages/messages.py`)

**Comece aqui.** Antes de entender agentes, você precisa entender como a comunicação com a API funciona.

Este arquivo define as 5 classes que representam tudo que trafega entre seu código e o LLM:

- `SystemMessage` — instruções para o modelo ("Você é um assistente...")
- `HumanMessage` — mensagem do usuário
- `AIMessage` — resposta do modelo
- `ToolCall` — quando o modelo decide chamar uma ferramenta
- `ToolMessage` — resultado da execução de uma ferramenta

**O que prestar atenção:**
- Como cada classe sabe se serializar para o formato JSON da Responses API via `to_dict()`
- Por que `ToolCall` e `ToolMessage` **não** herdam de `Message` (a Responses API usa formatos diferentes)
- A diferença entre o formato da Responses API e a Chat Completions API clássica

### Passo 2: Tools (`simpleagent/agents/tools/base.py`)

**Aqui você entende como funções Python viram ferramentas para o LLM.**

A classe `Tool` recebe uma função Python e extrai automaticamente tudo que a API precisa:
- Nome → `func.__name__`
- Descrição → `func.__doc__`
- Schema dos parâmetros → `inspect.signature()` + `get_type_hints()` → JSON Schema

**O que prestar atenção:**
- O mapeamento `PYTHON_TYPE_TO_JSON_SCHEMA` — como tipos Python viram tipos JSON Schema
- Como `inspect.signature()` e `get_type_hints()` fazem a introspecção
- Como parâmetros sem default viram `required` no schema
- O método `to_openai_tool()` que monta o formato final para a API

### Passo 3: Chat Models (`simpleagent/chat_models/base.py`)

**Aqui você entende a interface com o LLM.** É a ponte entre mensagens Python e a API.

A classe `ChatModel` oferece 3 modos de interação:
- `invoke()` — síncrono, retorna tudo de uma vez
- `stream()` — síncrono com streaming, yield chunks em tempo real
- `astream()` — assíncrono com streaming

**O que prestar atenção:**
- `_parse_response()` — como o response da API é parseado em `AIMessage` e `ToolCall`
- Os 3 tipos de output da Responses API: `reasoning`, `message`, `function_call`
- No `stream()`, como os Server-Sent Events são consumidos e re-emitidos como dicts tipados
- No `astream()`, a diferença é apenas `async for` + `AsyncClient` — a lógica é idêntica

Depois leia `simpleagent/chat_models/openai.py` — é só 5 linhas que configuram URL e API key.

### Passo 4: Agent (`simpleagent/agents/base.py`)

**Este é o arquivo principal.** Tudo converge aqui.

A classe `Agent` implementa o padrão ReAct (Reasoning and Acting):

```
Usuário pergunta
    → LLM pensa (com histórico + tools disponíveis)
        → LLM responde direto? → Fim
        → LLM quer chamar tool? → Executa → Resultado volta pro histórico → LLM pensa de novo
```

**O que prestar atenção:**
- `__init__()` — como funções Python são auto-encapsuladas em `Tool`, e como o `tools_map` é montado
- `_loop()` — o coração do agente. Leia devagar. É um for simples, mas é onde a "mágica" acontece
- `_execute_tool()` — como o JSON de argumentos é parseado e a função é executada
- `astream_events()` — a mesma lógica do `_loop()`, mas emitindo eventos em tempo real

### Passo 5: DIY.py

**Demonstrativo que exercita tudo.** Rode para ver cada componente em ação:
- Chat direto com o LLM (invoke + stream)
- Criação e inspeção de Tools
- Agente completo com ReAct loop (run + astream_events)

---

## Estrutura

```
simpleagent/
├── messages/
│   └── messages.py       # 1. Tipos de mensagem (a base de tudo)
├── agents/
│   └── tools/
│       └── base.py       # 2. Conversão função Python → tool para a API
├── chat_models/
│   ├── base.py           # 3. Interface com LLMs (invoke, stream, astream)
│   └── openai.py         #    Implementação para OpenAI/Azure
├── agents/
│   └── base.py           # 4. Agent — ReAct loop (run, astream_events)
DIY.py                    # 5. Demonstrativo completo
```

## Setup

```bash
# 1. Clone e entre no diretório
git clone https://github.com/csmedeiros/simpleagent
cd simpleagent

# 2. Crie o ambiente virtual e instale dependências
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env com sua API key e URL base

# 4. Leia e rode o demonstrativo para visualizar o funcionamento.
python DIY.py
```

## Conceitos que você vai aprender lendo este código

Siga esta ordem — cada conceito depende do anterior:

| # | Arquivo | O que você vai aprender |
|---|---|---|
| 1 | `messages/messages.py` | Como mensagens são estruturadas na Responses API. A diferença entre os formatos baseados em `role` (System, Human, AI) e os baseados em `type` (ToolCall, ToolMessage). Serialização com Pydantic. |
| 2 | `agents/tools/base.py` | Introspecção Python: como extrair nome, docstring e type hints de uma função com `inspect.signature()` e `get_type_hints()`. Conversão de tipos Python → JSON Schema. |
| 3 | `chat_models/base.py` | Como chamar um LLM via API: parsing da resposta (`_parse_response`), streaming com Server-Sent Events (`stream`), e a versão assíncrona com `AsyncClient` + async generators (`astream`). |
| 4 | `chat_models/openai.py` | Como criar um provedor concreto — são 5 linhas que definem URL e variável de ambiente. |
| 5 | `agents/base.py` | O coração do projeto. Como o ReAct loop funciona: o LLM decide entre responder ou chamar tools, as tools são executadas, e o resultado volta pro histórico. Inclui a versão com streaming de eventos (`astream_events`). |
| 6 | `DIY.py` | Demonstrativo prático que exercita tudo junto: chat direto, inspeção de tools, agente síncrono e agente com streaming assíncrono. |
