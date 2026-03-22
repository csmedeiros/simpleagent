# simpleagent — Agente de IA do zero, sem frameworks

Implementação didática de um agente de IA com tool calling, construído do zero em Python.

O objetivo é **abrir a caixa preta** e entender o que acontece por baixo quando você usa LangChain, CrewAI ou qualquer outra abstração. Cada arquivo tem comentários detalhados explicando linha a linha o que está acontecendo e por quê.

Dependências mínimas: `openai`, `pydantic` e `python-dotenv`.

## Guia visual interativo

Antes de ler o código, veja o guia visual que explica passo a passo como um agente funciona:

**[https://csmedeiros.github.io/simpleagent/](https://csmedeiros.github.io/simpleagent/)**

Ou rode localmente:
```bash
python app.py
# Abre http://localhost:8000 automaticamente
```

---

## Guia de leitura do código

O código foi escrito para ser lido em sequência. Cada módulo constrói sobre o anterior.

### Passo 1: Mensagens (`simpleagent/messages/messages.py`)

**Comece aqui.** As 5 classes que representam tudo que trafega entre seu código e o LLM:
`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolCall`, `ToolMessage`.

**Preste atenção em:**
- Serialização via `to_dict()` para o formato da Responses API
- Por que `ToolCall` e `ToolMessage` **não** herdam de `Message`

### Passo 2: Tools (`simpleagent/agents/tools/base.py`)

Como funções Python viram ferramentas para o LLM via introspecção:
`func.__name__` → nome, `func.__doc__` → descrição, `inspect.signature()` + `get_type_hints()` → JSON Schema.

**Preste atenção em:**
- Mapeamento `PYTHON_TYPE_TO_JSON_SCHEMA`
- Como parâmetros sem default viram `required`
- `to_openai_tool()` que monta o formato final

### Passo 3: Chat Models (`simpleagent/chat_models/base.py`)

A ponte entre mensagens Python e a API. Três modos: `invoke()`, `stream()`, `astream()`.

**Preste atenção em:**
- `_parse_response()` — como o response vira `AIMessage` e `ToolCall`
- No `stream()`, como Server-Sent Events são consumidos
- `pending_tool_items` — como `call_id` é pareado entre eventos de streaming

Depois leia `openai.py` — são 5 linhas que configuram URL e API key.

### Passo 4: Agent (`simpleagent/agents/base.py`)

**O arquivo principal.** O padrão ReAct:

```
Usuário pergunta
    → LLM pensa (histórico + tools)
        → Responde direto? → Fim
        → Quer chamar tool? → Executa → Resultado volta → LLM pensa de novo
```

**Preste atenção em:**
- `_loop()` — o coração do agente, um `for` simples onde a mágica acontece
- `_execute_tool()` — parsing de JSON e execução da função
- `astream_events()` — mesma lógica, mas emitindo eventos em tempo real

---

## Estrutura

```
simpleagent/
├── messages/
│   └── messages.py       # 1. Tipos de mensagem
├── agents/
│   ├── tools/
│   │   └── base.py       # 2. Função Python → tool para a API
│   └── base.py           # 4. Agent — ReAct loop
├── chat_models/
│   ├── base.py           # 3. Interface com LLMs
│   └── openai.py         #    Implementação OpenAI/Azure
docs/
└── index.html            # Guia visual interativo (GitHub Pages)
DIY.py                    # Demonstrativo CLI com rich
app.py                    # Servidor web para o guia visual
```

## Setup

```bash
git clone https://github.com/csmedeiros/simpleagent
cd simpleagent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edite o .env com sua API key e URL base

# Guia visual no navegador
python app.py

# Ou demonstrativo no terminal
python DIY.py
```

## Conceitos que você vai aprender

| # | Arquivo | Conceito |
|---|---|---|
| 1 | `messages/messages.py` | Formato da Responses API, serialização com Pydantic |
| 2 | `agents/tools/base.py` | Introspecção Python (`inspect`, `get_type_hints`), JSON Schema |
| 3 | `chat_models/base.py` | Chamada à API, streaming via SSE, async generators |
| 4 | `chat_models/openai.py` | Provedor concreto (5 linhas) |
| 5 | `agents/base.py` | ReAct loop, tool calling, streaming de eventos |
