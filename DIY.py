"""
DIY.py — Como um agente de IA funciona por baixo dos panos.

Demonstração interativa passo a passo do loop agentivo (ReAct).
Cada etapa pausa e espera [Enter] para continuar, permitindo
que você entenda exatamente o que acontece em cada fase.

Para rodar:
    python DIY.py
"""

import os
import json
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

load_dotenv()

from simpleagent.chat_models.openai import OpenAIChatModel
from simpleagent.messages import SystemMessage, HumanMessage, AIMessage, ToolCall, ToolMessage
from simpleagent.agents.tools import Tool

console = Console()


def pause():
    """Pausa e espera o usuário pressionar Enter para continuar."""
    console.print("\n[dim]Pressione [bold]Enter[/bold] para continuar...[/dim]")
    input()


def show(markdown_text: str):
    """Renderiza markdown no terminal via rich."""
    console.print(Markdown(markdown_text))


def show_json(data: dict, title: str = ""):
    """Renderiza JSON formatado com syntax highlighting."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
    if title:
        console.print(Panel(syntax, title=title, border_style="blue"))
    else:
        console.print(syntax)


# ============================================================================
# TOOLS — Funções que o agente pode usar
# ============================================================================

def calcular(operacao: str, a: float, b: float) -> float:
    """Faz um cálculo matemático entre dois números. Operações: soma, subtracao, multiplicacao, divisao."""
    ops = {"soma": a + b, "subtracao": a - b, "multiplicacao": a * b, "divisao": a / b}
    return ops.get(operacao, "Operação inválida")


def buscar_informacao(topico: str) -> str:
    """Busca informações sobre um tópico. Use para perguntas que precisam de conhecimento factual."""
    dados = {
        "python": "Python é uma linguagem de programação criada por Guido van Rossum em 1991. É conhecida pela sintaxe simples e pela comunidade ativa.",
        "react": "ReAct (Reasoning and Acting) é um padrão onde agentes de IA alternam entre raciocínio e ação usando LLMs e ferramentas.",
        "openai": "OpenAI é uma empresa de IA fundada em 2015. Criou o GPT-4, ChatGPT e a Responses API.",
    }
    for chave, valor in dados.items():
        if chave in topico.lower():
            return valor
    return f"Nenhuma informação encontrada sobre '{topico}'."


# ============================================================================
# DEMONSTRAÇÃO PASSO A PASSO
# ============================================================================

def main():
    console.clear()
    show("# Como um agente de IA funciona por baixo dos panos")
    show("""
Este demonstrativo vai executar o **loop agentivo (ReAct)** passo a passo,
mostrando exatamente o que acontece em cada etapa — desde a montagem das
mensagens até a execução de tools e a resposta final.

Vamos construir tudo na mão, sem usar a classe `Agent`, para que você veja
cada peça funcionando.
""")
    pause()

    # ========================================================================
    # PASSO 1: Criar o modelo
    # ========================================================================
    show("---")
    show("## Passo 1 — Criar o ChatModel")
    show("""
O `ChatModel` é a interface com o LLM. Ele sabe como:
- Converter mensagens Python para o formato da API (`to_dict()`)
- Enviar para a Responses API (`invoke()` / `stream()`)
- Parsear o retorno em `AIMessage` e `ToolCall`
""")

    llm = OpenAIChatModel(
        model="gpt-4o",
        base_url=os.getenv("MODEL_BASE_URL"),
        api_key=os.getenv("MODEL_API_KEY"),
    )

    console.print(Panel(
        f"modelo: [bold]{llm.model}[/bold]\nbase_url: [dim]{llm.client.base_url}[/dim]",
        title="ChatModel criado",
        border_style="green",
    ))
    pause()

    # ========================================================================
    # PASSO 2: Converter funções em Tools
    # ========================================================================
    show("---")
    show("## Passo 2 — Converter funções Python em Tools")
    show("""
A classe `Tool` faz **introspecção** de uma função Python e gera o JSON Schema
que a API espera. Ela extrai:
- **Nome** → `func.__name__`
- **Descrição** → `func.__doc__`
- **Parâmetros** → `inspect.signature()` + `get_type_hints()` → JSON Schema

Vamos converter duas funções: `calcular` e `buscar_informacao`.
""")

    tools = [Tool(calcular), Tool(buscar_informacao)]
    tools_map = {tool.name: tool for tool in tools}
    openai_tools = [tool.to_openai_tool() for tool in tools]

    for tool in tools:
        show_json(tool.to_openai_tool(), title=f"Tool: {tool.name}")
        console.print()

    show("Isso é o que vai ser enviado no parâmetro `tools=` de toda chamada ao LLM.")
    pause()

    # ========================================================================
    # PASSO 3: Montar o histórico inicial
    # ========================================================================
    show("---")
    show("## Passo 3 — Montar o histórico inicial")
    show("""
O histórico é uma **lista de mensagens** que o LLM recebe como contexto.
Começamos com:
1. `SystemMessage` — instruções de comportamento
2. `HumanMessage` — a pergunta do usuário
""")

    system_prompt = "Você é um assistente que usa ferramentas quando necessário. Seja conciso."
    user_input = "Quanto é 42 * 17? E o que é Python?"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]

    show("**Histórico atual:**")
    for msg in messages:
        role = msg.role.upper()
        console.print(f"  [bold cyan]{role}[/bold cyan]: {msg.content}")

    show(f"\n> O LLM vai receber essas {len(messages)} mensagens + {len(tools)} tools disponíveis.")
    pause()

    # ========================================================================
    # PASSO 4+: Loop agentivo (ReAct)
    # ========================================================================
    show("---")
    show("## Passo 4 — O Loop Agentivo (ReAct)")
    show("""
Agora entra o **coração do agente**. O loop funciona assim:

```
enquanto não tiver resposta final:
    1. Enviar histórico + tools para o LLM
    2. O LLM retorna AIMessage e/ou ToolCall(s)
    3. Se retornou ToolCall(s):
       → Executar cada tool
       → Adicionar ToolCall + ToolMessage ao histórico
       → Voltar ao passo 1
    4. Se retornou só AIMessage:
       → É a resposta final. Fim do loop.
```

Vamos executar isso agora, passo a passo.
""")
    pause()

    max_iterations = 5
    for iteration in range(1, max_iterations + 1):
        show(f"---")
        show(f"### Iteração {iteration} — Chamando o LLM")
        show(f"Enviando **{len(messages)} mensagens** + **{len(tools)} tools** para o LLM...")
        console.print()

        # Chama o LLM
        result = llm.invoke(messages, tools=openai_tools)

        # Separa AIMessage e ToolCalls
        ai_messages = [m for m in result if isinstance(m, AIMessage)]
        tool_calls = [m for m in result if isinstance(m, ToolCall)]

        # Mostra o que o LLM retornou
        if ai_messages and ai_messages[0].content:
            show(f"**LLM respondeu com texto:**")
            console.print(Panel(
                Markdown(ai_messages[0].content),
                border_style="green",
                title="AIMessage",
            ))

        if tool_calls:
            show(f"**LLM pediu {len(tool_calls)} tool call(s):**")
            for tc in tool_calls:
                show_json(
                    {"name": tc.name, "arguments": json.loads(tc.arguments), "call_id": tc.call_id},
                    title=f"ToolCall: {tc.name}",
                )
        pause()

        # Se tem tool calls, executar
        if tool_calls:
            show(f"### Iteração {iteration} — Executando Tools")
            show("""
O LLM não executa as tools — ele só **decide** quais chamar e com quais argumentos.
Quem executa somos nós. O fluxo é:
1. Parsear os `arguments` de JSON string → dict Python
2. Buscar a função no `tools_map` pelo `name`
3. Chamar a função com os argumentos
4. Criar um `ToolMessage` com o resultado
5. Adicionar `ToolCall` + `ToolMessage` ao histórico
""")

            # Adiciona AIMessage ao histórico (se teve)
            for m in ai_messages:
                messages.append(m)

            for tc in tool_calls:
                # Adiciona o ToolCall ao histórico
                messages.append(tc)

                # Parseia e executa
                args = json.loads(tc.arguments)
                tool = tools_map[tc.name]
                tool_result = tool(**args)

                # Converte resultado para string
                result_str = tool_result if isinstance(tool_result, str) else json.dumps(tool_result, ensure_ascii=False)

                # Cria ToolMessage
                tool_msg = ToolMessage(call_id=tc.call_id, output=result_str)
                messages.append(tool_msg)

                console.print(Panel(
                    f"[bold]{tc.name}[/bold]({json.dumps(args, ensure_ascii=False)})\n\n"
                    f"[green]Resultado:[/green] {result_str}",
                    title=f"Tool executada",
                    border_style="yellow",
                ))

            show(f"\nHistórico agora tem **{len(messages)} mensagens**. Voltando ao LLM...")
            pause()

        else:
            # Sem tool calls — resposta final
            messages.extend(result)

            show("### O LLM respondeu diretamente — Fim do loop!")
            show(f"""
Não houve `ToolCall` nesta iteração, então o LLM deu a **resposta final**.
O loop agentivo terminou após **{iteration} iteração(ões)**.
""")
            pause()
            break
    else:
        show(f"> Limite de {max_iterations} iterações atingido.")

    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    show("---")
    show("## Resumo — Histórico completo do agente")
    show("""
Este é o histórico completo de mensagens que foi construído durante o loop.
Cada mensagem tem um tipo e um formato específico para a Responses API:
""")

    for i, msg in enumerate(messages):
        match msg:
            case SystemMessage():
                console.print(f"  [dim]{i}.[/dim] [bold blue]SYSTEM[/bold blue]     {msg.content[:80]}...")
            case HumanMessage():
                console.print(f"  [dim]{i}.[/dim] [bold cyan]USER[/bold cyan]       {msg.content}")
            case AIMessage():
                text = msg.content or "[sem texto — apenas tool calls]"
                console.print(f"  [dim]{i}.[/dim] [bold green]AI[/bold green]         {text[:100]}{'...' if len(text) > 100 else ''}")
            case ToolCall():
                console.print(f"  [dim]{i}.[/dim] [bold yellow]TOOL CALL[/bold yellow]  {msg.name}({msg.arguments})")
            case ToolMessage():
                console.print(f"  [dim]{i}.[/dim] [bold magenta]TOOL RESULT[/bold magenta] {msg.output[:80]}{'...' if len(msg.output) > 80 else ''}")

    console.print()
    show("""
---

**O que você acabou de ver é tudo que um framework como LangChain faz por baixo dos panos.**

A classe `Agent` encapsula exatamente esse loop em dois métodos:
- `agent.run()` — executa tudo de uma vez e retorna o histórico
- `agent.astream_events()` — faz streaming assíncrono emitindo eventos a cada etapa

O código completo está em `simpleagent/agents/base.py`.
""")


if __name__ == "__main__":
    main()
