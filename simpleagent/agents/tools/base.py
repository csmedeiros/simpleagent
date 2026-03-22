"""Módulo base para conversão de funções Python em tools compatíveis com a OpenAI API."""

from typing import Callable, Any, get_type_hints
import inspect


PYTHON_TYPE_TO_JSON_SCHEMA: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class Tool:
    """Wrapper que converte uma função Python em uma tool compatível com a OpenAI API.

    Faz introspecção da função (nome, docstring, type hints, defaults) para gerar
    automaticamente o JSON Schema esperado pela Responses API.

    Args:
        func: A função Python a ser convertida em tool.

    Attributes:
        func: A função original armazenada.
        name: Nome da função extraído de ``func.__name__``.
        description: Docstring da função, usada como descrição da tool.
        parameters: JSON Schema dos parâmetros no formato ``{"type": "object", "properties": ..., "required": ...}``.

    Example::

        def search_web(query: str, limit: int = 10):
            \"\"\"Busca na web.\"\"\"
            ...

        tool = Tool(search_web)
        tool.to_openai_tool()  # dict pronto para a API
        tool(query="python")   # chama search_web(query="python")
    """

    func: Callable
    """A função original armazenada."""

    name: str
    """Nome da função (extraído de ``func.__name__``)."""

    description: str
    """Docstring da função, usada como descrição da tool."""

    parameters: dict[str, Any]
    """JSON Schema dos parâmetros no formato esperado pela OpenAI."""

    def __init__(self, func: Callable) -> None:
        """Cria uma Tool a partir de uma função Python.

        Faz introspecção da assinatura da função para extrair nome, descrição
        e gerar o JSON Schema dos parâmetros automaticamente.

        Args:
            func: A função Python a ser convertida em tool. O nome, a docstring
                e os type hints são usados para montar o schema.
        """
        # Armazena a função original para poder chamá-la depois
        self.func = func

        # func.__name__ retorna o nome da função como string
        # Ex: def search_web(...) -> name = "search_web"
        self.name: str = func.__name__

        # func.__doc__ retorna a docstring da função (o texto entre """ """)
        # Ex: def search_web(...):
        #         """Busca na web"""  -> description = "Busca na web"
        # Se não tiver docstring, usa string vazia
        self.description: str = func.__doc__ or ""

        # inspect.signature() analisa a assinatura da função e retorna
        # um objeto Signature que contém todos os parâmetros.
        # Ex: def search_web(query: str, limit: int = 10)
        #     -> Signature(parameters={'query': <Parameter>, 'limit': <Parameter>})
        sig = inspect.signature(func)

        # get_type_hints() é mais robusto que param.annotation para pegar type hints.
        # Ele resolve forward references (strings como "List[str]") e lida com
        # Annotated types corretamente.
        # Ex: def search_web(query: str, limit: int = 10) -> {"query": str, "limit": int, "return": str}
        type_hints = get_type_hints(func)

        # Aqui vamos construir o JSON Schema dos parâmetros.
        # O formato final que o OpenAI espera é:
        # {
        #     "type": "object",
        #     "properties": {
        #         "query": {"type": "string", "description": "..."},
        #         "limit": {"type": "integer", "description": "..."}
        #     },
        #     "required": ["query"]
        # }
        properties: dict[str, Any] = {}
        required: list[str] = []

        # sig.parameters é um OrderedDict onde:
        #   - chave = nome do parâmetro (string)
        #   - valor = objeto Parameter com info sobre o parâmetro
        for param_name, param in sig.parameters.items():
            # param.annotation retorna o type hint do parâmetro.
            # Se não tiver type hint, retorna inspect.Parameter.empty.
            # Preferimos usar get_type_hints() que já resolve tudo,
            # mas fazemos fallback para str se não encontrar.
            python_type = type_hints.get(param_name, str)

            # Converte o tipo Python para o tipo JSON Schema usando nosso mapeamento.
            # Se o tipo não estiver no mapa (ex: uma classe custom), usa "string" como fallback.
            json_type = PYTHON_TYPE_TO_JSON_SCHEMA.get(python_type, "string")

            properties[param_name] = {"type": json_type}

            # param.default retorna o valor default do parâmetro.
            # Se NÃO tiver default, retorna inspect.Parameter.empty.
            # Ex: def search_web(query: str, limit: int = 10)
            #     - query.default = inspect.Parameter.empty  (obrigatório)
            #     - limit.default = 10                       (opcional)
            #
            # Parâmetros sem default são obrigatórios no JSON Schema,
            # então adicionamos na lista "required".
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        # Monta o schema completo dos parâmetros no formato JSON Schema
        self.parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def to_openai_tool(self) -> dict[str, Any]:
        """Converte a tool para o formato de function tool da Responses API do OpenAI.

        Returns:
            Dict com as chaves ``type``, ``name``, ``description`` e ``parameters``
            prontas para enviar à API::

                {
                    "type": "function",
                    "name": "search_web",
                    "description": "Busca na web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __call__(self, **kwargs: Any) -> Any:
        """Executa a função original encapsulada pela tool.

        Permite usar a instância de ``Tool`` como callable, delegando
        a chamada para ``self.func``.

        Args:
            **kwargs: Argumentos nomeados repassados diretamente à função original.

        Returns:
            O retorno da função original.

        Example::

            tool = Tool(search_web)
            tool(query="python")  # equivale a search_web(query="python")
        """
        return self.func(**kwargs)
