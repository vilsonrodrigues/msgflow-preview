import ast
import re
from typing import (
    Any,
    Dict,
    GenericAlias,    
    List,
    Literal,
    Optional,
    Set,
    Union,    
    Tuple,
    Type,
    get_type_hints,
    get_origin,    
)
import msgspec
from msgflow.generation.reasoning.cot import ChainOfThoughts, COT_SYSTEM_MESSAGE
from msgflow.generation.reasoning.react import ReAct, REACT_SYSTEM_MESSAGE
from msgflow.generation.reasoning.self_consistency import SelfConsistency, SELF_CONSISTENCY_SYSTEM_MESSAGE
from msgflow.generation.reasoning.tot import TreeOfThoughts, TOT_SYSTEM_MESSAGE
from msgflow.utils.chat import apply_xml_tags


SIGNATURE_SYSTEM_MESSAGES = {
    ChainOfThoughts: COT_SYSTEM_MESSAGE,
    ReAct: REACT_SYSTEM_MESSAGE,
    SelfConsistency: SELF_CONSISTENCY_SYSTEM_MESSAGE,
    TreeOfThoughts: TOT_SYSTEM_MESSAGE
}


SIGNATURE_DEFAULT_SYSTEM_MESSAGE = """

"""

class InputField:
    """
    Represents an input field in a model signature.

    Attributes:
        desc: A description of the input field. Defaults to an empty string.
        ex: Input parameter example.
    """

    def __init__(self, desc: Optional[str] = "", ex: Optional[str] = None):
        self.desc = desc
        self.ex = ex


class OutputField:
    """
    Represents an output field in a model signature.

    Attributes:
        desc: A description of the output field. Defaults to an empty string.
        ex: Output parameter example.
    """

    def __init__(self, desc: Optional[str] = "", ex: Optional[str] = None):
        self.desc = desc
        self.ex = ex
        

class _SignatureMeta(type):
    """
    Metaclass to process input and output fields in a model signature.

    This metaclass collects all `InputField` and `OutputField` instances defined in a class
    and stores them in `_inputs` and `_outputs` dictionaries, respectively.
    """

    def __new__(cls, name, bases, dct):
        inputs = {}
        outputs = {}

        for key, value in dct.items():
            if isinstance(value, InputField):
                inputs[key] = value
            elif isinstance(value, OutputField):
                outputs[key] = value

        dct["_inputs"] = inputs
        dct["_outputs"] = outputs
        return super().__new__(cls, name, bases, dct)


class Signature(metaclass=_SignatureMeta):
    """
    Base class for model signatures.

    This class provides functionality to define and inspect input and output fields
    of a model. It uses the `_SignatureMeta` metaclass to automatically collect
    `InputField` and `OutputField` instances.

    Example:
        ```python
        class CheckCitationFaithfulness(Signature):
            \"\"\"Verify that the text is based on the provided context.\"\"\"

            context: str = InputField(desc="Facts here are assumed to be true")
            text: str = InputField()
            faithfulness: bool = OutputField()
            evidence: dict[str, list[str]] = OutputField(
                desc="Supporting evidence for claims"
            )

        # Get the class docstring
        print(CheckCitationFaithfulness.get_instructions())
        # Output: "Verify that the text is based on the provided context."

        # Get the signature in string format
        print(CheckCitationFaithfulness.get_str_signature())
        # Output: "context: str, text: str -> faithfulness: bool, evidence: dict[str, list[str]]"

        # Get input descriptions
        print(CheckCitationFaithfulness.get_input_descriptions())
        # Output: [('context', 'str', 'Facts here are assumed to be true'), ('text', 'str', '')]

        # Get output descriptions
        print(CheckCitationFaithfulness.get_output_descriptions())
        # Output: [('faithfulness', 'bool', ''), ('evidence', 'dict[str, list[str]]', 'Supporting evidence for claims')]
        ```
    """

    @classmethod
    def _type_to_str(cls, type_obj: Any) -> str:
        """
        Converts a type object to a readable string representation.

        Args:
            type_obj: The type object to convert.

        Returns:
            A string representation of the type.
        """
        if isinstance(type_obj, GenericAlias):  # For generic types
            return str(type_obj)
        return type_obj.__name__

    @classmethod
    def _get_inputs(cls) -> Dict[str, str]:
        """
        Retrieves the input fields with their names and types.

        Returns:
            A dictionary mapping input field names to their types.
        """
        type_hints = get_type_hints(cls)
        return {key: cls._type_to_str(type_hints[key]) for key in cls._inputs}

    @classmethod
    def _get_outputs(cls) -> Dict[str, str]:
        """
        Retrieves the output fields with their names and types.

        Returns:
            A dictionary mapping output field names to their types.
        """
        type_hints = get_type_hints(cls)
        return {key: cls._type_to_str(type_hints[key]) for key in cls._outputs}

    @classmethod
    def get_str_signature(cls) -> str:
        """
        Returns the signature of the parameters in string format.

        Returns:
            A string representation of the input and output fields.
        """
        inputs = [f"{key}: {typ}" for key, typ in cls._get_inputs().items()]
        outputs = [f"{key}: {typ}" for key, typ in cls._get_outputs().items()]
        return ", ".join(inputs) + " -> " + ", ".join(outputs)

    @classmethod
    def get_input_descriptions(cls) -> List[Tuple[str, str, str]]:
        """
        Returns the descriptions and types of the input parameters.

        Returns:
            A list of tuples containing the input field name,
            type, and description.
        """
        inputs = cls._get_inputs()
        return [(key, typ, cls._inputs[key].desc) for key, typ in inputs.items()]

    @classmethod
    def get_output_descriptions(cls) -> List[Tuple[str, str, str]]:
        """
        Returns the descriptions and types of the output parameters.

        Returns:
            A list of tuples containing the output field name,
            type, and description.
        """
        outputs = cls._get_outputs()
        return [(key, typ, cls._outputs[key].desc) for key, typ in outputs.items()]

    @classmethod
    def get_instructions(cls) -> Optional[str]:
        """
        Returns the class docstring.

        Returns:
            The docstring of the class, or `None` if no docstring is present.
        """
        return cls.__doc__.strip() if cls.__doc__ else None

    @classmethod
    def get_input_examples(cls) -> Dict[str, Optional[str]]:
        """
        Returns a mapping of input field names to their examples.

        Returns:
            A dictionary where keys are input field names and values are
            their corresponding examples ('ex'). If an example is not
            provided for a field, its value will be None.
        """
        return {key: field.ex for key, field in cls._inputs.items()}

    @classmethod
    def get_output_examples(cls) -> Dict[str, Optional[str]]:
        """
        Returns a mapping of output field names to their examples.

        Returns:
            A dictionary where keys are output field names and values are
            their corresponding examples ('ex'). If an example is not
            provided for a field, its value will be None.
        """
        return {key: field.ex for key, field in cls._outputs.items()}


ALLOWED_TYPES = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": List,
    "dict": Dict,
    "literal": Literal,
    "any": Any,
    "union": Union,
    "optional": Optional,
    "tuple": Tuple,
    "none": type(None),
    "set": Set
}

def _parse_literal_args(args_str: str) -> tuple:
    """Analisa com segurança os argumentos dentro de Literal[...] usando ast."""
    try:
        # Envolve a string em colchetes para analisá-la como uma lista literal
        parsed_node = ast.parse(f'[{args_str}]', mode='eval')
        # Extrai os valores do nó da lista literal analisada
        # ast.literal_eval avalia com segurança strings, números, tuplas, listas, dicts, bools, None
        values = [ast.literal_eval(node) for node in parsed_node.body.elts]
        return tuple(values)
    except (SyntaxError, ValueError, TypeError) as e:
        raise ValueError(f"Argumentos literais inválidos: '{args_str}'. Erro: {e}")

def _split_args(args_str: str) -> list[str]:
    """
    Divide uma string de argumentos (ex: "str, list[str], Literal['a', 'b']")
    respeitando colchetes aninhados e aspas simples/duplas básicas.
    """
    args = []
    level = 0  # Nível de aninhamento de colchetes/parênteses
    current_arg_start = 0
    in_quotes = None # Controla se está dentro de aspas simples ou duplas

    # Lida com o caso especial de argumentos vazios como em Tuple[()]
    if not args_str.strip():
        return []

    for i, char in enumerate(args_str):
        if char in ('[', '{', '(') and not in_quotes:
            level += 1
        elif char in (']', '}', ')') and not in_quotes:
            level -= 1
        elif char == "'" or char == '"':
            if in_quotes == char:
                in_quotes = None # Saiu das aspas
            elif in_quotes is None:
                in_quotes = char # Entrou nas aspas

        # Divide por vírgula apenas se não estiver aninhado e não dentro de aspas
        elif char == ',' and level == 0 and not in_quotes:
            args.append(args_str[current_arg_start:i].strip())
            current_arg_start = i + 1

    # Adiciona o último argumento (ou o único argumento)
    args.append(args_str[current_arg_start:].strip())
    return [arg for arg in args if arg] # Filtra strings vazias

def _parse_type_string(type_str: str) -> type:
    """
    Analisa recursivamente uma string de tipo (ex: "dict[str, list[str]]")
    para um objeto de tipo Python real, usando ALLOWED_TYPES (case-insensitive).
    Substitui o uso inseguro de eval().
    """
    type_str = type_str.strip()

    if not type_str:
        raise ValueError("A string de tipo não pode estar vazia.")

    type_str_lower = type_str.lower() # Converte para minúsculas para lookup

    # Caso base: Tipos simples (str, int, bool, any, none)
    # Verifica se NÃO é um tipo genérico que precisa de análise de argumentos
    simple_types_lower = {"str", "int", "float", "bool", "any", "none"}
    if type_str_lower in simple_types_lower and type_str_lower in ALLOWED_TYPES:
         return ALLOWED_TYPES[type_str_lower]

    # Caso recursivo: Tipos genéricos como list[T], dict[K, V], Union[X, Y], Literal[...], Optional[T], Tuple[...]
    # Usa regex para encontrar o padrão NomeTipo[argumentos]
    # O nome do tipo (grupo 1) pode ser case-insensitive agora
    match = re.match(r"^\s*(\w+)\s*\[(.*)\]\s*$", type_str, re.DOTALL) # re.DOTALL para abranger múltiplas linhas nos args
    if match:
        base_type_name, args_str = match.groups()
        base_type_name = base_type_name.strip()
        args_str = args_str.strip() # Argumentos internos

        base_type_name_lower = base_type_name.lower() # Converte para minúsculas para lookup

        if base_type_name_lower not in ALLOWED_TYPES:
            # Usa o nome original no erro para clareza
            raise ValueError(f"Tipo base não suportado: '{base_type_name}' em '{type_str}'")

        base_type = ALLOWED_TYPES[base_type_name_lower] # Busca usando a versão minúscula

        # --- Tratamento Específico para Tipos Genéricos ---
        # A lógica interna permanece a mesma, usando a variável `base_type` obtida

        if base_type is Literal:
            try:
                parsed_args = _parse_literal_args(args_str)
                if not parsed_args:
                     raise ValueError("Literal[...] não pode estar vazio")
                return Literal[parsed_args]
            except Exception as e:
                 raise ValueError(f"Falha ao analisar argumentos Literal '{args_str}': {e}")

        elif base_type is Optional:
            arg_strs_list = _split_args(args_str)
            if len(arg_strs_list) != 1:
                 raise ValueError(f"Optional[...] requer exatamente 1 argumento, obteve {len(arg_strs_list)} em '{type_str}'")
            inner_type = _parse_type_string(arg_strs_list[0])
            return Union[inner_type, type(None)]

        elif base_type in (List, Dict, Union, Tuple):
             arg_strs_list = _split_args(args_str)

             if base_type is Tuple:
                 if not arg_strs_list:
                     return Tuple[()]
                 if len(arg_strs_list) == 2 and arg_strs_list[1] == "...":
                     item_type = _parse_type_string(arg_strs_list[0])
                     return Tuple[item_type, ...]

             if not arg_strs_list and base_type is not Tuple:
                 # Usa o nome original no erro
                 raise ValueError(f"{base_type_name}[...] deve conter argumentos.")

             parsed_args = tuple(_parse_type_string(arg) for arg in arg_strs_list)

             if base_type is Dict and len(parsed_args) != 2:
                 raise ValueError(f"Dict requer exatamente 2 argumentos (chave, valor), obteve {len(parsed_args)} em '{type_str}'")
             if base_type is List and len(parsed_args) != 1:
                 raise ValueError(f"List requer exatamente 1 argumento, obteve {len(parsed_args)} em '{type_str}'")
             if base_type is Union:
                  if len(parsed_args) == 0:
                      raise ValueError(f"Union[...] não pode estar vazio em '{type_str}'")
                  if len(parsed_args) == 1:
                      return parsed_args[0] # Union[T] simplifica para T

             try:
                 return base_type[parsed_args]
             except TypeError as e:
                 # Usa o nome original no erro
                 raise ValueError(f"Argumentos de tipo inválidos para {base_type_name}: {parsed_args}. Erro: {e}")
        else:
             # Usa o nome original no erro
             raise ValueError(f"A construção do tipo genérico '{base_type_name}' não está implementada ou é inválida.")

    # Se não for um tipo simples e não corresponder ao padrão genérico NomeTipo[...]
    # Verifica novamente se é um tipo simples permitido (usando minúsculas)
    if type_str_lower in ALLOWED_TYPES:
        return ALLOWED_TYPES[type_str_lower]

    # Se chegou até aqui, a string de tipo não é reconhecida
    raise ValueError(f"String de tipo não suportada ou malformada: '{type_str}'")

def parse_annotations(signature: str) -> List[Tuple[str, str]]:
    """
    Analisa uma string de assinatura no formato "campo1: tipo1, campo2: tipo2, campo3".
    Assume `str` como tipo padrão se omitido.
    Lida com tipos aninhados e literais com vírgulas.
    """
    fields = []
    current_pos = 0
    level = 0  # Nível de aninhamento de colchetes/parênteses
    in_quotes = None # Controla aspas simples ou duplas
    current_field_start = 0
    signature = signature.strip()

    if not signature:
        return [] # Retorna lista vazia se a assinatura for vazia

    while current_pos < len(signature):
        char = signature[current_pos]

        if char in ('[', '{', '(') and not in_quotes:
            level += 1
        elif char in (']', '}', ')') and not in_quotes:
            level -= 1
            if level < 0: # Verifica aninhamento inválido
                 raise ValueError(f"Aninhamento de colchetes/parênteses desbalanceado perto de: '{signature[current_pos:]}'")
        elif char == "'" or char == '"':
             if in_quotes == char:
                 in_quotes = None # Saiu das aspas
             elif in_quotes is None:
                 in_quotes = char # Entrou nas aspas

        # Divide por vírgula apenas se não estiver aninhado e não dentro de aspas
        if char == ',' and level == 0 and not in_quotes:
            field_str = signature[current_field_start:current_pos].strip()
            if field_str: # Evita adicionar strings vazias se houver vírgulas extras
                fields.append(field_str)
            current_field_start = current_pos + 1

        current_pos += 1

    # Verifica se o aninhamento terminou corretamente
    if level != 0:
        raise ValueError("Aninhamento de colchetes/parênteses desbalanceado na assinatura.")
    if in_quotes:
        raise ValueError("Aspas não fechadas na assinatura.")

    # Adiciona o último campo
    last_field_str = signature[current_field_start:].strip()
    if last_field_str:
        fields.append(last_field_str)

    result = []
    for field_str in fields:
        # Divide pelo primeiro ':' encontrado
        parts = field_str.split(":", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            value_type = parts[1].strip()
            if not key:
                 raise ValueError(f"Nome do campo não pode ser vazio em '{field_str}'")
            if not value_type:
                 raise ValueError(f"Tipo não pode ser vazio após ':' em '{field_str}'")
        else:
            # Assume tipo 'str' se não houver ':'
            key = field_str.strip()
            value_type = "str" # Tipo padrão
            if not key:
                 raise ValueError(f"Nome do campo não pode ser vazio em '{field_str}'")

        result.append((key, value_type))

    return result

def create_struct_from_signature(signature: str, struct_name: Optional[str] = "DynamicStruct"):
    """
    Cria uma classe struct msgspec a partir de uma string de assinatura,
    usando um parser de tipos seguro e case-insensitive para nomes de tipos.

    Args:
        signature: String de assinatura (ex: "campo1: tipo1, campo2: tipo2").
        struct_name: Nome da classe struct a ser criada.

    Returns:
        Uma classe msgspec struct.

    Raises:
        ValueError: Se a assinatura ou os tipos forem inválidos/não suportados.
        RuntimeError: Para erros inesperados durante o parsing.
    """
    # Analisa a assinatura em pares (nome, string_tipo)
    annotations = parse_annotations(signature)

    struct_fields = []
    for name, type_str in annotations:
        try:
            # Usa o parser seguro (que agora é case-insensitive internamente)
            parsed_type = _parse_type_string(type_str)
            struct_fields.append((name, parsed_type))
        except ValueError as e:
            # Re-levanta o erro com mais contexto
            raise ValueError(f"Erro ao analisar o tipo para o campo '{name}' (tipo='{type_str}'): {e}")
        except Exception as e: # Captura outros erros inesperados
            raise RuntimeError(f"Erro inesperado ao analisar o tipo '{type_str}' para o campo '{name}': {e}")

    # Verifica se algum campo foi realmente analisado se a assinatura não estava vazia
    if not struct_fields and signature.strip():
        raise ValueError("Não foi possível analisar nenhum campo da assinatura fornecida.")

    # Cria a classe struct dinamicamente usando defstruct
    # Permite a criação de structs vazias se a assinatura for vazia
    try:
        DynamicStruct = msgspec.defstruct(struct_name, struct_fields)
    except Exception as e:
        raise RuntimeError(f"Erro ao criar msgspec.defstruct '{struct_name}': {e}")

    return DynamicStruct

def _parse_example_str(example_str: str, target_type: Type) -> Any:
    """
    Tenta analisar uma string de exemplo para o tipo Python de destino.
    Usa ast.literal_eval para segurança e para lidar com literais Python.
    """
    origin_type = get_origin(target_type)
    
    # Trata 'None' literal -> None Python
    if example_str.strip().lower() == 'none':
        # Verifica se o tipo de destino permite None (Optional)
        # Esta é uma verificação básica, pode precisar de mais robustez para tipos complexos
        is_optional = origin_type is Union and type(None) in get_args(target_type)
        if target_type is type(None) or is_optional:
             return None
        else:
             raise ValueError(f"String 'None' recebida, mas o tipo de destino '{target_type}' não é None ou Optional.")

    # Analisa baseado no tipo de destino
    if target_type is str:
        return example_str  # Retorna a string como está
    elif target_type is bool:
        low_ex = example_str.strip().lower()
        if low_ex in ('true', 'yes', '1'):
            return True
        elif low_ex in ('false', 'no', '0'):
            return False
        else:
            raise ValueError(f"Não foi possível analisar '{example_str}' como booleano.")
    elif target_type is int:
        return int(example_str)
    elif target_type is float:
        return float(example_str)
    # Tenta analisar tipos como list, dict, tuple, set e seus equivalentes em typing
    elif origin_type in (list, dict, tuple, set, typing.List, typing.Dict, typing.Tuple, typing.Set):
        try:
            parsed_value = ast.literal_eval(example_str)
            # Validação básica do tipo após a análise
            expected_type = list if origin_type in (list, typing.List) else \
                            dict if origin_type in (dict, typing.Dict) else \
                            tuple if origin_type in (tuple, typing.Tuple) else \
                            set if origin_type in (set, typing.Set) else None
            
            if expected_type and not isinstance(parsed_value, expected_type):
                 raise TypeError(f"Valor analisado '{parsed_value}' não é do tipo esperado {expected_type} para {target_type}")
                 
            return parsed_value
        except (ValueError, SyntaxError, TypeError, MemoryError) as e:
            raise ValueError(f"Falha ao analisar '{example_str}' como {target_type} usando ast.literal_eval: {e}") from e
    else:
        # Tenta ast.literal_eval como fallback genérico (para tipos simples como NoneType ou outros literais)
        # Use com cautela, pode não funcionar para tipos complexos/customizados.
        try:
            # Pode ser útil se o tipo for, por exemplo, Optional[int] e o exemplo for "123"
            # No entanto, os casos específicos (int, float, bool, etc.) já foram tratados.
            # Isso pode pegar casos como um exemplo "None" para um Optional[str].
            return ast.literal_eval(example_str)
        except (ValueError, SyntaxError, TypeError, MemoryError):
             # Se falhar, indica que não é um literal Python simples e não foi tratado acima.
             raise ValueError(f"Tipo não suportado '{target_type}' para análise automática da string de exemplo '{example_str}'")


def get_examples_from_signature(
    signature_cls: Type[Signature]
) -> Optional[Tuple[Dict[str, str], str]]:
    """
    Processa exemplos de uma classe Signature, retornando um dict para inputs
    e uma string JSON para outputs, somente se todos os exemplos estiverem presentes.

    Args:
        signature_cls: A classe Signature (que herda de Signature) a ser processada.

    Returns:
        Uma tupla contendo:
        - Dicionário mapeando nomes de inputs para suas strings de exemplo.
        - String JSON mapeando nomes de outputs para seus exemplos *analisados* (parsed).
        Retorna None se qualquer campo (input ou output) não tiver um exemplo definido.
        Levanta ValueError/TypeError se ocorrer um erro na análise dos exemplos de output
        ou na codificação JSON.
    """
    input_descs = signature_cls.get_input_descriptions()
    output_descs = signature_cls.get_output_descriptions()

    # Verifica se TODOS os campos têm um exemplo (não None)
    all_inputs_have_examples = all(desc[3] is not None for desc in input_descs)
    all_outputs_have_examples = all(desc[3] is not None for desc in output_descs)

    if not (all_inputs_have_examples and all_outputs_have_examples):
        return None  # Retorna None se faltar algum exemplo

    # 1. Cria o dicionário de exemplos de input (nome -> exemplo string)
    # Garantido que desc[3] (exemplo) não é None por causa da verificação acima
    input_examples_dict = {desc[0]: desc[3] for desc in input_descs} 

    # 2. Cria o dicionário de exemplos de output (nome -> exemplo *analisado*)
    output_parsed_dict = {}
    type_hints = get_type_hints(signature_cls) # Obtém os tipos reais

    try:
        for name, _, _, example_str in output_descs:
             # example_str é garantido como não None aqui
             target_type = type_hints.get(name)
             if target_type is None:
                 # Isso não deveria acontecer se a classe Signature estiver bem definida
                 raise ValueError(f"Type hint não encontrado para o campo de output '{name}' em {signature_cls.__name__}")
             
             # Usa a função auxiliar para analisar a string de exemplo para o tipo correto
             parsed_value = _parse_example_str(example_str, target_type) 
             output_parsed_dict[name] = parsed_value

    except (ValueError, TypeError) as e:
        # Captura erros da análise (parsing)
        raise ValueError(f"Erro ao analisar exemplos de output para {signature_cls.__name__}: {e}") from e

    # 3. Codifica o dicionário de outputs analisados para uma string JSON
    try:
        # ensure_ascii=False é bom para caracteres non-ASCII
        #output_json_string = json.dumps(output_parsed_dict, ensure_ascii=False, indent=2) # Adiciona indentação para legibilidade
        output_json_string = msgspec.json.encode(output_parsed_dict) # Adiciona indentação para legibilidade
    except TypeError as e:
        # Captura erros se algum tipo analisado não for serializável em JSON
        raise TypeError(f"Erro ao codificar outputs analisados para JSON em {signature_cls.__name__}: {e}") from e

    return input_examples_dict, output_json_string

def get_expected_output_from_signature(
    inputs_desc: List[Tuple[str, str, str, Union[str, None]]],
    outputs_desc: List[Tuple[str, str, str, Union[str, None]]]
) -> str:
    expected_output = "Your task inputs are:\n\n"
    for i, input_desc in enumerate(inputs_desc, 1):
        part = f"{i}. `{input_desc[0]}` ({input_desc[1]})"
        if input_desc[2]:
            part += f": {input_desc[2]}"
        expected_output += part + "\n"
    expected_output += "\nYour final answer should have:\n\n"
    for i, output_desc in enumerate(outputs_desc, 1):
        part = f"{i}. `{output_desc[0]}` ({output_desc[1]})"
        if output_desc[2]:
            part += f": {output_desc[2]}" 
        expected_output += f"{part}\n"
    expected_output += "\nBe consise in choosing your answers. Write an encoded JSON."
    return expected_output

def get_task_template_from_signature(inputs_desc: List[Tuple[str, str, str, Union[str, None]]]) -> str:
    task_template = ""
    for input_desc in inputs_desc:
        part = apply_xml_tags(input_desc[0], f"{{{{ {input_desc[0]} }}}}")
        task_template += part + "\n"
    task_template = task_template.strip()
    return task_template
