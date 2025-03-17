import re

def camel_snake_to_capitalize(nome):
    """ Convert a name to capitalize format """
    if "_" in nome:
        nome = nome.replace("_", " ")
    else:
        nome = re.sub(r"(?<!^)([A-Z])", r" \1", nome)
    return nome.title()

def convert_camel_to_snake_case(camel_str) -> str:
    snake_str = re.sub(r"(?<!^)([A-Z])", r"_\1", camel_str).lower()
    return snake_str    