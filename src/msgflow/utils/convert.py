import re

def camel_snake_to_capitalize(nome):
    """ Convert a name to capitalize format """
    if "_" in nome:
        nome = nome.replace("_", " ")
    else:
        nome = re.sub(r"(?<!^)([A-Z])", r" \1", nome)
    return nome.title()