import re
import shlex
import os
import random

class Agent:
    def __init__(self, references=None, secrets=None):
        self.references = references
        self.secrets = secrets

    def _get_secrets(self):
        secrets = {}
        for secret_key, secret_env_key in self.secrets.items():
            secrets[secret_key] = os.getenv(secret_env_key, str(random.random()))
        return secrets

    def _apply_references(self, model_response, dynamic_refs=None):
        ref = {}
        if self.references:
            ref.update(self.references)
        if self.secrets:
            ref.update(self._get_secrets())
        if dynamic_refs:
            ref.update(dynamic_refs)

        if ref:
            refs = {"ref": ref}

            if isinstance(model_response, str): # Text generation
                model_response = self._replace_references_in_text(model_response, refs)
            elif isinstance(model_response, list): # Tool call
                for tool_call in model_response:
                    params = tool_call[2]
                    if isinstance(params, dict):
                        for key, value in params.items():
                            if isinstance(value, str) and value.startswith("{{ref.") and value.endswith("}}"):
                                # Replaces full references directly
                                params[key] = self._get_reference_value(value, refs)
                            elif isinstance(value, str):
                                # Process text with partial references
                                params[key] = self._replace_references_in_text(value, refs)
        return model_response

    def _replace_references_in_text(self, text, refs):
        # Pattern to capture {{ref.name}} or {{ref.funcao(parameters)}}
        pattern = r"\{\{ref\.(\w+)(?:\((.*?)\))?\}\}"
        matches = re.findall(pattern, text)

        for ref_key, params_str in matches:
            value = self._get_reference_value(f"{{{{ref.{ref_key}{f'({params_str})' if params_str else ''}}}}}", refs)
            full_ref = f"{{{{ref.{ref_key}{f'({params_str})' if params_str else ''}}}}}"
            text = text.replace(full_ref, str(value))
        return text

    def _get_reference_value(self, ref_str, refs):
        # Extract the key and parameters from the reference
        pattern = r"\{\{ref\.(\w+)(?:\((.*?)\))?\}\}"
        match = re.match(pattern, ref_str)
        if match:
            ref_key, params_str = match.groups()
            if ref_key in refs:
                value = refs[ref_key]
                if callable(value):
                    if params_str:
                        params = self.parse_params(params_str)
                        if isinstance(params, str):
                            return params # Parsing error
                        try:
                            return value(**params)
                        except Exception as e:
                            # logger.error(str(e))
                            return "" # If has error, return a empty string
                    else:
                        return value()
                return value
            else:
                return "" # If unknow ref, return a empty string
        return ref_str

    def parse_params(self, params_str):
        try:
            args = shlex.split(params_str)
            params = {}
            for arg in args:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    params[key] = value
                else:
                    params[arg] = True
            return params
        except Exception as e:
            # logger.error(str(e))
            return "" # Parser error, return a empty string

    def __call__(self, model_response, dynamic_refs=None):
        return self._apply_references(model_response, dynamic_refs)


# Função de exemplo para temperatura
def get_temperature(city):
    temperatures = {"new_york": 25, "london": 18}
    return temperatures.get(city, "Cidade desconhecida")

# Configuração e testes
references = {
    "arnold_phrase": "Bodybuilding é como qualquer outro esporte.",
    "temperature": get_temperature,
    "temperature_new_york": lambda: get_temperature("new_york")
}

agent = Agent(references=references)

# Teste 1: Texto simples
model_response = "Arnold disse: {{ref.arnold_phrase}}"
print(agent(model_response))
# Saída esperada: "Arnold disse: Bodybuilding é como qualquer outro esporte."

# Teste 2: Função com parâmetros
model_response = "A temperatura em Nova York é {{ref.temperature(city=\"new_york\")}}°C."
print(agent(model_response))
# Saída esperada: "A temperatura em Nova York é 25°C."

# Teste 3: Função sem parâmetros
model_response = "A temperatura em NY é {{ref.temperature_new_york}}°C."
print(agent(model_response))
# Saída esperada: "A temperatura em NY é 25°C."

# Teste 4: Chamada de ferramenta
tool_callings = [('123', 'tool_name', {'temp': '{{ref.temperature(city="london")}}'})]
result = agent(tool_callings)
print(result)
# Saída esperada: [('123', 'tool_name', {'temp': 18})]        