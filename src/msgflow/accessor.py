from collections import OrderedDict
from msgflow.utils.validation import is_builtin_type


class Accessor:
    """ Auxiliar class to get and set python objects in a class using strings.

    !!! example
        ```python
        class ChildClass(Accessor):
            def __init__(self):
                super().__init__()
                    self.content = "Default content"
                    self.response = "Default response"
                    self.audios = []

        child = ChildClass()

        print(child.get("audios"))  # []

        child.set("audios", "https://audio1.mp3")
        print(child.get("audios"))  # ['https://audio1.mp3']
        print(child.get("audios.0"))  # https://audio1.mp3

        child.set("audios", "https://audio2.mp3")
        print(child.get("audios"))  # ['https://audio1.mp3', 'https://audio2.mp3']
        print(child.get("audios.1"))  # https://audio2.mp3

        child.set("audios.2", "https://audio3.mp3")
        print(child.get("audios"))  # ['https://audio1.mp3', 'https://audio2.mp3', 'https://audio3.mp3']
        print(child.get("audios.2"))  # https://audio3.mp3

        child.set("nested.list", [])
        child.set("nested.list", "item1")
        child.set("nested.list", "item2")
        print(child.get("nested.list"))  # ['item1', 'item2']
        print(child.get("nested.list.1"))  # item2

        child.set("list2", [])
        child.set("list2", 1) # [1]
        child.set("list2", 2) # [1, 2]
        print(child.get("list2.1")) # 2
        ```
    """
    def __init__(self):
        super().__setattr__("_attributes", {})

    def __setattr__(self, name, value):
        if name == "_attributes":
            super().__setattr__(name, value)
        else:
            self.set(name, value)

    def get(self, attr):
        parts = attr.split(".")
        value = self._attributes
        for part in parts:
            if isinstance(value, (dict, OrderedDict)):
                value = value.get(part)
            elif isinstance(value, list):
                try:
                    index = int(part)
                    value = value[index] if 0 <= index < len(value) else None
                except ValueError:
                    return None
            else:
                return None
            if value is None:
                return None
        return value

    def set(self, attr, value):
        parts = attr.split(".")

        # Check if path starts with specific prefixes and class has _route attribute
        if hasattr(self, "_route") and any(attr.startswith(prefix) for prefix in ["context.", "outputs.", "response."]):
            self._route.append(parts[-1])

        # Trace content in a same context        
        if hasattr(self, "trace") and is_builtin_type(value):
            self.trace(parts[-1], value)            

        target = self._attributes

        for i, part in enumerate(parts[:-1]):
            if part not in target:
                # If the next part is a number, create a list
                if parts[i + 1].isdigit():
                    target[part] = []
                else:
                    target[part] = (
                        OrderedDict() if isinstance(target, OrderedDict) else {}
                    )

            if isinstance(target[part], list):
                index = int(parts[i + 1])
                while len(target[part]) <= index:
                    target[part].append(None)
                if i == len(parts) - 2:  # If it is the last level
                    target[part][index] = value
                    return
                target = target[part]
            else:
                target = target[part]

        last_part = parts[-1]
        if isinstance(target, list):
            if last_part.isdigit():
                index = int(last_part)
                while len(target) <= index:
                    target.append(None)
                target[index] = value
            else:
                target.append(value)
        elif isinstance(target, (dict, OrderedDict)):
            if isinstance(target.get(last_part), list):
                target[last_part].append(value)
            else:
                target[last_part] = value
