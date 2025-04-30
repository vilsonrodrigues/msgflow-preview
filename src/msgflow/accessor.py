from collections import OrderedDict
from typing import Any

class Accessor:
    """
    Auxiliary class to get and set python objects within a class instance using dot-separated strings.

    The `set` method has been modified:
    - If the target attribute currently holds a list AND the new value being assigned is also a list,
      the `extend()` method is used to add the elements of the new list to the existing one.
    - If the target attribute holds a list BUT the new value is NOT a list, the new value is appended
      to the existing list using `append()`.
    - Otherwise (if the target attribute is not a list or doesn't exist), the attribute is set or
      overwritten with the new value.

    Handles creation of nested dictionaries (using OrderedDict for predictable key order) and lists
    as needed based on the attribute path.

    !!! example
        ```python
        class ChildClass(Accessor):
            def __init__(self):
                super().__init__()
                # Initialize attributes directly or via set
                self.content = "Default content"
                self.response = "Default response"
                self.audios = [] # Initialize as list using direct assignment (__setattr__)
                # Example for _route tracking (optional)
                self._route = [] # Initialize private attribute directly

        child = ChildClass()

        print(f"Initial audios: {child.get('audios')}")  # []

        child.set("audios", "[https://audio1.mp3](https://audio1.mp3)")
        print(f"After setting audio 1: {child.get('audios')}")  # ['[https://audio1.mp3](https://audio1.mp3)']
        print(f"Accessing audio 0: {child.get('audios.0')}")  # [https://audio1.mp3](https://audio1.mp3)

        # Appends because target is list, value is not
        child.set("audios", "[https://audio2.mp3](https://audio2.mp3)")
        print(f"After setting audio 2: {child.get('audios')}")  # ['[https://audio1.mp3](https://audio1.mp3)', '[https://audio2.mp3](https://audio2.mp3)']
        print(f"Accessing audio 1: {child.get('audios.1')}")  # [https://audio2.mp3](https://audio2.mp3)

        # Sets value at index 2
        child.set("audios.2", "[https://audio3.mp3](https://audio3.mp3)")
        print(f"After setting audio at index 2: {child.get('audios')}")  # ['[https://audio1.mp3](https://audio1.mp3)', '[https://audio2.mp3](https://audio2.mp3)', '[https://audio3.mp3](https://audio3.mp3)']
        print(f"Accessing audio 2: {child.get('audios.2')}")  # [https://audio3.mp3](https://audio3.mp3)

        # Overwrites value at index 0
        child.set("audios.0", "https://audio0_new.mp3")
        print(f"After overwriting audio at index 0: {child.get('audios')}") # ['https://audio0_new.mp3', '[https://audio2.mp3](https://audio2.mp3)', '[https://audio3.mp3](https://audio3.mp3)']


        child.set("nested.list", [])
        print(f"\nAfter setting nested.list to []: {child.get('nested.list')}") # []
        # Appends because target is list, value is not
        child.set("nested.list", "item1")
        print(f"After setting nested.list to 'item1': {child.get('nested.list')}") # ['item1']
        # Appends because target is list, value is not
        child.set("nested.list", "item2")
        print(f"After setting nested.list to 'item2': {child.get('nested.list')}") # ['item1', 'item2']
        print(f"Accessing nested.list.1: {child.get('nested.list.1')}")  # item2

        # --- List Extension Example ---
        child.set("chat_history", [])
        print(f"\nInitial chat_history: {child.get('chat_history')}") # []
        # Appends because target is list, value is not
        child.set("chat_history", 1)
        print(f"After setting chat_history to 1: {child.get('chat_history')}") # [1]
        # Extends because target is list, value is list
        child.set("chat_history", [2, 3])
        print(f"After setting chat_history to [2, 3]: {child.get('chat_history')}") # [1, 2, 3]
        # Appends because target is list, value is not
        child.set("chat_history", 4)
        print(f"After setting chat_history to 4: {child.get('chat_history')}") # [1, 2, 3, 4]
        # Extends because target is list, value is list
        child.set("chat_history", [5, 6])
        print(f"After setting chat_history to [5, 6]: {child.get('chat_history')}") # [1, 2, 3, 4, 5, 6]

        # Example with _route tracking
        child.set("context.user_query", "hello")
        print(f"\nRoute after context.user_query: {child.get('_route')}) # ['user_query']
        child.set("outputs.final_answer", "world")
        print(f"Route after outputs.final_answer: {child.get('_route')}) # ['user_query', 'final_answer']

        # Example of setting a non-list value, then extending
        child.set("new_list", 1)
        print(f"\nAfter setting new_list to 1: {child.get('new_list')}") # 1
        try:
            # This will fail because 'new_list' is not a list yet
            child.set("new_list", [2])
        except TypeError as e:
             print(f"Error as expected: {e}") # Should error or handle differently

        # Correct way: initialize as list first if extend/append is desired
        child.set("new_list_correct", [])
        child.set("new_list_correct", 1)
        child.set("new_list_correct", [2,3])
        print(f"Value of new_list_correct: {child.get('new_list_correct')}") # [1, 2, 3]
        ```
    """
    def __init__(self):
        super().__setattr__("_attributes", OrderedDict())

    def __setattr__(self, name, value):
        if name == "_attributes":
            super().__setattr__(name, value)
        else:
            self.set(name, value)

    def get(self, attr: str):
        """
        Retrieves an attribute value using a dot-separated string path.

        Args:
            attr: The dot-separated path string (e.g., "data.users.0.name").

        Returns:
            The value found at the specified path, or None if the path is invalid
            or any intermediate part does not exist.
        """
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

    def set(self, attr: str, value: Any) -> Any:
        parts = attr.split(".")

        # Check if path starts with specific prefixes and class has _route attribute
        if hasattr(self, "_route") and any(attr.startswith(prefix) for prefix in ["context.", "outputs.", "response."]):
            self._route.append(parts[-1])

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
                if isinstance(value, list):
                    target[last_part].extend(value)
                else:
                    target[last_part].append(value)
            else:
                target[last_part] = value
