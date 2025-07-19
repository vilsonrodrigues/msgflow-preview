import re
from typing import Any, Callable, Dict, List
from msgflow.dotdict import dotdict
from msgflow.nn import functional as F


class InlineDSL:
    """
    Parses and executes a mini domain-specific language (DSL) for 
    defining workflow pipelines over modules.

    Supported Node Types:

    1. Module Node:
        Syntax: 
            `"module_name"`
        Description:
            A single module that processes the message.
        Example:
            `"a"`

    2. Parallel Node:
        Syntax:
            `"[module1, module2, ...]"`
        Description:
            Executes multiple modules in parallel using broadcast and gather. 
            The same input message is passed to all modules, and their results are merged.
        Example:
            `"[feat_a, feat_b]"`

    3. Conditional Node:
        Syntax:
            `"{condition?true_branch[,false_branch]}"`
        Description:
            Conditionally executes a branch of modules depending on the evaluation
            of the condition against the `message`.
            The condition must follow the format `key_path operator value`, 
            e.g., `output.agent == "xpto"`.
            The `true_branch` and `false_branch` are comma-separated module names.
        Example:
            `"{output.agent == 'xpto'?a,b}"`
            Executes `a` if the condition is true, `b` otherwise.

    4. Arrow Separator (`->`):
        Description:
            Defines the sequence of operations in the pipeline.
        Example:
            `"prep -> [feat_a, feat_b] -> combine"`
    """
    def __init__(self):
        self.patterns = {
            "arrow": r"\s*->\s*",
            "parallel": r"\[(.*?)\]",
            "conditional": r"\{(.*?)\?(.*?)(?:,(.*?))?\}",
            "identifier": r"[a-zA-Z_][a-zA-Z0-9_]*",
            "comparison": r"([a-zA-Z0-9_.]+)\s*(==|!=|<=|>=|<|>)\s*(.*)"
        }

    def parse(self, notation: str) -> List[Dict[str, Any]]:
        """Parse DSL notation into a list of steps."""
        steps = []
        parts = re.split(self.patterns["arrow"], notation.strip())
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            conditional_match = re.match(self.patterns["conditional"], part)
            if conditional_match:
                condition, true_branch, false_branch = conditional_match.groups()
                steps.append({
                    "type": "conditional",
                    "condition": condition.strip(),
                    "true_branch": self._parse_branch(true_branch),
                    "false_branch": self._parse_branch(false_branch) if false_branch else []
                })
                continue

            parallel_match = re.match(self.patterns["parallel"], part)
            if parallel_match:
                modules = [m.strip() for m in parallel_match.group(1).split(',')]
                steps.append({
                    "type": "parallel",
                    "modules": modules
                })
                continue
                
            if re.match(self.patterns["identifier"], part):
                steps.append({
                    "type": "module",
                    "module": part
                })
            else:
                raise ValueError(f"Invalid DSL syntax or unknown module: `{part}`")
                
        return steps

    def _parse_branch(self, branch: str) -> List[str]:
        """Parse a conditional branch into a list of modules."""        
        if not branch:
            return []
        # Ensures that even a single entry without a comma is treated as a list        
        return [m.strip() for m in branch.split(",") if m.strip()]

    def _evaluate_condition(self, condition: str, message: dotdict) -> bool:
        """Evaluates a condition using the essage's .get() method."""
        match = re.match(self.patterns["comparison"], condition)
        if not match:
            raise ValueError(f"Invalid condition format: `{condition}`. "
                             "Expected `key_path operator value`.")
            
        key_path, operator, expected_value_str = match.groups()
        
        actual_value = message.get(key_path, None)

        # Remove quotes from the expected value and attempt
        # to convert to the appropriate type
        expected_value_str = expected_value_str.strip().strip("'\"")

        try:
            # Attempts to convert to int or float for numeric comparisons
            if '.' in expected_value_str:
                expected_value = float(expected_value_str)
            else:
                expected_value = int(expected_value_str)
            
            # Try to convert the actual value also to the same type for comparison
            # If actual_value is None, it cannot be converted, resulting in False 
            # for most comparisons
            if actual_value is None:
                # If the value does not exist in the message, we cannot compare numerically
                return False
            actual_value = type(expected_value)(actual_value)

        except (ValueError, TypeError):
            # If conversion fails, treat as string
            expected_value = expected_value_str
            # Ensures the current value is a string for consistent comparison if not None
            actual_value = str(actual_value) if actual_value is not None else None

        # Performs comparison based on the operator
        # Add handling for `None` in comparisons other than `==` or `!=`
        if actual_value is None and operator not in ["==", "!="]:
            # If the value does not exist and is not an 
            # equality/inequality comparison with None
            return False
        
        if operator == "==":
            return actual_value == expected_value
        elif operator == "!=":
            return actual_value != expected_value
        elif operator == "<":
            return actual_value < expected_value
        elif operator == ">":
            return actual_value > expected_value
        elif operator == "<=":
            return actual_value <= expected_value
        elif operator == ">=":
            return actual_value >= expected_value
        else:
            raise ValueError(f"Unknown comparison operator: {operator}")

    def __call__(self, notation: str, modules: Dict[str, Callable], message: dotdict) -> dotdict:
        """Execute the DSL pipeline."""
        steps = self.parse(notation)
        current_message = message

        for step in steps:
            if step["type"] == "module":
                module = modules.get(step["module"])
                if not module:
                    raise ValueError(f"Module `{step['module']}` not found.")
                # Passes the current message to the module and updates the message with the result
                current_message = module(current_message)
                
            elif step["type"] == "parallel":
                # Execute modules in parallel using msg_bcast_gather
                parallel_modules = []
                for mod_name in step["modules"]:
                    module = modules.get(mod_name)
                    if not module:
                        raise ValueError(f"Module `{mod_name}` not found for parallel execution.")
                    parallel_modules.append(module)
                    
                if not parallel_modules:
                    raise ValueError(f"No valid modules found for parallel execution in {step['modules']}.")

                current_message = F.msg_bcast_gather(parallel_modules, current_message)

            elif step["type"] == "conditional":
                # Evaluates condition and executes appropriate branch
                condition_result = self._evaluate_condition(step["condition"], current_message)
                branch = step["true_branch"] if condition_result else step["false_branch"]
                
                for module_name in branch:
                    module = modules.get(module_name)
                    if not module:
                        raise ValueError(f"Module `{module_name}` not found in conditional branch.")
                    current_message = module(current_message) # Pass the updated message
                    
        return current_message

def inline(
    notation: str, modules: Dict[str, Callable], message: dotdict
) -> dotdict:
    """
    Executes a workflow defined in DSL notation over a given `message`.

    Args:
        notation:
            A string describing the execution pipeline using DSL syntax.
            Supports sequential (`->`), parallel (`[...]`), and conditional 
            (`{...?...}`) logic.
        modules:
            A dictionary mapping module names (as strings) to callable.
            Each function must accept and return a `message` object.
        message:
            The input message to be passed through the pipeline.

    Returns:
        The transformed message after passing through the pipeline.

    Raises:
        TypeError:
            If message is not a `msgflow.dotdict` instance.    
        ValueError:
            If a module is not found, if the DSL syntax is invalid, 
            or if a condition cannot be parsed.

    Examples:
        from msgflow import dotdict, inline

        def prep(msg: Message) -> Message:
            print(f"Executing prep, current msg: {msg}")
            msg['output'] = {'agent': 'xpto', 'score': 10, 'status': 'success'}
            return msg

        def feat_a(msg: Message) -> Message:
            print(f"Executing feat_a, current msg: {msg}")
            msg['feat_a'] = 'result_a'
            return msg

        def feat_b(msg: Message) -> Message:
            print(f"Executing feat_b, current msg: {msg}")
            msg['feat_b'] = 'result_b'
            return msg

        def final(msg: Message) -> Message:
            print(f"Executing final, current msg: {msg}")
            msg['final'] = 'done'
            return msg            

        my_modules = {
            "prep": prep,
            "feat_a": feat_a,
            "feat_b": feat_b,
            "final": final
        }
        input_msg = dotdict()
        result = inline(
            "prep -> [feat1, feat2] -> {output.agent == 'x'?model_a,model_b} -> final",
            modules=my_modules,
            message=input_msg
        )
    """
    if not isinstance(message, dotdict):
        raise TypeError("`message` must be an instance of `msgflow.dotdict`")    
    dsl = InlineDSL()
    message = dsl(notation, modules, message)
    return message
