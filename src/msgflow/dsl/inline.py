import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
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
            Supports logic operators:
                * AND (&): condition1 & condition2
                * OR (||): condition1 || condition2
                * NOT (!): !(condition)
            None verification
                * is None: user.name is None
                * is not None: user.name is not None
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
            "comparison": r"([a-zA-Z0-9_.]+)\s*(==|!=|<=|>=|<|>|is|is not)\s*(.*)"
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
                modules = [m.strip() for m in parallel_match.group(1).split(",")]
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
                raise ValueError(f"Invalid DSL syntax or unknown module: {part}")

        return steps

    def _parse_branch(self, branch: str) -> List[str]:
        """Parse a conditional branch into a list of modules."""
        if not branch:
            return []
        # Ensures that even a single entry without a comma is treated as a list
        return [m.strip() for m in branch.split(",") if m.strip()]

    def _tokenize_condition(self, condition: str) -> List[str]:
        """Tokenize condition string into logical components."""
        condition = condition.strip() # Remove black spaces

        # Pattern for tokenization that captures logical operators, parentheses and expressions
        token_pattern = r"(\|\||&|!|\(|\)|[^&|!()]+)"
        tokens = re.findall(token_pattern, condition)

        # Removes empty tokens and trim
        return [token.strip() for token in tokens if token.strip()]

    def _parse_logical_expression(
        self, tokens: List[str], index: Optional[int] = 0
    ) -> Tuple:
        """Parse logical expression recursively."""
        result, index = self._parse_or_expression(tokens, index)
        return result, index

    def _parse_or_expression(self, tokens: List[str], index: int) -> Tuple:
        """Parse OR expressions (lowest precedence)."""
        left, index = self._parse_and_expression(tokens, index)

        while index < len(tokens) and tokens[index] == "||":
            index += 1  # skip "||"
            right, index = self._parse_and_expression(tokens, index)
            left = ("OR", left, right)

        return left, index

    def _parse_and_expression(self, tokens: List[str], index: int) -> Tuple:
        """Parse AND expressions (medium precedence)."""
        left, index = self._parse_not_expression(tokens, index)

        while index < len(tokens) and tokens[index] == "&":
            index += 1  # skip "&"
            right, index = self._parse_not_expression(tokens, index)
            left = ("AND", left, right)

        return left, index

    def _parse_not_expression(self, tokens: List[str], index: int) -> Tuple:
        """Parse NOT expressions (highest precedence)."""
        if index < len(tokens) and tokens[index] == "!":
            index += 1  # skip "!"
            expr, index = self._parse_primary_expression(tokens, index)
            return ("NOT", expr), index
        else:
            return self._parse_primary_expression(tokens, index)

    def _parse_primary_expression(self, tokens: List[str], index: int) -> Tuple:
        """Parse primary expressions (comparisons or parenthesized expressions)."""
        if index < len(tokens) and tokens[index] == "(":
            index += 1  # skip "("
            expr, index = self._parse_logical_expression(tokens, index)
            if index < len(tokens) and tokens[index] == ")":
                index += 1  # skip ")"
                return expr, index
            else:
                raise ValueError("Missing closing parenthesis")
        else:
            # This should be a comparison expression
            if index < len(tokens):
                comparison_expr = tokens[index]
                index += 1
                return ("COMPARISON", comparison_expr), index
            else:
                raise ValueError("Expected comparison expression")

    def _evaluate_comparison(self, comparison_str: str, message: dotdict) -> bool:
        """Evaluate a single comparison expression."""
        match = re.match(self.patterns["comparison"], comparison_str.strip())
        if not match:
            raise ValueError(f"Invalid condition format: {comparison_str}. "
                             "Expected key_path operator value.")

        key_path, operator, expected_value_str = match.groups()

        actual_value = message.get(key_path, None)
        expected_value_str = expected_value_str.strip()

        # Handle None/null checks
        if operator in ["is", "is not"]:
            if expected_value_str.lower() in ["none", "null"]:
                if operator == "is":
                    return actual_value is None
                else:  # is not
                    return actual_value is not None
            else:
                raise ValueError(f"`is` and `is not` operators only "
                                 "support `None` or `null` comparisons")

        # Remove quotes from the expected value and attempt
        # to convert to the appropriate type
        expected_value_str = expected_value_str.strip("'\"")

        try:
            # Handle boolean values
            if expected_value_str.lower() == "true":
                expected_value = True
            elif expected_value_str.lower() == "false":
                expected_value = False
            # Attempts to convert to int or float for numeric comparisons
            elif "." in expected_value_str:
                expected_value = float(expected_value_str)
            else:
                try:
                    expected_value = int(expected_value_str)
                except ValueError:
                    expected_value = expected_value_str

            # Try to convert the actual value also to the same type for comparison
            if actual_value is None:
                # If the value does not exist in the message, we cannot compare numerically
                return False

            # Handle boolean conversion for actual value
            if isinstance(expected_value, bool):
                if isinstance(actual_value, str):
                    actual_value = actual_value.lower() == "true"
                else:
                    actual_value = bool(actual_value)
            elif isinstance(expected_value, (int, float)):
                actual_value = type(expected_value)(actual_value)

        except (ValueError, TypeError):
            # If conversion fails, treat as string
            expected_value = expected_value_str
            # Ensures the current value is a string for consistent comparison if not None
            actual_value = str(actual_value) if actual_value is not None else None

        # Performs comparison based on the operator
        if actual_value is None and operator not in ["==", "!="]:
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

    def _evaluate_logical_tree(self, tree: tuple, message: dotdict) -> bool:
        """Evaluate a logical expression tree."""
        if tree[0] == "COMPARISON":
            return self._evaluate_comparison(tree[1], message)
        elif tree[0] == "AND":
            left_result = self._evaluate_logical_tree(tree[1], message)
            right_result = self._evaluate_logical_tree(tree[2], message)
            return left_result and right_result
        elif tree[0] == "OR":
            left_result = self._evaluate_logical_tree(tree[1], message)
            right_result = self._evaluate_logical_tree(tree[2], message)
            return left_result or right_result
        elif tree[0] == "NOT":
            expr_result = self._evaluate_logical_tree(tree[1], message)
            return not expr_result
        else:
            raise ValueError(f"Unknown logical operator: {tree[0]}")

    def _evaluate_condition(self, condition: str, message: dotdict) -> bool:
        """Evaluates a condition with logical operators support."""
        # Tokenize the condition
        tokens = self._tokenize_condition(condition)

        if not tokens:
            raise ValueError("Empty condition")

        # Parse the logical expression
        tree, final_index = self._parse_logical_expression(tokens)

        if final_index != len(tokens):
            raise ValueError(f"Unexpected tokens after parsing: {tokens[final_index:]}")

        # Evaluate the tree
        return self._evaluate_logical_tree(tree, message)

    def __call__(self, notation: str, modules: Mapping[str, Callable], message: dotdict) -> dotdict:
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
                        raise ValueError(f"Module {mod_name} not found for parallel execution.")
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
    notation: str, modules: Mapping[str, Callable], message: dotdict
) -> dotdict:
    """
    Executes a workflow defined in DSL notation over a given `message`.

    Args:
        notation:
            A string describing the execution pipeline using a Domain-Specific Language (DSL).
            
            The DSL supports:
            
            **Sequential execution**:
                Use `->` to define a linear pipeline.
                Example: `"prep -> transform -> output"`
            
            **Parallel execution**:
                Use square brackets `[...]` to group modules that run in parallel.
                Example: `"prep -> [feat_a, feat_b] -> combine"`
            
            **Conditional execution**:
                Use curly braces with a ternary-like structure: `{condition ? then_module, else_module}`.
                Example: `"{user.age > 18 ? adult_module, child_module}"`
            
            **Logical operations in conditions**:
                - **AND**: `cond1 & cond2`
                - **OR**: `cond1 || cond2`
                - **NOT**: `!cond`
                Example: `"{user.is_active & !user.is_banned ? allow, deny}"`
            
            **None checking in conditions**:
                - `is None`: Example: `user.name is None`
                - `is not None`: Example: `user.name is not None`

            These conditionals are evaluated against the `message` object context.

        modules:
            A dictionary mapping module names (as strings) to callables.
            Each function must accept and return a `message` object.

        message:
            The input message (dotdict) to be passed through the pipeline.
    
    Returns:
        The resulting `message` after executing the defined workflow.

    Raises:
        TypeError:
            If message is not a `msgflow.dotdict` instance.
        TypeError:
            If modules is not a `Mapping` instance.            
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
    if not isinstance(modules, Mapping):
        raise TypeError("`modules` must be an instance of `Mapping`")     
    dsl = InlineDSL()
    message = dsl(notation, modules, message)
    return message
