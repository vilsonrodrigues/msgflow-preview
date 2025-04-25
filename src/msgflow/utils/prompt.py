from typing import List, Tuple, Union


def format_examples(examples: List[Union[Tuple[str, str], Tuple[str, str, str]]]) -> str:
    """
    Formats a list of examples into XML-style string format.
    
    Each example in the list should be a tuple containing input and output strings,
    with an optional title as the third element. The function generates sequential IDs
    for each example starting from 1.
    
    Args:
        examples: A list of tuples where each tuple contains:
            - Input string (required)
            - Output string (required)
            - Title string (optional)
    
    Returns:
        A formatted XML-style string containing all examples.
    
    Example:
        >>> examples = [
        ...     ("What is your name?", "My name is GPT-3.5.", "Introduction"),
        ...     ("What day is today?", "Today is Tuesday."),
        ... ]
        >>> print(format_examples(examples))
        <example id=1 title="Introduction">
        <input>What is your name?</input>
        <output>My name is GPT-3.5.</output>
        </example>
        
        <example id=2>
        <input>What day is today?</input>
        <output>Today is Tuesday.</output>
        </example>
    """
    result = []
    
    for i, example in enumerate(examples, start=1):
        if len(example) == 3:
            input_text, output_text, title = example
            result.append(f'<example id={i} title="{title}">')
        else:
            input_text, output_text = example
            result.append(f"<example id={i}>")
        
        result.append(f"<input>{input_text}</input>")
        result.append(f"<output>{output_text}</output>")
        result.append("</example>\n")
    
    return "\n".join(result)

# Example usage
if __name__ == "__main__":
    examples = [
        ("Que dia é hoje?", "Hoje é terça", "Day"),
        ("Qual é o seu nome?", "Meu nome é Claude."),
        ("Quanto é 2+2?", "A soma de 2+2 é 4.", "Math")
    ]
    
    formatted_xml = format_examples(examples)
    print(formatted_xml)