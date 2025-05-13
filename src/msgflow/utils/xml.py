import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, Optional


_type_converters = {
    "int": int,
    "float": float,
    "str": str,
    "bool": lambda x: x.lower() == "true",
    "dict": lambda x: x,
    "list": lambda x: x
}

def apply_xml_tags(id: str, content: str, output_id: Optional[str] = None) -> str:
    if output_id is None:
        output_id = id
    return f"<{id}>\n{content}\n</{output_id}>"

def _xml_to_typed_value(element: ET.Element) -> Any:
    """Convert an XML element to a Python value based on type."""    
    dtype_attr = element.attrib.get("dtype", "str") # Assumes "str" ​​if "dtype" is not specified

    if dtype_attr == "dict":        
        result = {} # Create a dictionary with your children
        for child in element:
            result[child.tag] = _xml_to_typed_value(child)
        return result
    elif dtype_attr == "list":
        return [_xml_to_typed_value(child) for child in element] # Create a list with the children
    elif dtype_attr in _type_converters:        
        converter = _type_converters[dtype_attr] # Converts text to the specified type
        return converter(element.text)
    else:
        raise ValueError(f"Unknown dtype: {dtype_attr}")

def xml_to_typed_dict(xml_string: str) -> Dict[str, Any]:
    """Converts an XML into a typed dictionary, returning direct values ​​for single tags.
    
    Args:
        xml_string: Text content xml-based

    Returns:
        A dict with typed entities extracted

    ::: example
        xml_string = '''
        <person dtype="dict">
            <name dtype="str">Prevost</name>
            <age dtype="int">69</age>
            <hobbies dtype="list">
                <hobby>Evangelize</hobby>
                <hobby>Defend the sick</hobby>
            </hobbies>
        </person>
        <message>God loves everyone, and evil will not prevail.</message>
        <good_pope dtype="bool">true</good_pope>
        '''
        print(xml_to_typed_dict(xml_string))
    """
    # Add root in xml string
    xml_string = apply_xml_tags("root", xml_string)
    root = ET.fromstring(xml_string)
    tag_count = defaultdict(int)
    temp_result = defaultdict(list)
    
    for child in root: # Counts how many times each tag appears and collects the values
        tag_count[child.tag] += 1
        value = _xml_to_typed_value(child)
        temp_result[child.tag].append(value)

    # Adjust the result: direct value if the tag appears once, list if it appears multiple times
    result = {}
    for tag, values in temp_result.items():
        if tag_count[tag] == 1:
            result[tag] = values[0]  # Return single value
        else:
            result[tag] = values     # Return list
    return result
