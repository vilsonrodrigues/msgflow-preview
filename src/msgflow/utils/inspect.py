import ast
import inspect
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse


def get_mime_type(source: str) -> str:
    """Tries to guess the MIME type, with fallback"""
    mime_type, _ = mimetypes.guess_type(source)
    if mime_type:
        return mime_type
    # Extension-based fallbacks (simplistic)
    ext = Path(source).suffix.lower()
    if ext == ".jpeg" or ext == ".jpg": return "image/jpeg"
    if ext == ".png": return "image/png"
    if ext == ".gif": return "image/gif"
    if ext == ".webp": return "image/webp"
    if ext == ".mp3": return "audio/mpeg"
    if ext == ".wav": return "audio/wav"
    if ext == ".ogg": return "audio/ogg"
    if ext == ".flac": return "audio/flac"
    if ext == ".opus": return "audio/opus"
    if ext == ".m4a": return "audio/mp4"
    if ext == ".webm": return "audio/webm"
    if ext == ".pdf": return "application/pdf"
    # Generic fallback
    return "application/octet-stream"

def get_fn_name():
    return inspect.currentframe().f_back.f_code.co_name

def get_filename(data_path: str) -> str:
    if data_path.startswith(("http://", "https://", "ftp://")):
        parsed_url = urlparse(data_path)
        filename = os.path.basename(parsed_url.path)    
    else: # Local file
        filename = os.path.basename(data_path)    
    return filename

def get_decorators(node):
    """ Extracts the decorators from an AST node and converts them to string."""
    decorators = []
    for decorator in node.decorator_list:
        dec = ast.unparse(decorator)
        decorators.append(dec)
    return decorators

def extract_info_from_py_file(file_path):
    """
    Extracts functions, classes and their details (parameters, docstrings, 
    decorators and whether it is async) from a Python file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=file_path)
    
    functions = []
    classes = []
    
    # Process only top-level definitions
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_info = {
                "name": node.name,
                "async": isinstance(node, ast.AsyncFunctionDef),
                "args": [arg.arg for arg in node.args.args],
                "doc": ast.get_docstring(node),
                "decorators": get_decorators(node) if node.decorator_list else []
            }
            functions.append(func_info)
        elif isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "doc": ast.get_docstring(node),
                "decorators": get_decorators(node) if node.decorator_list else [],
                "methods": []
            }
            # Process the class methods (supports async methods too)
            for body_item in node.body:
                if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_info = {
                        "name": body_item.name,
                        "async": isinstance(body_item, ast.AsyncFunctionDef),
                        "args": [arg.arg for arg in body_item.args.args],
                        "doc": ast.get_docstring(body_item),
                        "decorators": get_decorators(body_item) if body_item.decorator_list else []
                    }
                    class_info["methods"].append(method_info)
            classes.append(class_info)
    
    return {"functions": functions, "classes": classes}

def scan_py_repository(root_path):
    """
    Recursively traverses the Python files in a repository,
    extracting the structure of each file.
    """
    repo_info = {}
    for dirpath, _, filenames in os.walk(root_path):
        for file in filenames:
            if file.endswith(".py"):
                file_path = os.path.join(dirpath, file)
                try:
                    info = extract_info_from_py_file(file_path)
                    repo_info[file_path] = info
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return repo_info
