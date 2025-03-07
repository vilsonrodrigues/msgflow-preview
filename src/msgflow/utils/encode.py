import base64
import io
import os
import requests
import tempfile
from typing import Union


def encode_base64_from_url(url: str) -> str:
    with requests.get(url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")
    return result

def encode_local_file_in_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def to_io_object(input_data: Union[bytes, str]) -> io.IOBase:
    """
    Converts an input to a file IO object (such as io.BytesIO or a file opened in binary mode).

    Supports:
        - URLs (downloads the content and returns an io.BytesIO).
        - Base64 strings (decodes to an io.BytesIO).
        - Local paths to files (opens the file in binary mode).
        - Bytes (returns an io.BytesIO directly).

    Args:
        input_data: The input to convert to an IO object.

    Returns:
        The IO object containing the data.
    """
    if isinstance(input_data, bytes):
        return io.BytesIO(input_data)

    if isinstance(input_data, str):
        if input_data.startswith("http://") or input_data.startswith("https://"):
            response = requests.get(input_data)
            response.raise_for_status()
            return io.BytesIO(response.content)

        try:
            decoded_data = base64.b64decode(input_data)
            return io.BytesIO(decoded_data)
        except (base64.binascii.Error, ValueError):
            pass

        if os.path.exists(input_data) and os.path.isfile(input_data):
            return open(input_data, "rb")

    raise ValueError(
        f"Invalid input: must be a URL, Base64, file path, or bytes. Given: {type(input_data)}"
    )

def to_bytes(input_data: Union[bytes, str]) -> io.BufferedReader:
    """
    Converts an input to a BufferedReader, ensuring consistency in the returned type.

    Supports:
    - URLs (downloads the content and converts to BufferedReader)
    - Base64 strings (decodes to BufferedReader)
    - Local paths to files (opens the file in binary mode)
    - Bytes (converts to BufferedReader)

    Args:
        input_data: The input to convert to BufferedReader.

    Returns:
        The BufferedReader object containing the data.
    """

    # Aux function to create temporary file and return as BufferedReader
    def _create_temp_file(data):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(data)
        temp_file.close()
        return open(temp_file.name, "rb")

    if isinstance(input_data, bytes):
        return _create_temp_file(input_data)

    if isinstance(input_data, str):
        if input_data.startswith(("http://", "https://")):
            response = requests.get(input_data)
            response.raise_for_status()
            return _create_temp_file(response.content)

        try:
            decoded_data = base64.b64decode(input_data)
            return _create_temp_file(decoded_data)
        except (base64.binascii.Error, ValueError):
            pass

        if os.path.exists(input_data) and os.path.isfile(input_data):
            return open(input_data, "rb")

    raise ValueError(
        f"Invalid input: must be a URL, Base64, file path, or bytes. Given: {type(input_data)}"
    )
