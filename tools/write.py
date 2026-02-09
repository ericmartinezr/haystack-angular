import os
from uuid import uuid4
from haystack.tools import tool
from constants import RESULT_DIR


@tool
def write_todo(file_content: str) -> str:
    """
    Writes a TODO.md file using Markdown style

    Arguments:
    - dir_name (str): The directory where the TODO.md will be placed
    - file_content (str): The content of the TODO.md file

    Returns:
    - The file path if succesfully written, otherwise empty.
    """
    try:
        # Creates the directoy
        dir_name = str(uuid4())
        dir_path = os.path.join(RESULT_DIR, dir_name)
        os.makedirs(dir_path, exist_ok=False)

        # Writes the file
        file_path = os.path.join(dir_path, "TODO.md")
        with open(file_path, "w") as f:
            content_written = f.write(file_content)

        return file_path if content_written > 0 else ""
    except Exception as e:
        return ""
