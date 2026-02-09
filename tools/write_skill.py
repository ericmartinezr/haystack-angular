import os
from haystack.tools import tool
from constants import SKILLS_DIR


@tool
def write_skill(dir_name: str, file_content: str) -> str:
    """
    Writes a SKILL.md file using frontmatter Markdown style

    Arguments:
    - dir_name (str): The directory where the SKILL.md will be placed
    - file_content (str): The content of the SKILL.md file

    Returns:
    - The file path if succesfully written, otherwise empty.
    """
    try:
        # Creates the directoy
        dir_path = os.path.join(SKILLS_DIR, dir_name)
        os.makedirs(dir_path, exist_ok=False)

        # Writes the file
        file_path = os.path.join(dir_path, "SKILL.md")
        with open(file_path, "w") as f:
            content_written = f.write(file_content)

        return file_path if content_written > 0 else ""
    except Exception as e:
        return ""
