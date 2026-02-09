import os
import frontmatter
from haystack.tools import tool
from constants import SKILLS_DIR


@tool
def read_skills_descriptions() -> list[dict]:
    """
    Read every skill and return the file path, the name and description.

    Returns:
    - A list of dictionaries with the name and description for each SKILL
    """
    try:
        # Reads the file
        skills = []
        for root, _, files in os.walk(SKILLS_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    post = frontmatter.load(f)
                    skills.append(
                        {"file_path": file_path, "nombre": post["name"], "description": post["description"]})

        return skills
    except Exception:
        return []
