import os
from haystack.tools import tool
from constants import EXAMPLE_SKILLS_DIR


@tool
def read_example_skills() -> str:
    """
    Reads the example skills for adding to LLM's context

    Returns:
    - Returns the content of the file
    """
    skill_content = ""
    for root, _, files in os.walk(EXAMPLE_SKILLS_DIR):
        for file in files:
            skill_path = os.path.join(root, file)
            with open(skill_path, "r") as f:
                skill_content = f.read()
    return skill_content
