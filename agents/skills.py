from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.tools import ComponentTool, PipelineTool
from haystack.dataclasses import ChatMessage
from haystack.core.super_component import SuperComponent
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses.byte_stream import ByteStream
from tools.read_example_skills import read_example_skills
from tools.read_skills import read_skills_descriptions
from tools.write_skill import write_skill
from models.ollama import thinking_generator

agent = Agent(
    chat_generator=thinking_generator,
    system_prompt="""You're a helpful AI agent expert on Skills designed by Anthropic. 
    Your job is to analyze the user query and identify the SKILLs within it, 
    then create a SKILL.md file for each one.
    
    Brief definition:
    A SKILL is a set of instructions to accomplish a specific task, it has a frontmatter with metadata and a body with the instructions.

    How to identify SKILLs:
    1. if the user requests "Generate a simple angular componente", there is one skill : "angular".
    2. if the user requests "Add a authentication system to my angular app", there are two skills: "angular" and "authentication".

    
    # Workflow
    1. Read the SKILLs definition and the example SKILL ONCE to understand the format with the tool `read_example_skill`.
    2. STRICTLY FOLLOW the syntax and examples provided by the tool `read_example_skills` (frontmatter markdown style).
    3. Analyze the user query and identify the SKILLs
    4. Search for existing SKILLs that match the identified ones with the tool `read_skills_descriptions`, if they exist, return 'skill already exists' and do not create a new one.
    5. If the SKILL doesn't exist, create a new SKILL.md file for each identified SKILL with the tool `write_skill`, providing a concise and clear description and instructions in markdown format.
    5.1 The SKILL has to be generic enough to be reusable for other similar requests.
    6. If the SKILL is correctly created (`write_skill` returns True) then return the text 'skill created', do not add anything else.

    # Tools available
    1. `read_example_skills`: Tool to read the example SKILL.md file
    2. `write_skill`: Tool to write a SKILL.md file
        - Parameters
            - `dir_name`: a one-word lowercase appropriate name for the directory (e.g., 'unix', 'windows', 'python', 'pdf', etc)
            - `file_content`: the content to be written

    # Constraints
    1. The parameter `dir_name` and the 'name' property in the frontmatter must match.
    """,
    tools=[read_example_skills, read_skills_descriptions, write_skill],
    exit_conditions=["text"]
)


@component
class FixedLinkContentFetcher:
    def __init__(self):
        self.fetcher = LinkContentFetcher(
            timeout=3,
            raise_on_failure=False,
            retry_attempts=2
        )

    @component.output_types(streams=list[ByteStream])
    def run(self):
        urls = ["https://agentskills.io/what-are-skills",
                "https://agentskills.io/specification"]
        return {"streams": self.fetcher.run(urls)["streams"]}


pipeline = Pipeline(max_runs_per_component=1)
pipeline.add_component("fetcher", FixedLinkContentFetcher())
pipeline.add_component("html", HTMLToDocument())
pipeline.add_component("summarizer", ChatPromptBuilder(
    template=[
        ChatMessage.from_user("""
        Consider the following definition about SKILLs (developed by Anthropic [Claude]).
        You have to summarize it and return it in clear Markdown format.
        <skill_definition>
        {% for doc in docs %}
            {{ doc.content }}
        {% endfor %}                  
        </skill_definition>
        """)],
    required_variables=["docs"]
))
"""
        Consider the following summarized definition of SKILLs:
        <skill_definition>
        {{replies[0].text}}
        </skill_definition>
                              
        For the code to build the SKILLs, you MUST strictly follow the syntax and examples in the following documents:
        <angular_documentation>
        {% for doc in documentation %}
            {{ doc.content }}
        {% endfor %}
        </angular_documentation>
                              
        User request:
        <user_request>
        {{query}}
        </user_request>
"""
pipeline.add_component("builder", ChatPromptBuilder(
    template=[
        ChatMessage.from_user("""
        Consider the following summarized definition of SKILLs:
        <skill_definition>
        {{replies[0].text}}
        </skill_definition>
                              
        For the code to build the SKILLs, you MUST strictly follow the syntax and examples
        from documentation_tool.
                              
        User request:
        <user_request>
        {{query}}
        </user_request>
        """)
    ],
    required_variables=["replies", "query"]
))
pipeline.add_component("chat_summarizer", thinking_generator)
pipeline.add_component("agent", agent)

pipeline.connect("fetcher.streams", "html.sources")
pipeline.connect("html.documents", "summarizer.docs")
pipeline.connect("summarizer.prompt", "chat_summarizer.messages")
pipeline.connect("chat_summarizer.replies", "builder.replies")
pipeline.connect("builder.prompt", "agent.messages")

skill_tool = PipelineTool(
    pipeline=pipeline,
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user request to identify the SKILLs"
            }
        }
    },
    input_mapping={
        # Mapea la variable "query" del componente "builder" a la variable "query"
        # Tambien mapea la variable "documentation" que viene del State "relevant_documentation" de "inputs_from_state"
        "query": ["builder.query"],
        # "documentation": ["builder.documentation"]
    },
    # TODO: Keeping it just in case
    # Mapea la variable del State "relevant_documentation" a la variable "documentation" (parametro del tool)
    # Se crea en documentation.py
    # inputs_from_state={
    #    "relevant_documentation": "documentation"
    # },
    # TODO: Cambiar a outputs_to_state? Es una tool y no se está conectando a nada directamente
    output_mapping={
        # Mapea el output "messages" del componente "agent" a "messages"
        "agent.messages": "messages"
    },
    name="skill_tool",
    description="Generates SKILLs based on the user request"
)
