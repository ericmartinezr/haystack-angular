from haystack import Pipeline
from haystack.tools import PipelineTool
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from models.ollama import thinking_generator
from tools.write import write_todo
from agents.schema import state

agent = Agent(
    chat_generator=thinking_generator,
    tools=[write_todo],
    max_agent_steps=10,
    system_prompt="""
Your job is to generate a TODO list with the steps to solve the user's request. 
The TODO list should be in markdown format, with each step as a bullet point.

# Workflow
1. Analyze the Angular documentation and the user's request and break it down into smaller and actionable steps.
2. Write the TODO list in markdown format, with each step as a bullet point. **DO NOT** use code in the TODO list, only plain text describing the steps to follow.
3. Save the TODO list in a file using the tool `write_todo`, providing the content of the TODO.md file as an argument.
""",
    state_schema={
        "documentation": {"type": list}
    },
    exit_conditions=["text"]
)

agent.warm_up()

# Prompt
pipeline = Pipeline(max_runs_per_component=1)
pipeline.add_component("builder", ChatPromptBuilder(
    template=[
        ChatMessage.from_user(
            """ 
            <angular_documentation>
            {% for doc in documentation %}
                {{ doc.content }}
            {% endfor %}        
            </angular_documentation>

            User request: {{ query }}
            """)],
    required_variables=["query", "documentation"]
))
pipeline.add_component("agent", agent)

pipeline.connect("builder.prompt", "agent.messages")

todo_tool = PipelineTool(
    pipeline=pipeline,
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's request for which the TODO list should be generated."
            }
        },
        "required": ["query"]
    },
    # Mapea el input "query" que recibe "coder_tool" a las variables "query" de "builder"
    # Tambien mapea la variable "documentation" que viene del State "relevant_documentation" de "inputs_from_state"
    input_mapping={
        "query": ["builder.query"],
        "documentation": ["builder.documentation"]
    },
    # Mapea la variable del State "relevant_documentation" a la variable "documentation" (parametro del tool)
    # Se crea en documentation.py
    inputs_from_state={
        "relevant_documentation": "documentation"
    },
    name="todo_tool",
    description="Generates a TODO list based on the user's request. The user request is in the 'query' variable.",
)
