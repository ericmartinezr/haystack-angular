from pathlib import Path
from haystack import Pipeline, tracing
from haystack_integrations.components.connectors.langfuse import LangfuseConnector
from haystack.components.agents import Agent
from agents.todo import todo_tool
from agents.coder import coder_tool
from agents.skills import skill_tool
from tools.documentation import documentation_tool
from models.ollama import thinking_generator
from dotenv import load_dotenv

load_dotenv()

tracing.tracer.is_content_tracing_enabled = True


agent = Agent(
    chat_generator=thinking_generator,
    tools=[
        todo_tool,
        coder_tool,
        documentation_tool,
        skill_tool
    ],
    max_agent_steps=12,
    system_prompt="""
You're an expert AI agent designed to orchestrate various agents to accomplish complex tasks. 
You have access to a range of tools that can help you write code, review it and improve it.
Your task is to analyze the user's request, determine which tools to use, 
and orchestrate the agents to accomplish the task effectively.

# Tools available
- `documentation_tool`: Loads in the state the relevant angular documentation related to the user's request.
- `skill_tool`: Manages the skills that will be used during the execution of the task. It creates new skills and reads existing ones.
- `todo_tool`: Generates a TODO list with the user's request broken into smaller steps.

# Workflow
1. Execute the tool `documentation_tool` with a concise query to load the relevant Angular documentation.
2. Provide a detailed query to the tool `skill_tool` to process the required SKILLs.
3. Later provide a detailed query to the tool `todo_tool` to generate a TODO list for the user's request.

Answer with the path of the TODO file generated.
""",
    exit_conditions=["text"],
    state_schema={
        "relevant_documentation": {"type": list}
    }
)

agent.warm_up()

angular = Pipeline(max_runs_per_component=1)
angular.add_component("tracer", LangfuseConnector("Angular Haystack"))
angular.add_component("agent", agent)

angular.draw(path=Path("pipeline.png"))
