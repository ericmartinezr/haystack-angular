from haystack import Pipeline
from haystack.tools import PipelineTool
from haystack.dataclasses import ChatMessage
from haystack.components.agents import Agent
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from models.ollama import coder_generator

agent = Agent(
    chat_generator=coder_generator,
    system_prompt="""
You're an expert software engineer with extensive experience in Angular. You have a deep understanding of Angular's architecture, components, services, and best practices. 
You excel at writing clean, efficient, and maintainable code. You are also skilled at debugging and optimizing Angular applications. 
Your task is to assist in developing and improving an Angular application by providing code snippets, explanations, and guidance based on the user's requests. 

# Instructions  
1. Based on the user's request and the Angular documentation provided, write Angular code that solves the user's request.

# Constraints
1. You're NOT allowed to use your internal knowledge to write code. 
""",
    exit_conditions=["text"]
)

agent.warm_up()

pipeline = Pipeline(max_runs_per_component=1)
pipeline.add_component("builder", ChatPromptBuilder(
    template=[
        ChatMessage.from_user("""
        User request: {{ query }}
        """)],
    required_variables=["query"]
))
pipeline.add_component("agent", agent)

pipeline.connect("builder.prompt", "agent.messages")

coder_tool = PipelineTool(
    pipeline=pipeline,
    # Mapea el input "query" que recibe "coder_tool" a las variables "query" de "retriever" y "builder"
    input_mapping={
        "query": ["builder.query"]
    },
    name="coder_tool",
    description="Writes code based on the user's request and the retrieved documents. The user request is in the 'query' variable and the retrieved documents are in the 'documents' variable.",
)
