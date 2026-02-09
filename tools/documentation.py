from typing import List
from haystack import Pipeline, component
from haystack.tools import PipelineTool, Tool
from haystack.components.caching import CacheChecker
from haystack.dataclasses.byte_stream import ByteStream
from haystack.components.writers import DocumentWriter
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.converters import MarkdownToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from models.ollama import thinking_generator
from models.ollama import doc_embedder, text_embedder
from constants import THINKING_MODEL
from dotenv import load_dotenv

load_dotenv()

# https://angular.dev/llms-full.txt
# https://angular.dev/assets/context/llms-full.txt
LLM_URL = "https://angular.dev/assets/context/llms-full.txt"

# Vector store
document_store = PgvectorDocumentStore(
    embedding_dimension=768,  # Because of nomic embed text
    vector_function="cosine_similarity",
    # recreate_table=True,
    search_strategy="exact_nearest_neighbor"
)

# Retrieves documents from the vector store
retriever = PgvectorEmbeddingRetriever(
    document_store=document_store,
    vector_function="cosine_similarity",
    top_k=2
)


@component
class AngularLLMFetcher:
    def __init__(self):
        self.fetcher = LinkContentFetcher(
            timeout=3,
            raise_on_failure=False,
            retry_attempts=2
        )

    @component.output_types(streams=List[ByteStream])
    def run(self):
        cache = CacheChecker(document_store=document_store,
                             cache_field="meta.url")
        results = cache.run(items=[LLM_URL])
        return {"streams": self.fetcher.run(results["misses"])["streams"]}


# def documentation_pipeline(query: str):
@component
class DocumentationPipeline:

    @component.output_types(relevant_documentation=list)
    def run(self, query: str):
        # Indexing pipeline
        index_pipeline = Pipeline(max_runs_per_component=1)
        # index_pipeline.add_component("cache_checker", CacheChecker(
        #    document_store=document_store, cache_field="url"))
        index_pipeline.add_component("fetcher", AngularLLMFetcher())
        index_pipeline.add_component(
            "converter", MarkdownToDocument(store_full_path=True))
        index_pipeline.add_component("cleaner", DocumentCleaner())
        index_pipeline.add_component(
            "splitter",
            DocumentSplitter(
                split_by="word",
                split_length=250,
                split_overlap=30
            )
        )
        index_pipeline.add_component("embedder", doc_embedder)
        index_pipeline.add_component(
            "document_writer",
            DocumentWriter(document_store=document_store)
        )
        index_pipeline.connect("fetcher.streams", "converter.sources")
        index_pipeline.connect("converter.documents", "cleaner.documents")
        index_pipeline.connect("cleaner.documents", "splitter.documents")
        index_pipeline.connect("splitter.documents", "embedder.documents")
        index_pipeline.connect("embedder.documents",
                               "document_writer.documents")

        # Searching pipeline
        search_pipeline = Pipeline(max_runs_per_component=1)
        search_pipeline.add_component("text_embedder", text_embedder)
        search_pipeline.add_component("retriever", retriever)
        search_pipeline.connect("text_embedder.embedding",
                                "retriever.query_embedding")

        # Execute both pipelines sequentially
        index_pipeline.run(data={})
        results = search_pipeline.run({"text_embedder": {"text": query}})

        return {"relevant_documentation": results["retriever"]["documents"]}


thinking_generator = OllamaChatGenerator(
    model=THINKING_MODEL,
    timeout=3600,
    generation_kwargs={
        "temperature": 0.1
    }
)

pipeline = Pipeline(max_runs_per_component=1)
pipeline.add_component("documentation", DocumentationPipeline())
pipeline.add_component("builder", ChatPromptBuilder(
    template=[
        ChatMessage.from_user("""
        Analyze the following Angular documentation and guidelines, extract the most relevant information 
        and best practices related to the user request, and return it in clear Markdown format.
                              
        <angular_documentation>
        {% for doc in docs %}
            {{ doc.content }}
        {% endfor %}                  
        </angular_documentation>
                              
        User request: {{query}}
        """)],
    required_variables=["docs", "query"]
))
pipeline.add_component("generator", thinking_generator)

pipeline.connect("documentation.relevant_documentation",
                 "builder.docs")
pipeline.connect("builder.prompt", "generator.messages")

documentation_tool = PipelineTool(
    pipeline=pipeline,
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user request to identify the documentation and guidelines"
            }
        }
    },
    input_mapping={
        # Mapea el parametro "query" a la variable "query" del componente "builder" y "documentation"
        "query": ["builder.query", "documentation.query"],
    },
    # output_mapping={
    #    "generator.replies": "relevant_replies"
    # },
    # outputs_to_state={
    #    # Mapea el output "relevant_documentation" del componente "documentation" a "messages"
    #    "relevant_documentation": {"source": "relevant_replies"}
    # },
    name="documentation_tool",
    description="Retrieves documentation and guidelines for Angular development."
)


# documentation_tool = Tool(
#    parameters={
#        "type": "object",
#        "properties": {
#            "query": {
#                "type": "string",
#                "description": "The query to search for in the documentation."
#            }
#        }
#    },
#    # Toma la variable "relevant_documentation" retornado por la funcion y la guarda en el State
#    # con el nombre "relevant_documentation"
#    outputs_to_state={
#        "relevant_documentation": {"source": "relevant_documentation"}
#    },
#    function=documentation_pipeline,
#    name="documentation_tool",
#    description="Retrieves documentation and guidelines for Angular development."
# )
