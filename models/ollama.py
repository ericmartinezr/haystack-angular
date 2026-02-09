from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from constants import CODER_MODEL, THINKING_MODEL, EMBEDDER_MODEL

thinking_generator = OllamaChatGenerator(
    model=THINKING_MODEL,
    timeout=3600,
    generation_kwargs={
        "temperature": 0.1
    }
)


coder_generator = OllamaChatGenerator(
    model=CODER_MODEL,
    timeout=3600,
    generation_kwargs={
        "temperature": 0.5
    }
)


doc_embedder = OllamaDocumentEmbedder(
    model=EMBEDDER_MODEL,
    url="http://localhost:11434",
    progress_bar=True,
    generation_kwargs={
        "temperature": 0.1
    }
)

text_embedder = OllamaTextEmbedder(
    model=EMBEDDER_MODEL,
    url="http://localhost:11434",
    generation_kwargs={
        "temperature": 0.1
    }
)
