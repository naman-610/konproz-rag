from openai import OpenAI
from typing import List
from loguru import logger

from src.core.domain.embedding_generator import EmbeddingGenerator
from src.core.domain.vector_db import FAISSIndexer
from src.core.settings import config_settings


class RAGPipeline:
    def __init__(
        self,
        indexer: FAISSIndexer,
        corpus: List[str],
        embedding_generator: EmbeddingGenerator,
        openai_model: str = config_settings.TEXT_LLM_MODEL,
    ):
        self.indexer = indexer
        self.corpus = corpus
        self.embedding_generator = embedding_generator
        self.openai_model = openai_model
        self._llm: OpenAI = OpenAI()

    def generate_response(self, query: str, top_k: int = 15) -> str:
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]

        indices, distances = self.indexer.search(query_embedding, top_k)
        retrieved_docs = [self.corpus[idx] for idx in indices]

        # Step 3: Prepare prompt for OpenAI
        prompt = (
            f"Question: {query}\n\nContext:\n"
            + "\n".join(retrieved_docs)
            + "\n\nAnswer:"
        )

        # Step 4: Generate response using OpenAI
        try:
            response = self._llm.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.\nOnly answer the question from the given context",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            logger.info(response.choices[0].message)
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "I'm sorry, I couldn't process your request at the moment."
