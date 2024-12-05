from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        embeddings = []
        total = len(texts)
        for start_idx in range(0, total, batch_size):
            batch_texts = texts[start_idx : start_idx + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=True)
            embeddings.extend(batch_embeddings)
            print(
                f"Processed batch {start_idx // batch_size + 1} / { (total // batch_size) + 1}"
            )
        return embeddings
