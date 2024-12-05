import faiss
import numpy as np
from typing import List


class FAISSIndexer:
    def __init__(self, embedding_dimension: int):
        self.dimension = embedding_dimension
        self.index = faiss.IndexFlatL2(self.dimension)

    def build_index(self, embeddings: List[List[float]]):
        print("Building FAISS index...")
        faiss_embeddings = np.array(embeddings).astype("float32")
        self.index.add(faiss_embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors")

    def search(self, query_embedding: List[float], top_k: int = 5):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), top_k
        )
        return indices[0], distances[0]
