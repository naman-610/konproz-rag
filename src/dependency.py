from src.core.domain.pdf_loader import PDFLoader
from src.core.domain.embedding_generator import EmbeddingGenerator
from src.core.domain.vector_db import FAISSIndexer
from src.core.domain.rag_pipeline import RAGPipeline


pdf_loader = None
embedding_generator = None
indexer = None
rag_pipeline = None
