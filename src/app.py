from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger
from torch.cuda import empty_cache

from src import dependency
from src.core.routers import rag as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize components
    global pdf_loader, embedding_generator, indexer, rag_pipeline
    pdf_path = "task_document.pdf"

    logger.debug("Loading PDF...")
    dependency.pdf_loader = dependency.PDFLoader(pdf_path)
    dependency.pdf_loader.load_pdf()

    logger.debug("Generating embeddings...")
    dependency.embedding_generator = dependency.EmbeddingGenerator()
    corpus_embeddings = dependency.embedding_generator.generate_embeddings(
        dependency.pdf_loader.pages
    )

    logger.debug("Building FAISS index...")
    embedding_dimension = len(corpus_embeddings[0])
    dependency.indexer = dependency.FAISSIndexer(embedding_dimension)
    dependency.indexer.build_index(corpus_embeddings)

    logger.debug("Initializing RAG Pipeline...")
    dependency.rag_pipeline = dependency.RAGPipeline(
        indexer=dependency.indexer,
        corpus=dependency.pdf_loader.pages,
        embedding_generator=dependency.embedding_generator,
    )
    logger.debug("RAG Pipeline is ready.")

    yield
    # Cleanup code moved from shutdown_event
    logger.debug("Shutting down application and cleaning up resources...")
    del dependency.pdf_loader
    del dependency.embedding_generator
    del dependency.indexer
    del dependency.rag_pipeline
    empty_cache()

    logger.debug("Cleanup completed.")


app = FastAPI(lifespan=lifespan)

app.include_router(rag_router.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
