# RAG Application

Retrieval-Augmented Generation (RAG) is a system designed to combine information retrieval with generative AI models, enabling accurate and contextually relevant responses based on large document corpora or databases.

---


## Installation

### Prerequisites

- Python 3.12+
- OpenAI API Key

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/naman-610/konproz-rag.git

2. Install dependencies
   pip install .

3. python src/app.py

---

## Key notes

1. Initially the application will take time to load the pdf, since it is storing all the vector in local memory and clean it up post application shutdown

2. Add OpenAI API key in .env file

---

## RAG API

curl --location 'http://0.0.0.0:8003/_v1/rag' \
--header 'Content-Type: application/json' \
--data '{
    "query": "Explain the impact of the 101st Constitutional Amendment on India'\''s taxation system."
}'

