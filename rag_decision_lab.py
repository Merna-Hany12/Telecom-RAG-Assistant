
import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gc

DATA_PATH = r"C:\Users\merna\Desktop\telecom-rag-assistant\data\data"

EMBEDDING_MODELS = {
    "e5": "intfloat/multilingual-e5-base",
    "bge": "BAAI/bge-m3",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
}


# ============================================================
# 1. CHUNKING
# ============================================================

def chunk_text(text):
    max_size = 700

    # TODO: Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # TODO: clean paragraph
        # TODO: build chunks with size limit
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) > max_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================
# 2. EMBEDDINGS
# ============================================================

def create_embeddings(chunks, model_name):
    """
    Converts text chunks into vectors.
    """

    print(f"Creating embeddings with {model_name}...")

    # TODO: load model
    model = SentenceTransformer(model_name)

    # Handle E5 only
    if "e5" in model_name:
        chunks = ["passage: " + c for c in chunks]

    # TODO: encode chunks
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # TODO: convert to float32
    embeddings = np.array(embeddings).astype("float32")

    return model, embeddings


# ============================================================
# 3. FAISS INDEX
# ============================================================

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    # TODO: create index
    index = faiss.IndexFlatIP(dimension)

    # TODO: add embeddings to index
    index.add(embeddings)

    return index


# ============================================================
# 4. RETRIEVAL
# ============================================================

def retrieve(query, model, model_name, index, chunks, metadata, top_k):
    print(f"Searching ({model_name}) for: {query}")

    # TODO: encode query
    if "e5" in model_name:
        query = "query: " + query

    query_emb = model.encode([query], normalize_embeddings=True)
    query_emb = np.array(query_emb).astype("float32")

    # TODO: search index
    distances, indices = index.search(query_emb, top_k)

    results = []

    # TODO: build results list
    for i in range(top_k):
        idx = indices[0][i]

        results.append({
            "text": chunks[idx],
            "source": metadata[idx]["source"],
            "score": float(distances[0][i])
        })

    return results


# ============================================================
# 5. HTML OUTPUT
# ============================================================

def save_results_html(all_results, query):
    html = f"<h1>Query: {query}</h1>"

    for model_name, results in all_results.items():
        html += f"<h2>Model: {model_name}</h2>"

        for i, r in enumerate(results):
            html += f"""
            <div style='margin:15px; padding:10px; border:1px solid gray'>
                <b>Result {i+1}</b><br>
                <b>Score:</b> {r['score']}<br>
                <b>Source:</b> {r['source']}<br>
                <p>{r['text'][:300]}</p>
            </div>
            """

    with open("results.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(" Saved results to results.html")


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

if __name__ == "__main__":

    all_chunks = []
    metadata = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".md"):

            with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
                text = f.read()

            doc_chunks = chunk_text(text)

            for chunk in doc_chunks:
                all_chunks.append(chunk)
                metadata.append({"source": file})

    query = "ايه عروض النت ؟"

    all_results = {}

    for name, model_path in EMBEDDING_MODELS.items():
        model, embeddings = create_embeddings(all_chunks, model_path)


        index = build_faiss_index(embeddings)
        results = retrieve(
            query=query,
            model=model,
            model_name=model_path,
            index=index,
            chunks=all_chunks,
            metadata=metadata,
            top_k=5
        )

        all_results[name] = results

        del model
        del embeddings
        del index
        gc.collect()


    save_results_html(all_results, query)