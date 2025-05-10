import argparse, hashlib, pathlib, chromadb, tqdm
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"
def sha(s): return hashlib.sha256(s.encode()).hexdigest()[:12]

def chunk(text, max_tokens=300):
    buf, out = [], []
    for para in text.split("\n\n"):
        buf.append(para)
        if len(" ".join(buf).split()) > max_tokens:
            out.append("\n\n".join(buf)); buf=[]
    if buf: out.append("\n\n".join(buf))
    return out

def main(llms_full, collection, db_dir):
    txt = pathlib.Path(llms_full).read_text(encoding="utf-8")
    docs = chunk(txt)
    model = SentenceTransformer(EMBED_MODEL)
    embeds = model.encode(docs, batch_size=32, show_progress_bar=True)

    client = chromadb.PersistentClient(path=db_dir)
    col = client.get_or_create_collection(collection, embedding_function=None)
    col.add(
        documents=docs,
        embeddings=[e.tolist() for e in embeds],
        ids=[sha(d) for d in docs],
        metadatas=[{"source": llms_full}] * len(docs)
    )
    print(f"✔ {len(docs)} chunks → {collection}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("llms_full")
    ap.add_argument("--collection", default="docs")
    ap.add_argument("--db-dir", default="./chroma_db")
    args = ap.parse_args()
    main(args.llms_full, args.collection, args.db_dir)
