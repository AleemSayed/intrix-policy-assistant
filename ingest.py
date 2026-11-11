from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# Project folders
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "data" / "policies"
DB_DIR = BASE_DIR / "vectorstore" / "chroma_db"

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def split_into_chunks(text):
    """Split text into overlapping chunks."""
    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def load_documents():
    """Read PDF files and preserve page information."""
    documents = []
    metadatas = []
    ids = []

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIR}"
        )

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if not text.strip():
                continue

            chunks = split_into_chunks(text)

            for chunk_number, chunk in enumerate(chunks):
                documents.append(chunk)

                metadatas.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number
                })

                ids.append(
                    f"{pdf_path.stem}_page{page_number}_chunk{chunk_number}"
                )

    return documents, metadatas, ids


def main():
    print("\nStarting document ingestion...\n")

    # Extract and split PDF documents
    documents, metadatas, ids = load_documents()

    print(f"\nTotal chunks created: {len(documents)}")

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Convert chunks into embeddings
    print("\nCreating embeddings...")
    embeddings = model.encode(
        documents,
        show_progress_bar=True
    ).tolist()

    # Connect to persistent ChromaDB
    print("\nConnecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name="company_policies",
        metadata={"description": "Intrix sample company policies"}
    )

    # Upsert prevents duplicate IDs when ingestion is repeated
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("\nIngestion completed successfully!")
    print(f"Documents processed: {len(set(m['source'] for m in metadatas))}")
    print(f"Total chunks stored: {collection.count()}")
    print(f"Database location: {DB_DIR}")


if __name__ == "__main__":
    main() 