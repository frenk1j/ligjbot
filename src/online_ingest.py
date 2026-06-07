from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def load_pdf_as_documents(pdf_path: str) -> List[Document]:
    path = Path(pdf_path)
    loader = PyPDFLoader(str(path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    for d in chunks:
        meta = d.metadata or {}
        meta.setdefault("source", str(path.name))
        d.metadata = meta

    return chunks
