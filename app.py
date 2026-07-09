import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Assistant RH — Convention Syntec — API RAG")

DOCS_DIR = "/tmp/documents/"
_rag = None  # cache global

def get_rag():
    """Construit le pipeline au premier appel, puis le réutilise."""
    global _rag
    if _rag is not None:
        return _rag

    import urllib.request
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate

    os.makedirs(DOCS_DIR, exist_ok=True)
    dest = os.path.join(DOCS_DIR, "CONVENTION_SYNTEC.pdf")
    if not os.path.exists(dest):
        urllib.request.urlretrieve(
            "https://github.com/archiducarmel/SupDeVinci_M1_MachineLearning_DeepLearning/"
            "releases/download/datas/CONVENTION_SYNTEC.pdf",
            dest,
        )

    all_docs = []
    for f in sorted(os.listdir(DOCS_DIR)):
        if f.endswith(".pdf"):
            all_docs.extend(PyPDFLoader(os.path.join(DOCS_DIR, f)).load())

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    ).split_documents(all_docs)

    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-nemotron-embed-1b-v2", truncate="NONE"
    )
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatNVIDIA(
        model="openai/gpt-oss-120b", temperature=0.2, max_completion_tokens=2048
    )
    prompt = ChatPromptTemplate.from_template(
        "Tu es un assistant RH expert de la convention collective Syntec. "
        "Réponds à la QUESTION en t'appuyant UNIQUEMENT sur le CONTEXTE ci-dessous.\n"
        "Si l'information n'y figure pas, réponds exactement : « Je ne sais pas ».\n"
        "Sois concis et cite la source (document et numéro de page).\n\n"
        "CONTEXTE :\n{context}\n\nQUESTION : {input}"
    )

    def extract_answer(response):
        text = (response.content or "").strip()
        if not text:
            text = (response.additional_kwargs.get("reasoning_content", "") or "").strip()
        return text

    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('source', '?').split('/')[-1]} — page {d.metadata.get('page')}] "
            f"{d.page_content}"
            for d in docs
        )

    generation = prompt | llm | extract_answer
    _rag = (retriever, generation, format_docs)
    return _rag


class QuestionIn(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: QuestionIn):
    retriever, generation, format_docs = get_rag()
    docs = retriever.invoke(payload.question)
    answer = generation.invoke({"context": format_docs(docs), "input": payload.question})
    sources = [
        {"document": d.metadata.get("source", "?").split("/")[-1], "page": d.metadata.get("page")}
        for d in docs
    ]
    return {"answer": answer, "sources": sources}
