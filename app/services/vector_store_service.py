from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings

SEED_KNOWLEDGE = [
    "SIP (Systematic Investment Plan) means investing a fixed amount in a mutual fund at regular "
    "intervals, usually monthly. It averages out purchase cost over time (rupee-cost averaging) "
    "and builds the habit of regular investing.",
    "An emergency fund is money kept in a safe, easily accessible account to cover unexpected "
    "expenses. A common rule of thumb is 3 to 6 months of essential living expenses.",
    "Compound interest is interest earned on both the original principal and previously earned "
    "interest. The longer the money stays invested, the faster it grows.",
    "Diversification means spreading money across different asset classes such as equity, debt, "
    "gold and cash, so that a fall in one does not hurt the whole portfolio.",
    "The expense ratio is the yearly fee a mutual fund charges, shown as a percentage of assets. "
    "Lower expense ratios leave more of the returns with the investor.",
    "EMI (Equated Monthly Instalment) is the fixed monthly payment for a loan. It contains both "
    "interest and principal repayment. Longer tenure lowers the EMI but increases total interest.",
    "The 50/30/20 budgeting rule suggests spending 50% of income on needs, 30% on wants and "
    "saving or investing 20%.",
    "Inflation is the general rise in prices over time. It reduces purchasing power, so savings "
    "should ideally earn a return higher than inflation.",
]


class VectorStoreService:
    """Stores finance knowledge as embeddings and finds the most similar text."""

    def __init__(self, settings: Settings):
        # 1) Embedding model from HuggingFace (downloaded on first run, then cached).
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        # 2) The vector database (Chroma), saved to a local folder.
        self.store = Chroma(
            collection_name="finance_knowledge",
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_dir,
        )

        # 3) Text splitter that cuts long text into small chunks.
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)

    def add_texts(self, texts: list[str], source: str = "manual") -> None:
        docs = [Document(page_content=t, metadata={"source": source}) for t in texts]
        chunks = self.splitter.split_documents(docs)
        self.store.add_documents(chunks)

    def seed_if_empty(self) -> None:
        """Load the starter knowledge only once."""
        if not self.store.get(limit=1)["ids"]:
            self.add_texts(SEED_KNOWLEDGE, source="seed")

    def search(self, query: str, k: int = 3) -> list[Document]:
        return self.store.similarity_search(query, k=k)


    def get_retriever(self, k: int = 3):
        """Return a LangChain retriever (a Runnable: question -> list of Documents)."""
        return self.store.as_retriever(search_kwargs={"k": k})