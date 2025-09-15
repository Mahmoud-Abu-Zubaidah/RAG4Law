import os
import random
import warnings
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma

warnings.filterwarnings('ignore')
load_dotenv()
os.environ["HF_HOME"] = "/cache"

def prepare_vector_db():
    """Load documents, split them, embed, and store in Chroma."""
    laws_dir = os.getenv("laws_dir")
    VDB_name = os.getenv("VectorDB_Name")
    V_DB_dir = os.getenv("V_DB_dir")
    embedding_model_name = os.getenv("embedding_model_name")

    if not all([laws_dir, VDB_name, V_DB_dir, embedding_model_name]):
        raise ValueError("Missing required environment variables in .env file.")

    print("Start Splitting Text.")
    loader = TextLoader(laws_dir, encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=400, separator="\n", chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    print("Ends Splitting Text.")
    print("Summary:", len(texts), "Chunks")

    # Shuffle laws
    random.seed(42)
    random.shuffle(texts)

    # Embed + Store
    print("Start Storing Vectors to Local Chroma")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vector_store = Chroma.from_documents(
        texts,
        embeddings,
        collection_name=VDB_name,
        persist_directory=V_DB_dir,
    )

    print("Storing Finish.\nDataBase is ready to use.")

    return vector_store