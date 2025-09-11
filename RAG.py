import os
import warnings
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

warnings.filterwarnings('ignore')


def __init_env():

    """
    Load environment variables and set HF cache path.
    """

    load_dotenv()
    os.environ["HF_HOME"] = "D:/cache"

    embedding_model_name = os.getenv('embedding_model_name')
    V_DB_name = os.getenv('VectorDB_Name')
    V_DB_dir = os.getenv('V_DB_dir')

    if not embedding_model_name or not V_DB_name or not V_DB_dir:
        print("Error: Missing environment variables. Check your .env file.")

    return embedding_model_name, V_DB_name, V_DB_dir


def Retrive_relative_documents(prompt, K=5):
    embedding_model_name, V_DB_name, V_DB_dir = __init_env()

    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
    V_DB = Chroma(
        collection_name=V_DB_name,
        embedding_function=embedding,
        persist_directory=V_DB_dir,
    )

    retriever_setting = V_DB.as_retriever(
        search_type="similarity",
        search_kwargs={"k": K},
    )

    results = retriever_setting.get_relevant_documents(prompt)
    return [rule.page_content for rule in results]