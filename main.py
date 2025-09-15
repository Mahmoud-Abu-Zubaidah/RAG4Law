from dotenv import load_dotenv
from vectorizer import prepare_vector_db
from RAG import Retrive_relative_documents
from generator import initialize_decoder, get_response
from preparing_laws import preprocess_laws
import os

load_dotenv()
os.environ["HF_HOME"] = "/cache"

if __name__ == "__main__":
    prompt = "إذا ارتكب الجاني جريمته لمصلحة دولة أجنبية أو تنظيم غير مشروع"
    ## Optional
    # preprocess_laws()

    # 1. Build DB (only first time)
    # prepare_vector_db()

    # 2. Retrieve docs
    docs = Retrive_relative_documents(prompt, K=3)
    laws="".join([doc for doc in docs])
    # print("Retrieved docs:", laws)

    # 3. Load decoder model

    tokenizer, model, device = initialize_decoder()

    # 4. Generate response
    response = get_response(
        prompt, laws, tokenizer, model, device
    )
    print("\nAI Response:\n", response)