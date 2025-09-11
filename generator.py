import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def __init_env():
    """
    Load environment variables and set HF cache path.
    """
    load_dotenv()
    os.environ["HF_HOME"] = "D:/cache"

    model_name = os.getenv("decoder_model_name")
    token = os.getenv("hugging_face_key")

    if not model_name or not token:
        raise ValueError("Missing decoder_model_name or hugging_face_key in .env")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model_name, token, device


def initialize_decoder():
    """
    Load a HuggingFace causal LM model with 4-bit quantization.
    Returns (tokenizer, model).
    """
    global device
    model_name, token, device = __init_env()
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        quantization_config=quant_config,
        token=token,
    )
    return tokenizer, model


def get_response(text, tokenizer, model):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]

    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=0.4,
        max_length=2048 - input_len,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )

    response = tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    # Optional: clean response if AI markers exist
    response = response.split("### Response: [|AI|]")[-1].strip()
    return response