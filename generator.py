import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate


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

def template_prompt():
    """
    Create a prompt template for legal case analysis in Arabic.
    Returns a ChatPromptTemplate object.

    """
    template = """
    أنت مساعد قانوني ذكي. مهمتك تحليل الحالة القانونية التالية بدقة بناءً على نصوص القوانين المدرجة في السياق فقط، ثم تحديد مدى انطباق أي قانون على هذه الحالة مع تبرير رأيك.

    == وصف الحالة للمستخدم ==
    {case_description}

    == نصوص القوانين أو المواد ذات الصلة (سياق مسترجع) ==
    {legal_context}

    == التعليمات ==
    - حلل القضية بدقة ثم وضّح العناصر القانونية الأساسية.
    - اربط الوقائع مع القوانين المسترجعة فقط (لا تستخدم خبرتك الذاتية أو مصادر خارج ما هو معروض).
    - اذكر نصوص أو مواد القانون المناسبة وأسباب الانطباق.
    - إذا لم يطبق أي قانون بدقة، لا تحلل شيء واطلب المزيد من التوضيح.

    أجب باللغة العربية الفصحى وبأسلوب قانوني مختصر وواضح.
    جاوب بالنموذج التالي:
    ---
    تحليل الوقائع القانونية:
    [تحليل موجز]

    القوانين الملائمة:
    [أذكر المواد/النصوص المناسبة]

    الرأي النهائي:
    [رأي قانوني مختصر وإخلاء مسؤولية: هذا التحليل للأغراض التوضيحية فقط]
    """

    prompt = ChatPromptTemplate.from_template(template)
    # prompt = prompt.format(case_description = case_description,legal_context= legal_context)
    return prompt

def initialize_decoder():
    """
    Load a HuggingFace causal LM model with 4-bit quantization.
    Returns (tokenizer, model).
    """
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
    return tokenizer, model, device


def get_response(text, laws, tokenizer, model, device):
    """Generate a response from the model given input text."""
    prompt = template_prompt()
    prompt = prompt.format(case_description=text, legal_context=laws)
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generate_ids = model.generate(
        input_ids,
        top_p=0.9,
        temperature=0.4,
        max_new_tokens=2024,   
        min_new_tokens=4,
        repetition_penalty=1.2,
        do_sample=True,
    )

    response = tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    # Clean: remove the prompt if still present
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response