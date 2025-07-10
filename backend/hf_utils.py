import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config.config import config
import os

# Hugging Face Token 설정
hf_token = os.environ.get(config["huggingface"]["access_token"])
# hf_token = os.environ.get("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")

# 사용할 모델의 HuggingFace 경로 매핑
SLM_MODEL_CONFIG = {
    # "hyperclovax": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    "a.x": "skt/A.X-4.0-Light",
    "midm": "K-intelligence/Midm-2.0-Mini-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4b-Instruct",
    # 필요시 여기에 추가!
}

# 모델, 토크나이저 캐싱
SLM_MODEL_CACHE = {}

def load_slm_model(model_key):
    if model_key in SLM_MODEL_CACHE:
        return SLM_MODEL_CACHE[model_key]
    model_name = SLM_MODEL_CONFIG[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto", device_map="auto", token=hf_token)
    print(f"모델과 토크나이저가 로드되었습니다. GPU 사용 장치: {device}")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    SLM_MODEL_CACHE[model_key] = pipe
    return pipe

def run_slm_model(prompt: str, model_key: str, max_new_tokens: int = 512):
    pipe = load_slm_model(model_key)
    outputs = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6)
    # 결과 텍스트에서 프롬프트 부분을 제거
    return outputs[0]['generated_text'].replace(prompt, '').strip()
