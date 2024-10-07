from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# 모델과 토크나이저 불러오기
model_name = "meta-llama/Llama-2-7b-hf"  # LLaMA2 모델, 7B 파라미터 버전
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# 디바이스 설정 (GPU가 있다면 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 프롬프트 정의
prompt = """
You are a contract analysis AI. Your task is to identify risks and provide solutions from the following contract:

[Insert contract text here]

Please provide a detailed risk analysis along with mitigation strategies.
"""

# 프롬프트 토큰화
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 모델에 입력 및 결과 생성
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,  # 출력의 최대 길이
        num_return_sequences=1,  # 생성할 출력의 개수
        temperature=0.7,  # 샘플링 다양성 (낮을수록 결정적)
        top_p=0.9,  # 출력의 상위 확률 범위
        do_sample=True  # 샘플링 여부
    )

# 결과 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 결과 출력
print(generated_text)