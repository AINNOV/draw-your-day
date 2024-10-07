from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)


negative_keywords = [
    "immediately",
    "required",
    "all",
    "must",
    "any",
    "should",
    "forever",
    "terminate",
    "penalty"
]


def create_template(negative_keywords):
    return f"계약서의 다음 조항에 대해 분석해: {' '.join(negative_keywords)}"


def generate_response(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_p=0.95,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

for keyword in negative_keywords:
    template = create_template([keyword])
    response = generate_response(template)
    print(f"Template: {template}\nResponse: {response}\n")