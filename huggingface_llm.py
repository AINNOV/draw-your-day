from transformers import pipeline


from huggingface_hub import login

login(token='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')


generator = pipeline('text-generation', model='distilgpt2')

prompt = "My name is"
response = generator(prompt, max_length=50)
print(response)