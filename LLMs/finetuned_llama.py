from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import get_peft_model, PeftModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load your fine-tuned model instead of the original one
fine_tuned_model_path = "meta-llama/Llama-2-7b-chat-hf"  # cannot use template if not chat version

# Load quantized model
quantization_4bit = BitsAndBytesConfig(load_in_4bit=True)
model_4bit = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model_path,  # Use your fine-tuned model path here
    quantization_config=quantization_4bit,
    device_map="auto",
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)  # Load the tokenizer from the same path


model_4bit = PeftModel.from_pretrained(model_4bit, "./SYC_1029").eval()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# prompt = f"""
#         Below is my diary. Using this Diary, write a prompt for text-to-image process.
        
#         ### Diary
#         A young student sitting alone in a classroom, feeling frustrated and overwhelmed.
#         The student’s face shows sadness and frustration as they look at their books and notes.
#         Around them, other students are working confidently and smiling, creating a contrast.
#         The environment feels heavy with stress, and there is a pile of assignments on the desk.
#         The overall mood is gloomy, with muted colors and soft lighting to emphasize the emotional weight
        
#         ### Prompt
#         """

# template1 = """
#         Below is a request for answering the recipe for some random food. Fill the Recipe template
         
#         ### Question
#         """

# prompt = """ +
# "How do you make javachip frapuccino?"
# """

# template2 = """
# ### Answer: 
# """

# message = [
#     {
#         "role": "system",
#         "content": "You are Edward Lee who is a best cook(i mean chef) ever in the world",
#     },
#     {"role": "user", "content": "How do you make javachip frapuccino?"},
#  ]
message = [
    {
        "role": "system",
        "content": "You are \'Secure Your Contract\' that is an AI based system that takes a contract as an input, detects possibly disadvantageous/dangerous keywords/phrases and provide the refinement suggestion/solution for the input.",
    },
    {"role": "user", "content": "I need to draft a Lease Agreement ('Agreement') for an apartment rental. This Agreement is made on [Effective Date] between John Smith ('Landlord') and Emily Johnson ('Tenant').\n\n**1. Property Description**\nThe Landlord agrees to rent the residential property located at 123 Elm Street, Apt. 4B, Springfield, IL 62701 ('Property').\n\n**2. Term**\nThis Lease shall begin on [Start Date] and shall continue for a term of one year, ending on [End Date].\n\n**3. Rent**\nThe Tenant agrees to pay the Landlord $1,200 per month, due on the first day of each month. Payments must be made via bank transfer or check.\n\n**4. Security Deposit**\nThe Tenant shall pay a security deposit of $1,200, refundable upon lease termination, contingent upon the condition of the Property.\n\n**5. Maintenance and Repairs**\nThe Tenant shall maintain the Property in good condition and promptly report any necessary repairs to the Landlord.\n\n**6. Termination**\nIf the Tenant wishes to terminate this Lease, they must provide the Landlord with 30 days' written notice. The Landlord may terminate the Lease for non-payment of rent or breach of terms.\n\n**7. Governing Law**\nThis Agreement shall be governed by the laws of the State of Illinois.\n"},
]
        
inputs = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=False, return_tensors="pt")
# print(tokenizer.batch_decode(inputs))



start_event.record()


input_len = len(tokenizer.batch_decode(inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
# inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model_4bit.generate(inputs.to(device)) #.input_ids.to(device))
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][input_len:]

end_event.record()
torch.cuda.synchronize()

inference_time = start_event.elapsed_time(end_event)

print("\n🚨Answer🚨:", outputs)
print(f"\ninference time : {(inference_time * 1e-3):.2f} sec")

result = open("text2prompt_4bit.txt", 'w')
result.write(outputs)
result.write(f"\ninference time : {(inference_time * 1e-3):.2f} sec")