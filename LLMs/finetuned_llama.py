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


model_4bit = PeftModel.from_pretrained(model_4bit, "./SYC_1105").eval()

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

contract_text = '''This agreement, made on the …… day of the …………….month of the year………………
Between:
………………………………………………………(hereinafter referred to as "the Employer")
and
……………………………………………………… (hereinafter referred to as "the Employee")
WHEREAS the Employee and the Employer wish to enter into an employment agreement
governing the terms and conditions of employment;
THIS AGREEMENT WITNESSETH that in consideration of the premises and mutual covenants
and agreements hereinafter contained is hereby acknowledged and agreed by and between the
parties hereto as follows:
1. Term of Employment
The employment of the Employee shall commence from the date hereof and continue for an
indefinite term until terminated in accordance with the provisions of this agreement.
2. Probation
The parties hereto agree that the initial six (6) month period of this agreement is "Probationary"
in the following respects:
a. the Employer shall have an opportunity to assess the performance, attitude, skills and other
employment-related attributes and characteristics of the Employee;
b. the Employee shall have an opportunity to learn about both the Employer and the position of
employment;
c. either party may terminate the employment relationship at any time during the initial six
month period with advance notice of seven days with justifiable reason, in which case there
will be no continuing obligations of the parties to each other, financial or otherwise.
3. Compensation and Benefits
In consideration of the services to be provided by him hereunder, the Employee, during the term
of his employment, shall be paid a basic salary of Nu. _______ a month/ week, less applicable
statutory deductions. In addition, the Employee is entitled to receive benefits in accordance with
the Employer's standard benefit package, as amended from time to time. 
4. Duties and Responsibilities
The Employee shall be employed in the capacity of __________, the current duties and
responsibilities of which are set out in Annexure "A" annexed hereto and forming part of this
agreement. These duties and responsibilities may be amended from time to time in the sole
discretion of the Employer, subject to formal notification of same being provided to the
Employee.
5. Termination of Employment
Subsequent to completion of the probationary term of employment referred to in paragraph 2
herein, the Employer may terminate the employment of the Employee at any time:
a. for just cause at common law, in which case the Employee is not entitled to any advance
notice of termination or compensation in lieu of notice;
b. the Employee and employer may terminate their employment at any time by providing
atleast seven days notice for probationer and 1 month advance notice for their intention to
terminate the contract of employment or payment in lieu thereof.
The entitlements and termination of services will be governed by the Labour and Employment
Act, 2007 and its regulations and laws of the land.
 6. Confidentiality
The Employee acknowledges that, in the course of performing and fulfilling his duties
hereunder, he may have access to and be entrusted with confidential information concerning the
present and contemplated financial status and activities of the Employer, the disclosure of any of
which confidential information to competitors of the Employer would be highly detrimental to
the interests of the Employer. The Employee further acknowledges and agrees that the right to
maintain the confidentiality of such information constitutes a proprietary right which the
Employer is entitled to protect. Accordingly, the Employee covenants and agrees with the
Employer that he will not, during the continuance of this agreement, disclose any of such
confidential information to any person, firm or corporation, nor shall he use same, except as
required in the normal course of his engagement hereunder, and thereafter he shall not disclose
or make use of the same.
7. Assignment (Transfer of Contract of Employment)
This agreement shall be assigned by the Employer to any successor employer and be binding
upon the successor employer with the consent of the employee. The Employer shall ensure that
the successor employer shall continue the provisions of this agreement as if it were the original
party of the first part. This agreement may not be assigned by the Employee.
8. Severability
Each paragraph of this agreement shall be and remain separate from and independent of and
severable from all and any other paragraphs herein except where otherwise indicated by the
context of the agreement. The decision or declaration that one or more of the paragraphs are null
and void shall have no effect on the remaining paragraphs of this agreement. 
9. Working Conditions
Sr. Rights Provisions Remarks
1 Working
Hours and
rest periods
8 hours a day excluding
meal breaks
Minimum of 1.5 times at the rate of daily wage (10 PM to
8 AM in the following morning). One day rest period after
six days of work.
2 Public
holidays
Minimum ( ) days Excluding other leave entitlements (Casual, annual,
medical etc.)Both the parties may agree to substitute
public holiday with another public holiday
3 Leave Casual ( )
Annual Leave ( )
Sick Leave ( )
Maternity leave ( )
Paternity Leave ( )
The leave provided must at the minimum be provided as
prescribed by the Regulations on leave
4 Provident
Fund
 Contribution of minimum of (____ %) but must be above
the minimum ceiling and eligibility shall be governed by
the regulations on Provident Fund
5 Gratuity Eligible after completion of ( ) years of continuous
employment. Shall be calculated on the last basic salary
multiplied by number of years of service.
6. OHS
equipment
All Personal Protective Equipment (PPE) required for the occupation shall be
provided free of cost by the employers and shall be governed by the regulations in
force.
10. Notice
Any notice required to be given hereunder shall be deemed to have been properly given if
delivered personally or sent by pre-paid registered mail as follows:
a. to the Employee: [address]
b. to the Employer: [address]
and if sent by registered mail shall be deemed to have been received on the 5 working days of
uninterrupted postal service following the date of mailing. Either party may change its address
for notice at any time, by giving notice to the other party pursuant to the provisions of this
agreement. 
11. Interpretation of Agreement
The validity, interpretation, construction and performance of this agreement shall be governed by
the Labour and Employment Act, 2007 and its Regulations. This agreement shall be interpreted
with all necessary changes in gender and in number as the context may require and shall ensure
to the benefit of and be binding upon the respective successors and assigns of the parties hereto.
IN WITNESS WHEREOF the parties hereto have caused this agreement to be executed
as of …..day…… month…...year and shall each retain a copy of the agreement in original.
(Affix legal stamp) (Affix legal stamp)
Signed by the employer Signed by the employee
 ID No:………………….
at……………………… at………………………

WITNESS WITNESS
Name:………………….. Name:…………………..
ID No:................................ ID No:..............................
Contact No:……………….. Contact No:……………
Annexure A
The person in this position will be responsible to ……………………………. and undertake the
following tasks and responsibilities (should be countersigned by both the parties).
a. Job Responsibilities of _______________________
i.
ii.
iii.
iv.
v.
vi.
Signature of Employee Signature of Employer
'''

message = [
    {
        "role": "system",
        "content": "You are \'Secure Your Contract\' that is an AI based system that takes a contract as an input, detects possibly disadvantageous/dangerous keywords/phrases and provide the refinement suggestion/solution for the input.",
    },
    {"role": "user", "content": contract_text},
]
# message2 = [
#     {
#         "role": "system",
#         "content": "You are \'Secure Your Contract\' that is an AI based system that takes a contract as an input, detects possibly risky/possibly risky keywords/phrases and provide reasoans and the corresponding refinement suggestion/solution for the input.",
#     },

#     {"role": "user", "content": "This Consulting Agreement is made on October 15, 2022, by and between GreenTech Solutions Inc., located at 345 Innovation Blvd, San Francisco, CA 94105 (the \"Client\"), and James Lee, an independent consultant residing at 678 Business St, Seattle, WA 98101 (the \"Consultant\").\n\n**1. Scope of Services**\nThe Consultant agrees to provide environmental consulting services, including sustainability assessments, project management, and regulatory compliance advice.\n\n**2. Payment Terms**\nThe Client agrees to pay the Consultant a fee of $200 per hour. Invoices will be submitted monthly and are payable within 15 days of receipt.\n\n**3. Confidentiality**\nThe Consultant agrees to keep all Client information confidential and to only use it for the purpose of providing services outlined in this Agreement.\n\n**4. Intellectual Property**\nAll intellectual property developed by the Consultant during the term of this Agreement shall be the exclusive property of the Client.\n\n**5. Term and Termination**\nThis Agreement shall commence on October 15, 2022, and continue until the services are complete, unless terminated by either party with 15 days' written notice.\n\n**6. Liability Limitation**\nThe Consultant shall not be liable for any damages exceeding the total fees paid by the Client under this Agreement.\n\n**7. Governing Law**\nThis Agreement shall be governed by the laws of the State of California.\n\nSigned:\n\n_________________________\nGreenTech Solutions Inc.\n\n_________________________\nJames Lee\n\nDate: October 15, 2022\n"}
# ] # 2/2




        
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