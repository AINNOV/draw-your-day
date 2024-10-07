from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

ollama = Ollama(
    base_url='http://localhost:11434',
    model='llama3:8b'
)

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)

examples = [
    {
        "input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?",
        "output": "질소입니다."
    },
    {
        "input": "광합성에 필요한 주요 요소들은 무엇인가요?",
        "output": "빛, 이산화탄소, 물입니다."
    }
]

fewshot_prompt = FewShotChatMessagePromptTemplate(
    examples = examples,
    example_prompt = example_prompt,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 과학에 대해 잘 아는 교육자입니다."),
        fewshot_prompt,
        ("human", "{input}"),
    ]
)

chain = prompt | ollama

while 1:
    print("Write your message (write 'bye' to get out):")
    content = input()
    if content == "bye": 
        break
    
    # messages = [
    #     {
    #         'role': 'user',
    #         'content': content,
    #     }]

    response = chain.invoke({"input" : content})

    print(response)