from dotenv import load_dotenv
from time import time
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from prompts import rephrase, completion, named_entities, summarize, chat_prompt

load_dotenv()

model_name_or_path = "TheBloke/Yi-6B-200K-GPTQ"
# model_name_or_path = "TheBloke/deepseek-llm-7B-chat-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             cache_dir="./models/",
                                             # revision="gptq-8bit-32g-actorder_True")
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          trust_remote_code=True,
                                          cache_dir="./models/") #, use_fast=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=pipe)

#
# complete_chain = LLMChain(prompt=completion, llm=llm)
# rephrase_chain = LLMChain(prompt=rephrase, llm=llm)
# entity_chain = LLMChain(prompt=named_entities, llm=llm)
summarize_chain = LLMChain(prompt=chat_prompt, llm=llm)
#
# question = input("Q: ")
# i = 0
# while i < 4:
#     s = time()
#     n = complete_chain.run(in_var=question)
#     i += 1
#     question += n
#     print(i, time() - s)
#
# with open(f"BigAssStory.txt", "w") as f:
#     f.write(question)

system_message = "You are a chaotic good scientist called Dr. Chameleon. Hold a conversation with the human."

question_w_history = ""

while True:
    question = input("User: ")
    question_w_history += f"\nUser: {question}"
    if len(question_w_history) > 5000:
        question_w_history = question_w_history[-5000:]
    answer = summarize_chain.run(system_message=system_message, prompt=question_w_history)

    answer = answer.split("User")[0]

    question_w_history += f"\nAssistant: {answer}"
    print(f"Assistant: {answer}")


