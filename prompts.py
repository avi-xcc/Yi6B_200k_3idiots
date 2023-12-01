from langchain.prompts import PromptTemplate

# template = """You are an expert travel planner. The \
# following is a request by a person wanting to travel, \
# create a custom travel itinerary for them.
#
# Travelling from: {source_city}
# Travelling to: {destination}
# Interests: {interests}
# Number of days planned to stay: {num_days}
#
# Travel itinerary recommended by you:
# """
#
# prompt = PromptTemplate(template=template,
#                         input_variables=["source_city",
#                                          "destination",
#                                          "interests",
#                                          "num_days"])

# template = """You're Dr. Chameleon, a chaotic good scientist. Have a \
# conversation with a human.
#
# Human: {question}
# Dr. Chameleon: """
#
# prompt = PromptTemplate(template=template,
#                         input_variables=["question"])


rephrase = PromptTemplate(template="""Rephrase this: {in_var}""",
                             input_variables=["in_var"])


named_entities = PromptTemplate(template="""Find the named entities in the following: \
{in_var}

Named Entities:""",
                                input_variables=["in_var"])

summarize = PromptTemplate(template="""Summarize the following: \
{in_var}""",
                          input_variables=["in_var"])

completion = PromptTemplate(template="""{in_var}""",
                            input_variables=["in_var"])

chat_prompt = PromptTemplate(template="""{system_message}
{prompt}

Assistant:""", input_variables=["system_message", "prompt"])
