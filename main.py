from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model=OllamaLLM(model="llama3.2")

# Template for the prompt
template = """
you are and expert in answering questions about a pizza restaurant

here are some relevant reviews : {reviews}

here is the question to answer : {question}
"""

prompt=ChatPromptTemplate.from_template(template)  # Create prompt object
chain=prompt|model # Create a pipeline: the prompt formats the question and sends it to the LLM

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    reviews = retriever.invoke(question)# Get relevant reviews
    result = chain.invoke({"reviews": reviews, "question": question})# Get LLM answer
    print(result)