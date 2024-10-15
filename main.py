from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Define the template for the chat
template = """
Answer the questions.

This is the conversation history: {context}

Question: {question}

Answer:
"""

# Initialize the language model
model = OllamaLLM(model="llama3.2")
# Create a prompt based on the template
prompt = ChatPromptTemplate.from_template(template)
# Combine the prompt and the model into a chain
chain = LLMChain(prompt=prompt, llm=model)

# Function to handle the conversation
def handle_conversation():
    context = ""
    print("Welcome to the Chat. Type 'nah' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "nah":
            print("Goodbye!")
            break
        
        # Generate the result using the chain
        result = chain.invoke({"context": context, "question": user_input})
        
        # Print the bot's response
        print("Bot:", result['text'])  # Assuming the result is a dictionary with 'text' key
        
        # Update the context with the latest conversation
        context += f"\nUser: {user_input}\nAI: {result['text']}"

# Entry point of the script
if __name__ == "__main__":
    handle_conversation()
