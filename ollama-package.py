import ollama

# response = ollama.list()

#  chat example
# res = ollama.chat(
#     model="llama3.2",
#     messages=[{"role": "user", "content": "why is the sky"}],
#     stream=True,
# )


# for chunk in res:
#     print(chunk["message"]["content"], end="", flush=True)

# res = ollama.generate(model="llama3.2", prompt="Why is rick so smart")

# print(res)

modelFile = """
FROM llama3.2
SYSTEM You are jarvis the iron man(tony stark) ai assistant.
PARAMETER temperature 0.1
"""

ollama.create(model="jarvis", modelfile=modelFile)

res = ollama.generate(model="jarvis", prompt="What is your name?")
print(res)
