import ollama
import os

model = "llama3.2"

input_file = "./data/grocery-list.txt"
output_file = "./data/categorized-grocery-list.txt"

# Check if the input file exist
if not os.path.exists(input_file):
    print(f"Input file '{input_file}' not found.")
    exit(1)

# Read the uncategorised grocery items from the input file
with open(input_file, "r") as f:
    items = f.read().strip()

# Prepare the prompt for the model
prompt = f"""
You are an assitant that categorizes and sorts grocery items

Here is the list of grocery items:
{items}

Please:

1. Categorize these items into appropriate categorues such as Produce, Meat, Bakery, Beverage, etc.
2. Sort the items alphabetically within each category.
3. Present the categorised list in a clear and organised manner, using bullet points or numbering.
"""

try:
    response = ollama.generate(model=model, prompt=prompt)
    generated_text = response.get("response", "")
    print("=== Categorised List: === \n")
    print(generated_text)

    with open(output_file, "w") as f:
        f.write(generated_text.strip())

    print(f"Categorised grocery list has been saved to '{output_file}'.")
except Exception as e:
    print("An error occured:", str(e))
