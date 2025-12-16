import pandas as pd
import google.generativeai as genai

genai.configure(api_key="AIzaSyDM7R2lwIkxpoN-LufRrgqE5T5BaVv5pE4") 

csv_path = "cs_students.csv" 
df = pd.read_csv(csv_path)

print(" Dataset loaded successfully!")
print(f" Columns: {list(df.columns)}")
print(f" Total rows: {len(df)}")

data_snippet = df.head(100).to_csv(index=False)

model = genai.GenerativeModel("models/gemini-2.5-flash")
print(df.head(25))

print("\nAsk questions about the dataset (type 'exit' to quit):\n")

while True:
    question = input(">> ")

    if question.lower() == "exit":
        break

    # Prompt for Gemini
    prompt = f"""
You are a helpful data assistant. The user will ask questions about a dataset.
Here is the dataset (first 100 rows):

{data_snippet}

Now answer this question clearly and completely:

{question}
"""

    try:
        response = model.generate_content(prompt)
        print(f"\n Answer: {response.text.strip()}\n")
    except Exception as e:
        print(f"Error: {e}\n")
