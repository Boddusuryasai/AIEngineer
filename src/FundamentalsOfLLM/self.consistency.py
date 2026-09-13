from openai import OpenAI
from dotenv import load_dotenv
import os 
import re
from collections import Counter
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

question = "A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?"
prompt = f"Q: {question}\nA: Let's think step by step."

# Generate multiple reasoning paths in one request
response = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,     # Increased for diverse reasoning
    n=5                  # Number of reasoning paths to generate
)

# Store all generated paths
reasoning_paths = [choice.message.content for choice in response.choices]


extracted_answers = []

for idx, path in enumerate(reasoning_paths):
    print(f"--- Path {idx + 1} ---\n{path}\n")
    
    # Simple extraction: grab the last number in the generated text
    numbers = re.findall(r'\d+', path)
    if numbers:
        final_number = numbers[-1]
        extracted_answers.append(final_number)

# Apply Majority Voting
if extracted_answers:
    vote_counts = Counter(extracted_answers)
    majority_answer, count = vote_counts.most_common(1)[0]
    
    print(f"Extracted Answers: {extracted_answers}")
    print(f"Final Majority Vote: {majority_answer} (voted {count} times)")
else:
    print("Could not extract numerical answers.")