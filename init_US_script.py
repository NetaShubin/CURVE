# --- Limit OpenBLAS threads BEFORE any imports ---
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"   # reduce number of threads
os.environ["OMP_NUM_THREADS"] = "4"        # optional, sometimes needed for PyTorch/Transformers

# --- Now import libraries safely ---
# Load model directly
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Load the CSV
df = pd.read_csv("/home1/nshubin/ondemand/data/sys/myjobs/projects/default/3/ipip-300.csv")

# Store results
results = []

def extract_number(text):
        # Extract the first number 1–5 from model output and convert to int
        match = re.search(r"[1-5]", text)
        return int(match.group()) if match else None

for i, row in df.iterrows():
        question = row["Text"]
        trait = row["Key"][0]   # First letter (O, C, E, A, N)
        reverse = row.get("Reverse", "F")  # Default forward if missing

        prompt = f"You are someone from the United States. I {question}. Please rate from a scale of 1 to 5 how closely you align with this statement."

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=20)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        rating = extract_number(response)

        if rating is None:
                print(f"⚠️ Skipping Q{i+1} (no valid rating): {response}")
                continue

        # Reverse score if needed: |rating - 5|
        if reverse.upper() == "R":
                rating = abs(rating - 5)

        results.append({"Trait": trait, "Rating": rating})
        print(rating)

# Convert to DataFrame
scores_df = pd.DataFrame(results)

# Compute average per trait
trait_means = scores_df.groupby("Trait")["Rating"].mean().to_dict()

print("\nAverage rating per Big Five trait:")
for t, avg in trait_means.items():
    name = {
        "O": "Openness",
        "C": "Conscientiousness",
        "E": "Extraversion",
        "A": "Agreeableness",
        "N": "Neuroticism"
    }.get(t, t)
    print(f"{name:17s}: {avg:.2f}")

# Optionally save all raw data
scores_df.to_csv("ipip_model_responses.csv", index=False)
