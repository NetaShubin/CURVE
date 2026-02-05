import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct"
).to(device)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

df = pd.read_csv("/home1/nshubin/ondemand/data/sys/myjobs/projects/default/3/ipip-300.csv")

BATCH_SIZE = 16
results = []

def extract_number(text):
    match = re.search(r"[1-5]", text)
    return int(match.group()) if match else None

def infer_batch(batch_prompts, tokenizer, model, device):
    """
    Process a batch of prompts using direct model inference for better efficiency.
    This replaces the pipeline approach with direct tokenizer/model calls.
    """
    # Tokenize all prompts in the batch with padding
    inputs = tokenizer(
        batch_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=2048  # Adjust based on your prompt length
    ).to(device)

    with torch.no_grad():
        # Generate responses with sampling for diversity
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

    # Decode only the new tokens (not the input)
    responses = []
    for i, output in enumerate(outputs):
        # Skip the input tokens and decode only the generated part
        input_length = inputs['input_ids'][i].shape[0]
        generated_tokens = output[input_length:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        responses.append(response_text)

    return responses

# -------- persona generation --------
# Make all personas
age_groups = {
    "young adult": "<35",
    "middle aged": "35-60",
    "senior": "60+"
}
genders = ["male", "female"]
education_levels = ["high school", "college", "graduate degree"]

regions_by_country = {
    "USA": [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
        "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
        "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
        "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
        "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
        "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
        "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
    ],
    "Japan": [
        "Hokkaido", "Aomori", "Iwate", "Miyagi", "Akita", "Yamagata", "Fukushima", "Ibaraki",
        "Tochigi", "Gunma", "Saitama", "Chiba", "Tokyo", "Kanagawa", "Niigata", "Toyama",
        "Ishikawa", "Fukui", "Yamanashi", "Nagano", "Gifu", "Shizuoka", "Aichi", "Mie",
        "Shiga", "Kyoto", "Osaka", "Hyogo", "Nara", "Wakayama", "Wakayama", "Tottori",
        "Shimane", "Okayama", "Hiroshima", "Yamaguchi", "Tokushima", "Kagawa", "Ehime",
        "Kochi", "Fukuoka", "Saga", "Nagasaki", "Kumamoto", "Oita", "Miyazaki", "Kagoshima",
        "Okinawa"
    ],
    "India": [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
        "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
        "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
        "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
        "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh",
        "Lakshadweep", "Puducherry"
    ],
    "Brazil": [
        "Acre", "Alagoas", "Amapá", "Amazonas", "Bahia", "Ceará", "Distrito Federal", "Espírito Santo",
        "Goiás", "Maranhão", "Mato Grosso", "Mato Grosso do Sul", "Minas Gerais", "Pará", "Paraíba",
        "Paraná", "Pernambuco", "Piauí", "Rio de Janeiro", "Rio Grande do Norte", "Rio Grande do Sul",
        "Rondônia", "Roraima", "Santa Catarina", "São Paulo", "Sergipe", "Tocantins"
    ],
    "Saudi Arabia": [
        "Riyadh", "Makkah", "Madinah", "Eastern Province", "Asir", "Tabuk", "Hail", "Northern Borders",
        "Jazan", "Najran", "Al-Bahah", "Al-Jawf", "Al-Qassim"
    ],
    "South Africa": [
        "Eastern Cape", "Free State", "Gauteng", "KwaZulu-Natal", "Limpopo", "Mpumalanga",
        "Northern Cape", "North West", "Western Cape"
    ]
}

persona_prompt_template = "You are a {age} {gender} from {state_region}, {country} with a {education} degree."

all_persona_prompts = []

for age_label, age_range in age_groups.items():
    for gender in genders:
        for education in education_levels:
            for country, regions in regions_by_country.items():
                for region in regions:
                    # Construct the persona prompt
                    prompt = persona_prompt_template.format(
                        age=age_label, # Using the descriptive label like 'young adult'
                        gender=gender,
                        state_region=region,
                        country=country,
                        education=education
                    )
                    all_persona_prompts.append(prompt)

print(all_persona_prompts)

# ================= MAIN LOOP =================
for persona in all_persona_prompts:
    print(f"\n🧑 Persona: {persona}")

    prompts = []
    traits = []
    reverses = []

    # build all prompts ONCE per persona
    for _, row in df.iterrows():
        question = row["Text"]
        trait = row["Key"][0]
        reverse = row.get("Reverse", "F")

        prompt = (
            f"{persona} I {question}. "
            "Rate from 1 to 5. Respond with ONE number only."
        )

        print(prompt)

        prompts.append(prompt)
        traits.append(trait)
        reverses.append(reverse)

    # -------- batch inference --------
    for start in range(0, len(prompts), BATCH_SIZE):

        batch_prompts = prompts[start:start + BATCH_SIZE]

        responses = infer_batch(batch_prompts, tokenizer, model, device)

        # -------- parse results --------
        for i, resp in enumerate(responses):
            idx = start + i
            rating = extract_number(resp)

            if rating is None:
                continue

            # correct reverse scoring (1↔5)
            if reverses[idx].upper() == "R":
                rating = 6 - rating

            results.append({
                "Persona": persona,
                "Prompt": batch_prompts[i],
                "Trait": traits[idx],
                "Rating": rating
            })

    torch.cuda.empty_cache()

# ================= SAVE RESULTS =================
scores_df = pd.DataFrame(results)
scores_df.to_csv("ipip_model_responses_batching.csv", index=False)

trait_means = (
    scores_df
    .groupby(["Persona", "Trait"])["Rating"]
    .mean()
    .reset_index()
)

trait_means.to_csv("ipip_model_means_batching.csv", index=False)

print("✅ Finished successfully")
