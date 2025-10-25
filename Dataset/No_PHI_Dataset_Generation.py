# import random
from pathlib import Path
import requests
from faker import Faker
from collections import OrderedDict
import json
from typing import List, Tuple, Dict, Any
import re
import os
import csv

SEED = 48
SEED_LLM = 0

# Set seeds for reproducibility
fake = Faker('en_US')
Faker.seed(SEED)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3"

def generate_with_prompt(prompt: str,
                         max_new_tokens: int = 512,
                         temperature: float = 0.8, ) -> str:
    """Generate text using local Ollama API"""

    print(f"Temperatura: {temperature} \t SEED_LLM: {SEED_LLM}")

    url = f"{OLLAMA_BASE_URL}/api/generate"
    # Prepare the payload
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_new_tokens,
            "seed": SEED_LLM,
        }
    }

    try:
        # Send POST request to Ollama
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        # Parse the JSON response
        result = response.json()
        return result.get("response", "Error: No response generated")
    except:
        return "Error: Failed to generate response"


def detect_placeholders(text):
    all_placeholders = [
        "<author_clinical_condition>",
        "<author_medical_report>",
        "<author_genetic>",
        "<author_fertility>",
        "<author_disability>",
        "<author_addiction>"
    ]

    placeholders_list = []

    for placeholder in all_placeholders:
        if placeholder in text:
            placeholders_list.append(placeholder)

    return placeholders_list


def step1_generate_bio():
    name = fake.name()
    bio_prompt = (
        f"Generate a biography of a fictional man named {name}.\n"
        "Occupation: any average job you can come up with\n"
        "Personality: describe in 5 sentences\n"
        "Present results in json format with fields \"occupation\": str, \"personality\": str\n"
        "Print only the json, without anything else"
    )
    bio_output = generate_with_prompt(bio_prompt)
    bio = json.loads(bio_output)

    return name, bio


def step2_generate_clean_text(occupation: str,
                              personality: str,
                              domain_instructions: str) -> str:
    text_prompt = (
        "Step 1) Look through the personality of the text author "
        "and pretend to be that person.\n"
        f"occupation: {occupation}\n"
        f"personality: {personality}\n\n"
        "Step 2) Use the following instructions to generate a text:\n"
        f"{domain_instructions}\n\n"
        "Requirements:\n"
        "- At any circumstance do not include any personal information in generated text.\n"
        "- Also do not put any sympomts or clinical conditions in the text\n"
        "- Respond only with generated text with no commentary. Here goes your text:"
    )
    clean_text = generate_with_prompt(text_prompt)
    return clean_text


def step3_add_placeholders(text: str) -> str:
    prompt = (
        f"Text: {text}\n"
        "Task: You are an author of the above Text. Your task is to add new placeholders in the Text from the list below. "
        "You will be penalized for mentioning any placeholders other than what is listed below!\n"
        "Here is the list of placeholders representing your personal information:\n"
        "<author_personal_name> - A full or partial name of the text author\n"
        "<author_personal_email> - An author’s email address\n"
        "<author_personal_username> - An author’s username on any website, social media etc.\n"
        "<author_personal_phone_number> - A phone number associated with the author or his relatives\n"
        "<author_personal_url> - A link to author’s social media page or personal website\n"
        "<author_personal_address> - A full or partial street address that is associated with the author, such as home address\n"
        "<author_personal_identifier> - A number or sequence of characters that could be used to identify an author, such as a SSN or policy number\n"
        "Requirements:\n"
        "- Do NOT change existing placeholders\n"
        "- Distribute placeholders evenly throughout your text, do not stack them all in one place\n"
        "- New text must be more entity-dense than the previous one\n"
        "- Placeholders should not be inserted arbitrarily but must be coherent with the text.\n"
        "- Respond only with updated text with no commentary."
    )
    updated_text = generate_with_prompt(prompt)
    return updated_text


def step4_replace_placeholders(text: str) -> tuple[str | Any, dict[str, Any]]:
    replacements = {
        "<author_personal_name>": fake.name,
        "<author_personal_email>": fake.email,
        "<author_personal_username>": fake.user_name,
        "<author_personal_phone_number>": fake.phone_number,
        "<author_personal_url>": fake.url,
        "<author_personal_address>": fake.address,
        "<author_personal_identifier>": fake.ssn,
    }

    substitutions_dict = {}

    for placeholder, func in replacements.items():
        while placeholder in text:
            replacement_value = func()
            text = text.replace(placeholder, replacement_value)
            substitutions_dict[placeholder] = replacement_value

    return text, substitutions_dict


def step4_1_replace_placeholders(text: str) -> str | tuple[str, str]:

    prompt = (
        f"Text: {text}\n"
        "Requirements:\n"
        "- You don't have to replace other parts of speech.\n"
        "- Return ONLY text without any commentary. only put in output the message from an author to a Doctor:\n"
        "- Do not include any other text or explanation, just the text.\n"
        "- Respond only with the exact text, without any introductory sentence or explanation\n"
    )

    try:
        llm_output = generate_with_prompt(prompt)
        return llm_output

    except json.JSONDecodeError as error:
        print(f"[Error] Failed to parse LLM output as JSON: {error}")
        # print(f"[Error] Raw output was: {llm_output}")
        # return text, substitutions_dict
        return "-1", "-1"
    except KeyError as error:
        print(f"[Error] Missing expected key in LLM output: {error}")
        # print(f"[Error] Parsed output was: {parsed_output}")
        return "-1", "-1"

if __name__ == "__main__":

    i = 0
    while i < 60000:
        try:
            SEED_LLM = SEED_LLM + 1
            print(f"Creazione riga {i + 1}")
            name, bio = step1_generate_bio()
            domain_instructions = (
                "Write a medical consultation question asking for advice to a doctor"
            )
            clean_text = step2_generate_clean_text(
                occupation=bio["occupation"],
                personality=bio["personality"],
                domain_instructions=domain_instructions
            )

            text_with_placeholders = step3_add_placeholders(clean_text)
            final_text_1, substitutions_dict = step4_replace_placeholders(text_with_placeholders)
            final_text, final_substitutions_dict = step4_1_replace_placeholders(final_text_1)
            if final_text == "-1" and final_substitutions_dict == "-1":
                continue
                
            file_exists = os.path.isfile('dataset_no_pii_seed_48_con_4_1_modificato.csv')
            with open('dataset_no_pii_seed_48_con_4_1_modificato.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists or os.stat('dataset_no_pii_seed_48_con_4_1_modificato.csv').st_size == 0:
                    writer.writerow(['final_text', 'final_substitutions_dict'])
                writer.writerow([final_text, final_substitutions_dict])
            i = i + 1

        except json.JSONDecodeError as e:
            print(f"[Warning] JSON decoding failed on iteration of row {i+1}, retrying... ({e})")
            continue
        except Exception as e:
            print(f"Generic error occured on iteration of row {i+1}, retrying... ({e})")
            continue

