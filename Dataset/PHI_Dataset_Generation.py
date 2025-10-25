from pathlib import Path
import requests
from faker import Faker
from collections import OrderedDict
import json
from typing import List, Tuple, Dict, Any
import re
import os
import csv

SEED_FAKER = 42
SEED_LLM = 0

# Set seeds for reproducibility
fake = Faker('en_US')
Faker.seed(SEED_FAKER)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3"

def generate_with_prompt(prompt: str,
                         max_new_tokens: int = 512,
                         temperature: float = 0.8,) -> str:
    """Generate text using local Ollama API"""
    
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
        "<author_clinical_condition> - Any physical or mental health condition, diagnosis, or medical issue of the author\n"
        "<author_medical_report> - Medical reports, test results, or clinical outcomes related to the author\n"
        "<author_genetic> - Genetic information, DNA sequences, or hereditary traits of the author\n"
        "<author_fertility> - Information about fertility, pregnancy status, reproductive health of the author\n"
        "<author_disability> - Information about disabilities, limitations, or special needs of the author\n"
        "<author_addiction> - Information about dependencies, substance use, or addictive behaviors of the author\n"
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

def step4_1_replace_placeholders(text: str, substitutions_dict: dict) -> tuple[str, dict]:
    placeholders_dictionary = {
        "<author_clinical_condition>": ['Hypotension', 'Ischemic heart disease',
                                        'Myocardial infarction', 'Angina pectoris', 'Cardiac arrhythmias',
                                        'Atrial fibrillation', 'Tachycardia', 'Bradycardia',
                                        'Heart failure', 'Cardiomyopathy', 'Myocarditis',
                                        'Pericarditis', 'Endocarditis', 'Mitral valve disease',
                                        'Aortic stenosis', 'Aortic regurgitation', 'Aortic aneurysm',
                                        'Deep vein thrombosis', 'Pulmonary embolism', 'Varicose veins',
                                        'Peripheral arterial disease', 'Intermittent claudication',
                                        'Raynaud\'s phenomenon', 'Phlebitis', 'Hypercholesterolemia',
                                        'Dyslipidemia', 'Atherosclerosis', 'Bronchial asthma',
                                        'Chronic obstructive pulmonary disease', 'Pulmonary emphysema',
                                        'Chronic bronchitis', 'Acute bronchitis', 'Pneumonia',
                                        'Pleurisy', 'Pneumothorax', 'Pleural effusion',
                                        'Pulmonary fibrosis', 'Sarcoidosis', 'Sleep apnea',
                                        'Obstructive sleep apnea syndrome', 'Allergic rhinitis',
                                        'Sinusitis', 'Pharyngitis', 'Laryngitis', 'Tonsillitis',
                                        'Bronchiectasis',
                                        'Pulmonary tuberculosis', 'Silicosis', 'Asbestosis',
                                        'Gastritis', 'Gastric ulcer', 'Duodenal ulcer',
                                        'Gastroesophageal reflux disease', 'Esophagitis',
                                        'Irritable bowel syndrome', 'Crohn\'s disease',
                                        'Ulcerative colitis', 'Diverticulitis', 'Diverticulosis',
                                        'Hemorrhoids', 'Anal fissures', 'Chronic constipation',
                                        'Chronic diarrhea', 'Malabsorption', 'Celiac disease',
                                        'Lactose intolerance', 'Gastroenteritis', 'Pancreatitis',
                                        'Gallstones', 'Cholecystitis', 'Hepatitis A', 'Hepatitis B',
                                        'Hepatitis C', 'Liver cirrhosis', 'Fatty liver disease',
                                        'Intestinal polyposis',
                                        'Type 1 diabetes mellitus', 'Type 2 diabetes mellitus',
                                        'Gestational diabetes', 'Hypoglycemia', 'Hyperglycemia',
                                        'Hypothyroidism', 'Hyperthyroidism', 'Hashimoto\'s thyroiditis',
                                        'Goiter', 'Thyroid nodules', 'Graves\' disease',
                                        'Adrenal insufficiency', 'Cushing\'s syndrome',
                                        'Addison\'s disease', 'Hyperaldosteronism', 'Pheochromocytoma',
                                        'Hypoparathyroidism', 'Hyperparathyroidism', 'Osteoporosis',
                                        'Osteomalacia', 'Polycystic ovary syndrome',
                                        'Hypogonadism', 'Hypergonadism', 'Acromegaly',
                                        'Pituitary dwarfism', 'Diabetes insipidus', 'Migraine',
                                        'Tension headache', 'Cluster headache',
                                        'Epilepsy', 'Parkinson\'s disease', 'Essential tremor',
                                        'Multiple sclerosis', 'Amyotrophic lateral sclerosis',
                                        'Alzheimer\'s disease', 'Vascular dementia', 'Senile dementia',
                                        'Ischemic stroke', 'Hemorrhagic stroke', 'Transient ischemic attack',
                                        'Diabetic neuropathy', 'Peripheral neuropathy', 'Trigeminal neuralgia',
                                        'Facial nerve palsy', 'Carpal tunnel syndrome',
                                        'Herniated disc', 'Cervical spondylosis', 'Low back pain',
                                        'Sciatica', 'Meningitis', 'Encephalitis', 'Myasthenia gravis',
                                        'Rheumatoid arthritis', 'Osteoarthritis', 'Psoriatic arthritis',
                                        'Ankylosing spondylitis', 'Gout', 'Pseudogout',
                                        'Fibromyalgia', 'Polymyalgia rheumatica', 'Systemic lupus erythematosus',
                                        'Sjögren\'s syndrome', 'Scleroderma', 'Dermatomyositis',
                                        'Polymyositis', 'Bursitis', 'Tendinitis', 'Epicondylitis',
                                        'Rotator cuff syndrome', 'Sprain',
                                        'Contusion', 'Fracture', 'Dislocation', 'Muscle strain',
                                        'Muscle cramps', 'Myositis', 'Scoliosis',
                                        'Kyphosis', 'Lordosis', 'Cystitis',
                                        'Urethritis', 'Pyelonephritis', 'Glomerulonephritis',
                                        'Chronic kidney disease', 'Acute kidney injury',
                                        'Kidney stones', 'Benign prostatic hyperplasia',
                                        'Prostatitis', 'Prostate cancer', 'Erectile dysfunction',
                                        'Urinary incontinence', 'Urinary retention', 'Hematuria',
                                        'Proteinuria', 'Nephrotic syndrome', 'Nephritic syndrome',
                                        'Polycystic kidney disease', 'Endometriosis', 'Uterine fibroids',
                                        'Ovarian cysts', 'Vaginitis', 'Cervicitis', 'Dysmenorrhea',
                                        'Amenorrhea',
                                        'Atopic dermatitis', 'Contact dermatitis', 'Seborrheic dermatitis',
                                        'Psoriasis', 'Eczema', 'Hives', 'Acne', 'Rosacea',
                                        'Melasma', 'Actinic keratosis', 'Herpes simplex',
                                        'Herpes zoster', 'Chickenpox', 'Molluscum contagiosum',
                                        'Warts', 'Cutaneous mycoses', 'Cutaneous candidiasis',
                                        'Impetigo', 'Cellulitis', 'Folliculitis', 'Alopecia',
                                        'Hirsutism', 'Melanoma', 'Basal cell carcinoma',
                                        'Squamous cell carcinoma', 'Keloids', 'Hypertrophic scars',
                                        'Iron deficiency anemia', 'Megaloblastic anemia',
                                        'Hemolytic anemia', 'Aplastic anemia', 'Sickle cell anemia',
                                        'Thalassemia', 'Acute leukemia', 'Chronic leukemia',
                                        'Hodgkin\'s lymphoma', 'Non-Hodgkin\'s lymphoma', 'Multiple myeloma',
                                        'Thrombocytopenia', 'Thrombocytosis', 'Immune thrombocytopenic purpura',
                                        'Hemophilia', 'von Willebrand disease',
                                        'Disseminated intravascular coagulation',
                                        'Polycythemia vera', 'Myelofibrosis', 'Myelodysplastic syndrome',
                                        'Food allergies', 'Respiratory allergies', 'Drug allergies',
                                        'Anaphylaxis', 'Primary immunodeficiency', 'Secondary immunodeficiency',
                                        'Autoimmune diseases', 'Vasculitis', 'Antiphospholipid syndrome',
                                        'Behçet\'s disease', 'Acquired immunodeficiency syndrome',
                                        'Major depressive disorder', 'Bipolar disorder',
                                        'Generalized anxiety disorder', 'Panic disorder',
                                        'Obsessive-compulsive disorder', 'Post-traumatic stress disorder',
                                        'Social anxiety disorder', 'Agoraphobia', 'Specific phobias',
                                        'Borderline personality disorder', 'Narcissistic personality disorder',
                                        'Antisocial personality disorder', 'Schizophrenia',
                                        'Schizoaffective disorder', 'Delusional disorder',
                                        'Attention-deficit/hyperactivity disorder',
                                        'Autism spectrum disorders', 'Sleep disorders',
                                        'Insomnia', 'Eating disorders', 'Anorexia nervosa',
                                        'Bulimia nervosa', 'Binge-eating disorder', 'Bronchiolitis',
                                        'Acute laryngitis', 'Otitis media', 'Pharyngotonsillitis',
                                        'Acute gastroenteritis', 'Pediatric asthma',
                                        'Pediatric gastroesophageal reflux', 'Infantile colic',
                                        'Functional constipation', 'Nocturnal enuresis',
                                        'Recurrent fever', 'Febrile seizures', 'Infantile epilepsy',
                                        'Neurodevelopmental disorders', 'Psychomotor retardation',
                                        'Learning disorders', 'Behavioral disorders',
                                        'Fragility fractures', 'Sarcopenia', 'Cachexia',
                                        'Malnutrition in the elderly', 'Immobilization syndrome',
                                        'Pressure ulcers', 'Polypharmacy', 'Delirium'],
        "<author_medical_report>": ['Blood test results', 'Urine analysis', 'X-ray report', 'MRI scan',
                                    'CT scan', 'Ultrasound report', 'ECG report', 'Echocardiogram',
                                    'Biopsy results', 'Pathology report', 'Lab culture results',
                                    'Genetic test results', 'Allergy test panel', 'Hormone levels',
                                    'Cholesterol panel', 'Liver function tests', 'Kidney function tests',
                                    'Thyroid function tests', 'Cardiac stress test', 'Pulmonary function test',
                                    'Bone density scan', 'Colonoscopy report', 'Endoscopy report',
                                    'Mammography report', 'Pap smear results', 'PSA test results'],
        "<author_genetic>": ['BRCA1 mutation', 'BRCA2 mutation', 'APOE e4 allele',
                             'Factor V Leiden', 'Prothrombin mutation', 'MTHFR mutation',
                             'HLA-B27 positive', 'Tay-Sachs carrier', 'Sickle cell trait',
                             'Hemochromatosis mutation', 'Cystic fibrosis carrier',
                             'Huntington\'s disease gene', 'Lynch syndrome', 'Familial hypercholesterolemia'
                             ],
        "<author_fertility>": ['Infertility', 'Polycystic ovary syndrome', 'Endometriosis',
                               'Male factor infertility', 'Ovulation disorders', 'Tubal factor infertility',
                               'Unexplained infertility', 'Recurrent pregnancy loss',
                               'Premature ovarian failure', 'Varicocele', 'Azoospermia'],
        "<author_disability>": ['Visual impairment', 'Hearing impairment', 'Mobility impairment',
                                'Cognitive impairment', 'Speech impairment', 'Learning disability',
                                'Autism spectrum disorder', 'Intellectual disability',
                                'Chronic fatigue syndrome', 'Fibromyalgia', 'Multiple sclerosis',
                                'Spinal cord injury', 'Traumatic brain injury'],
        "<author_addiction>": ['Alcohol dependency', 'Nicotine addiction', 'Opioid dependency',
                               'Cocaine addiction', 'Cannabis dependency', 'Benzodiazepine dependency',
                               'Gambling addiction', 'Internet addiction', 'Food addiction',
                               'Prescription drug abuse', 'Stimulant abuse']
    }

    list_placeholders = detect_placeholders(text)


    unique_keys = list(OrderedDict.fromkeys(list_placeholders))
    filtered_placeholders_dictionary = {
        key: placeholders_dictionary[key]
        for key in unique_keys
        if key in placeholders_dictionary
    }

    placeholders_section = ""
    if filtered_placeholders_dictionary:
        placeholders_section = "For the following placeholders found in the text, replace each one with ONE appropriate value from the corresponding list:\n"
        for placeholder, values in filtered_placeholders_dictionary.items():
            placeholders_section += f"{placeholder} - Choose from: {', '.join(values)}\n"
        placeholders_section += "\n"

    prompt = (
        f"Text: {text}\n"
        f"{placeholders_section}"
        "Requirements:\n"
        "- You don't have to replace other parts of speech, you only have to replace the placeholders.\n"
        "- Return ONLY a valid JSON object with this exact structure:\n"
        '{\n'
        '  "Text": "the text with all placeholders replaced",\n'
        '  "Substitutions": {\n'
        '    "<placeholder_name>": "replacement_value"\n'
        '  }\n'
        '}\n'
        "- Do not include any other text or explanation, just the JSON.\n"
        "- Respond only with the exact JSON object, without any introductory sentence or explanation\n"
        "- Begin the response exactly with the { character."

    )

    try:
        llm_output = generate_with_prompt(prompt)
        parsed_output = json.loads(llm_output)
        final_text = parsed_output["Text"]
        llm_substitutions = parsed_output["Substitutions"]

        substitutions_dict.update(llm_substitutions)

        return final_text, substitutions_dict

    except json.JSONDecodeError as e:
        print(f"[Error] Failed to parse LLM output as JSON: {e}")
        #print(f"[Error] Raw output was: {llm_output}")
        #return text, substitutions_dict
        return "-1", {1:"-1"}
    except KeyError as e:
        print(f"[Error] Missing expected key in LLM output: {e}")
        #print(f"[Error] Parsed output was: {parsed_output}")
        return "-1", {1:"-1"}

if __name__ == "__main__":

    #aggregated_records = []
    i = 0
    while i < 60000:
        try:
            SEED_LLM = SEED_LLM+1
            print(f"Creazione riga {i+1}")
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
            final_text, final_substitutions_dict = step4_1_replace_placeholders(final_text_1, substitutions_dict)
            if final_text == "-1" and final_substitutions_dict == {1:"-1"}:
                continue

            file_exists = os.path.isfile('dataset_with_text_seed.csv')
            with open('dataset_with_text_seed.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists or os.stat('dataset_with_text_seed.csv').st_size == 0:
                    writer.writerow(['final_text', 'final_substitutions_dict'])
                writer.writerow([final_text, final_substitutions_dict])
            i = i+1

        except json.JSONDecodeError as e:
            print(f"[Warning] JSON decoding failed on iteration of row {i+1}, retrying... ({e})")
            continue
        except Exception as e:
            print(f"Generic error occured on iteration of row {i+1}, retrying... ({e})")
            continue
