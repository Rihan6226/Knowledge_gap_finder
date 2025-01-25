import pdfplumber
from sentence_transformers import SentenceTransformer
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
import os
import re
from dotenv import load_dotenv


def process_user_responses(quiz_data, user_response):
    processed_data = {}
    questions = {q['id']: q for q in quiz_data.get('questions', [])}
    topic = quiz_data.get('topic', [])
    response_map = user_response.get('response_map', {})
    for question_id, selected_option_id in response_map.items():
        question = questions.get(int(question_id), {})
        options = question.get('options', [])
        correct_answer = next((opt['description'] for opt in options if opt['is_correct']), None)
        user_answer = next((opt['description'] for opt in options if opt['id'] == selected_option_id), None)
        processed_data[question_id] = {
            "Question_description": question.get('description', ""),
            "Answer": correct_answer,
            "user_answer": user_answer,
            "is_correct":user_answer == correct_answer
        }
    return processed_data, topic



def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text



def split_text_into_chunks(text, chunk_size=1000):
    # Split text into paragraphs (or you can split by sentences if preferred)
    paragraphs = text.split("\n")  
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < chunk_size:
            current_chunk += " " + paragraph
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())  # Add last chunk if exists
    
    return chunks


def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can try other models here
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings


def save_embeddings(embeddings, file_path):
    torch.save(embeddings, file_path)


def title_generate(text_content):
    load_dotenv()
    api_key = os.getenv("API_KEY")
    genai.configure(api_key=api_key)
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type = content.Type.OBJECT,
        properties = {
        "title": content.Schema(
            type = content.Type.STRING,
        ),
        },
    ),
    "response_mime_type": "application/json",
    }


    model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    )

    chat_session = model.start_chat(
    history=[
    ]
    )

    response = chat_session.send_message(
        f"""
        Provide a topic name for the NCERT content given h\n\n\n
        {text_content}
        """,
    )
    json_title = json.loads(response.text)
    title = json_title['title']
    return title


def questions_generate(text_content):
    load_dotenv()
    api_key = os.getenv("API_KEY")
    genai.configure(api_key=api_key)
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    )

    chat_session = model.start_chat(
    history=[
        {
        "role": "user",
        "parts": [
            "Find 3 questions with 4 options, based on the given content:\n\nachieved by hormones which are secreted by the efferentia endocrine glands. The prominent endocrine glands found in frog are pituitary, thyroid, Fat parathyroid, thymus, pineal body, pancreatic bodies islets, adrenals and gonads. The nervous system is organised into a central nervous system (brain Testis Kidney and spinal cord), a peripheral nervous system (cranial and spinal nerves) and an autonomic Adrenal nervous system (sympathetic and gland parasympathetic). There are ten pairs of cranial Urino nerves arising from the brain. Brain is enclosed genital duct in a bony structure called brain box (cranium). Rectum The brain is divided into fore-brain, mid-brain Cloaca and hind-brain. Forebrain includes olfactory lobes, paired cerebral hemispheres and unpaired Urinary Cloacal bladder aperture diencephalon. The midbrain is characterised by a pair of optic lobes. Hind-brain consists of cerebellum and medulla oblongata. The medulla Figure 7.3 Male reproductive system\n",
        ],
        },
        {
        "role": "model",
        "parts": [
            "```json\n[\n    {\n        \"question\": \"Which of the following is NOT a prominent endocrine gland found in frogs?\",\n        \"options\": [\n            \"Pituitary\",\n            \"Thyroid\",\n            \"Kidney\",\n            \"Adrenals\"\n        ],\n        \"answer\": \"Kidney\"\n    },\n    {\n        \"question\": \"The central nervous system of a frog is composed of:\",\n       \"options\": [\n            \"Brain, cranial nerves, and spinal nerves\",\n            \"Brain, spinal cord\",\n             \"Cranial nerves and spinal nerves\",\n            \"Sympathetic and parasympathetic systems\"\n        ],\n        \"answer\": \"Brain, spinal cord\"\n    },\n    {\n        \"question\": \"Which part of the frog's brain includes the olfactory lobes and cerebral hemispheres?\",\n        \"options\": [\n            \"Mid-brain\",\n             \"Forebrain\",\n            \"Hind-brain\",\n            \"Medulla oblongata\"\n        ],\n        \"answer\": \"Forebrain\"\n    }\n]\n```",
        ],
        },
    ]
    )

    response = chat_session.send_message("""
                                        Find 3 questions with 4 options, based on the given content:
                                        \n\nachieved by hormones which are secreted by the efferentia endocrine glands. The prominent endocrine glands found in frog are pituitary, thyroid, Fat parathyroid, thymus, pineal body, pancreatic bodies islets, adrenals and gonads. The nervous system is organised into a central nervous system (brain Testis Kidney and spinal cord), a peripheral nervous system (cranial and spinal nerves) and an autonomic Adrenal nervous system (sympathetic and gland parasympathetic). There are ten pairs of cranial Urino nerves arising from the brain. Brain is enclosed genital duct in a bony structure called brain box (cranium). Rectum The brain is divided into fore-brain, mid-brain Cloaca and hind-brain. Forebrain includes olfactory lobes, paired cerebral hemispheres and unpaired Urinary Cloacal bladder aperture diencephalon. The midbrain is characterised by a pair of optic lobes. Hind-brain consists of cerebellum and medulla oblongata. The medulla Figure 7.3 Male reproductive system\n
                                        """,)

    questions = json.loads(response.text)
    return questions



def get_pdf_path(folder_path, query):

    query = re.sub(r'[:;,]', '', query) 
    query = query.lower().strip() 
    
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf') and filename[:-4].lower() == query: 
            return os.path.join(folder_path, filename)
    return None



def gap_finder(quiz_details, response_data):
    user_responses, topic_fr_quiz = process_user_responses(quiz_details, response_data)
    
    
    pdf_path = get_pdf_path('Chapters',topic_fr_quiz)
    pdf_text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(pdf_text)
    textbook_embeddings = generate_embeddings(chunks)


    
    incorrect_responses = {k: v for k, v in user_responses.items() if not v['is_correct']}
    incorrect_findings = [f"{response['Question_description']}, {response['Answer']}" for response in incorrect_responses.values()]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    user_answer_embeddings = model.encode(incorrect_findings)

    if isinstance(textbook_embeddings, torch.Tensor):
        textbook_embeddings = textbook_embeddings.numpy()
    similarity_scores = cosine_similarity(user_answer_embeddings, textbook_embeddings)

    gaps = {}
    for i, user_answer in enumerate(incorrect_findings):
        similarities = similarity_scores[i]
        best_match_index = similarities.argmax()
        best_match_chunk = chunks[best_match_index]
        title = title_generate(best_match_chunk)
        questions = questions_generate(best_match_chunk)
        title_and_qns = {"title":title,
                         "more_questions":questions}
        gaps[f"topic {i+1}"]=title_and_qns

    return gaps





def pretty_print(data):
    print("User needs to improve in these topics:\n")
    topics = [data[key]['title'] for key in data]
    for idx, topic in enumerate(topics, start=1):
        print(f"{idx}. {topic}")

    print("\nTry these questions after diving deep into the suggested topics:\n")
    for key, topic_data in data.items():
        print(f"{topic_data['title']}:")
        for idx, question_data in enumerate(topic_data['more_questions'], start=1):
            question = question_data['question']
            options = question_data['options']
            answer = question_data['answer']
            
            print(f"{idx}. {question}")
            for option_idx, option in enumerate(options, start=1):
                print(f"    {chr(96 + option_idx)}. {option}")
            print(f"    Correct Answer: {answer}\n")
        