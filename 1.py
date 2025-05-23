from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


import nltk
import math
import requests
from textblob import TextBlob
from collections import Counter
from openai import OpenAI
from mistralai import Mistral

nltk.download("punkt")

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY")

# Initialize clients
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://tfy.ml-apps-dev.siemens-healthineers.com/api/llm/api/inference/openai"
)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
mistral_model = "mistral-large-latest"

def google_fact_check(text):
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={text[:100]}&key={GOOGLE_FACTCHECK_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        claims = data.get("claims", [])
        if not claims:
            return ["No fact-checks found."]
        summaries = [c['text'] + " -> " + c['claimReview'][0]['textualRating'] for c in claims if 'claimReview' in c]
        return summaries[:3]
    except Exception as e:
        return [f"Fact check error: {str(e)}"]

def cosine_similarity_vec(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def get_similarity(text1, text2):
    try:
        embed_response = openai_client.embeddings.create(
            input=[text1, text2],
            model="siemens-azureopenai/embedding-3-large",
            extra_headers={
                "X-TFY-METADATA": '{"tfy_log_request":"true"}'
            }
        )
        vec1 = embed_response.data[0].embedding
        vec2 = embed_response.data[1].embedding
        return cosine_similarity_vec(vec1, vec2)
    except Exception as e:
        print(f"Embedding similarity error: {e}")
        return 0.0

def evaluate_relevance_and_coherence(text, prompt):
    relevance_score = get_similarity(text, prompt)
    sentences = nltk.sent_tokenize(text)
    coherence_score = len(sentences) / (len(text.split()) + 1e-5)
    return relevance_score, coherence_score

def evaluate_tone_and_style(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def evaluate_intent_alignment(text, prompt):
    text_tokens = set(nltk.word_tokenize(text.lower()))
    prompt_tokens = set(nltk.word_tokenize(prompt.lower()))
    return len(text_tokens & prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0

def call_openai(prompt):
    try:
        stream = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI bot."},
                {"role": "user", "content": prompt},
            ],
            model="siemens-azureopenai/gpt-4o",
            stream=True,
            temperature=0.7,
            max_tokens=256,
            top_p=0.8,
            stop=["</s>"],
            extra_headers={
                "X-TFY-METADATA": '{"tfy_log_request":"true"}'
            }
        )

        response_text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        return response_text.strip()

    except Exception as e:
        print(f"Error calling Siemens OpenAI: {e}")
        return ""

def call_mistral(prompt):
    try:
        response = mistral_client.chat.complete(
            model=mistral_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Mistral error: {e}")
        return ""

def llm_as_judge(prompt, openai_output, mistral_output):
    judge_prompt = f"""
You are an impartial judge. Two AI assistants gave different responses to the same user prompt.

Prompt:
{prompt}

Response from OpenAI:
{openai_output}

Response from Mistral:
{mistral_output}

Evaluate both answers on factual correctness, relevance, coherence, tone, and alignment with the user's intent. Then suggest improvements if needed and output a better version.

Format:
1. Evaluation Summary
2. Suggested Improvements
3. Combined/Better Answer
"""
    return call_openai(judge_prompt)

def evaluate_output(model_name, output, prompt):
    print(f"\n--- {model_name} Output ---\n{output}")
    print("\nEvaluation:")
    facts = google_fact_check(output)
    print(f"Factual Accuracy:\n{facts}")
    rel, coh = evaluate_relevance_and_coherence(output, prompt)
    print(f"Relevance: {rel:.2f}, Coherence: {coh:.2f}")
    tone, style = evaluate_tone_and_style(output)
    print(f"Tone (polarity): {tone:.2f}, Style (subjectivity): {style:.2f}")
    intent = evaluate_intent_alignment(output, prompt)
    print(f"Intent Alignment: {intent:.2f}")

def evaluate_llm_outputs(prompt):
    openai_output = call_openai(prompt)
    mistral_output = call_mistral(prompt)

    evaluate_output("OpenAI", openai_output, prompt)
    evaluate_output("Mistral", mistral_output, prompt)

    print("\n========== LLM AS A JUDGE ==========")
    judgment = llm_as_judge(prompt, openai_output, mistral_output)
    print(judgment)

# Example usage
if __name__ == "__main__":
    test_prompt = "Explain the theory of evolution by natural selection."
    evaluate_llm_outputs(test_prompt)
