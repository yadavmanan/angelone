from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
import os
import nltk
import requests
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from openai import OpenAI

# Download punkt tokenizer
nltk.download('punkt')

# Load API keys and initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://tfy.ml-apps-dev.siemens-healthineers.com/api/llm/api/inference/openai"
)

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

def fetch_wikipedia_page(title):
    url = f"https://en.wikipedia.org/wiki/Sidhu_Moose_Wala"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise ValueError(f"Failed to fetch Wikipedia page for {title}")

def split_into_passages(text, max_len=300):
    sentences = nltk.sent_tokenize(text)
    passages = []
    current_passage = ""
    for sent in sentences:
        if len(current_passage) + len(sent) < max_len:
            current_passage += " " + sent
        else:
            passages.append(current_passage.strip())
            current_passage = sent
    if current_passage:
        passages.append(current_passage.strip())
    return passages

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load NLI model and tokenizer
nli_model_name = "roberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

def verify_claim_with_context(claim, context):
    inputs = nli_tokenizer.encode_plus(context, claim, return_tensors="pt", truncation=True, max_length=512)
    outputs = nli_model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    labels = ["Contradiction", "Neutral", "Supported"]
    max_idx = probs.argmax()
    label = labels[max_idx]
    confidence = probs[max_idx]
    return label, confidence

def detect_hallucinations(llm_output, reference_text):
    claims = nltk.sent_tokenize(llm_output)
    passages = split_into_passages(reference_text)

    passage_embeddings = embedder.encode(passages, convert_to_tensor=True)

    results = []
    for claim in claims:
        claim_embedding = embedder.encode(claim, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(claim_embedding, passage_embeddings)[0]
        top_results = torch.topk(cos_scores, k=3)

        relevant_passages = [passages[idx] for idx in top_results.indices]

        context = " ".join(relevant_passages)

        label, confidence = verify_claim_with_context(claim, context)

        results.append({
            "claim": claim,
            "verdict": label,
            "confidence": confidence,
            "evidence": relevant_passages
        })

    hallucinated = [r for r in results if r['verdict'] != "Supported"]
    hallucination_rate = len(hallucinated) / len(results) if results else 0

    return results, hallucination_rate

if __name__ == "__main__":
    # Fetch reference context
    reference_text = fetch_wikipedia_page("Sidhu_Moose_Wala")

    # Prompt for the LLM
    prompt = "Write a brief biography of Sidhu Moose Wala highlighting his career and important life events."

    # Generate output from LLM
    llm_output = call_openai(prompt)
    print("Generated LLM Output:\n", llm_output)
    print("\nRunning hallucination detection...\n")

    # Detect hallucinations
    results, hallucination_rate = detect_hallucinations(llm_output, reference_text)

    # Print detailed results
    for res in results:
        print(f"Claim: {res['claim']}")
        print(f"Verdict: {res['verdict']} (Confidence: {res['confidence']:.2f})")
        print(f"Evidence snippet(s):")
        for ev in res['evidence']:
            print(f" - {ev[:200]}...")
        print()

    print(f"Overall hallucination rate: {hallucination_rate:.2%}")
