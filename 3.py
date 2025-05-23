import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup OpenAI client
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://tfy.ml-apps-dev.siemens-healthineers.com/api/llm/api/inference/openai"
)

# Semantic similarity model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
SIMILARITY_THRESHOLD = 0.85

# Test suite
test_suite = [
    {
        "prompt": "What is the capital of France?",
        "expected": "Paris is the capital city of France."
    },
    {
        "prompt": "Explain photosynthesis.",
        "expected": "Photosynthesis is the process in plants where sunlight is converted into chemical energy."
    },
    {
        "prompt": "Write the name of Mahatama Gandhi's birthplace.",
        "expected": "Mahatma Gandhi was born in Porbandar, India."
    }
]

# Prompt templates to test
prompt_templates = [
    "Answer this question within one line something like - prompt : What is the capital of France?,  expected: Paris is the capital city of France. : {}",
    "Please respond carefully: {}",
    "Q: {} A:",
    "{}"
]

def call_openai(prompt: str) -> str:
    try:
        stream = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI bot. Just answer the question in one line. Give direct facts."},
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

def evaluate_response(expected: str, actual: str) -> float:
    embeddings = embedding_model.encode([expected, actual], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

def run_template_comparison():
    template_results = []

    for template in prompt_templates:
        print(f"\nTesting prompt template: {template}")
        total_similarity = 0
        report = []

        for test in test_suite:
            prompt_filled = template.format(test['prompt'])
            actual = call_openai(prompt_filled)
            similarity = evaluate_response(test['expected'], actual)
            total_similarity += similarity

            report.append({
                "prompt": test['prompt'],
                "expected": test['expected'],
                "actual": actual,
                "similarity": similarity,
                "passed": similarity >= SIMILARITY_THRESHOLD
            })

            print(f"Prompt: {test['prompt']}\nExpected: {test['expected']}\nActual: {actual}\nSimilarity: {similarity:.3f}\n---")

        avg_similarity = total_similarity / len(test_suite)
        template_results.append((template, avg_similarity))
        print(f"Average similarity for template '{template}': {avg_similarity:.3f}")

    best_template, best_score = max(template_results, key=lambda x: x[1])
    print(f"\nBest prompt template selected: '{best_template}' with average similarity {best_score:.3f}")

if __name__ == "__main__":
    run_template_comparison()
