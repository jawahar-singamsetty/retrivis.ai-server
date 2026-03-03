"""
Simple RAGAS Evaluation Script
"""

import json
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

# Load your dataset
with open("evaluation/datasets/ragas_evaluation_dataset.json", "r") as f:
    data = json.load(f)

# Convert to RAGAS format
dataset = Dataset.from_dict({
    "question": [item["question"] for item in data],
    "answer": [item["answer"] for item in data],
    "contexts": [item["contexts"] for item in data],
})

# Set up evaluator (using GPT-4 for evaluation)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Run config to handle timeouts and retries
run_config = RunConfig(
    timeout=120,       # wait up to 120s per job (default is 30s)
    max_retries=3,     # retry up to 3 times on failure
    max_wait=60,       # wait up to 60s between retries
    seed=42,           # reproducible scores
)

# Run evaluation
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness, 
        answer_relevancy, 
        # context_precision, 
        # context_recall
    ],
    llm=llm,
    embeddings=embeddings,
    run_config=run_config,
)

# Convert to DataFrame first
df = results.to_pandas()

# Save to CSV
df.to_csv("evaluation/datasets/results.csv", index=False)
print("\n✅ Detailed results saved to results.csv")