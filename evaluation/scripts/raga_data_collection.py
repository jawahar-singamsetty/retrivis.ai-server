"""
RAGAS Data Collection Script
Runs test questions through your RAG system and collects evaluation data.
"""

import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.retrieval.index import retrieve_context
from src.rag.retrieval.utils import prepare_prompt_and_invoke_llm

# Configuration
PROJECT_ID = "<your project id here>"

TEST_QUESTIONS = [
    "<add your test questions here>"
]


def collect_rag_data(project_id: str, questions: list) -> list:
    """Run questions through RAG pipeline and collect data."""
    dataset = []
    
    for question in questions:
        print(f"Processing: {question}")
        
        # Retrieve context
        texts, images, tables, citations = retrieve_context(project_id, question)
        
        # Prepare contexts for RAGAS
        contexts = texts + [f"[TABLE]\n{table}" for table in tables]
        
        # Generate answer
        answer = prepare_prompt_and_invoke_llm(question, texts, [], tables)
        
        dataset.append({
            "question": question,
            "contexts": contexts or ["No context found"],
            "answer": answer
        })
    
    return dataset


if __name__ == "__main__":
    # Collect and save data
    dataset = collect_rag_data(PROJECT_ID, TEST_QUESTIONS)
    
    output_path = Path(__file__).parent /"evaluation" / "datasets" / "ragas_evaluation_dataset.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(dataset)} questions to {output_path}")