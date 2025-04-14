# reward_logger.py
import json

def save_training_example(concepts, papers, score, output_file="llm_training_data.jsonl"):
    example = {
        "concepts": concepts,
        "papers": papers,
        "score": score
    }
    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(example, f)
        f.write("\n")
