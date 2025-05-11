import json
from llm_evaluator import evaluate_reading_path_with_rubric

RL_FILE = "rl_results.jsonl"
NON_RL_FILE = "non_rl_results.jsonl"
OUTPUT_LOG = "comparison_scores.jsonl"

def flatten_papers(papers_by_concept):
    """
    Flatten a dictionary of concept -> list of papers into a unique list of papers.
    Removes duplicates based on 'paperId'.
    """
    if not isinstance(papers_by_concept, dict):
        return []

    seen_ids = set()
    flat_papers = []
    for paper_list in papers_by_concept.values():
        for paper in paper_list:
            pid = paper.get("paperId")
            if pid and pid not in seen_ids:
                flat_papers.append(paper)
                seen_ids.add(pid)
    return flat_papers

def evaluate_and_log():
    with open(RL_FILE, "r", encoding="utf-8") as f_rl, open(NON_RL_FILE, "r", encoding="utf-8") as f_non_rl:
        rl_entries = [json.loads(line) for line in f_rl if line.strip()]

        non_rl_entries = [json.loads(line) for line in f_non_rl if line.strip()]


    for idx, (rl_entry, non_rl_entry) in enumerate(zip(rl_entries, non_rl_entries), 1):
        abstract = rl_entry.get("abstract_title", "[no title]")
        print(f"\n🔍 Evaluating abstract {idx}: {abstract}")

        rl_papers = flatten_papers(rl_entry.get("papers", {}))
        non_rl_papers = flatten_papers(non_rl_entry.get("papers", {}))

        if not rl_papers and not non_rl_papers:
            print("⚠️ Skipping: No valid papers found in either RL or non-RL.")
            continue

        rl_score = evaluate_reading_path_with_rubric(rl_papers, abstract)
        non_rl_score = evaluate_reading_path_with_rubric(non_rl_papers, abstract)

        print(f"✅ RL Score = {rl_score}, non-RL Score = {non_rl_score}")

        # Save comparison
        result = {
            "abstract_title": abstract,
            "rl_score": rl_score,
            "non_rl_score": non_rl_score
        }
        with open(OUTPUT_LOG, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    evaluate_and_log()
