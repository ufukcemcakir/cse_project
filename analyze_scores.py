import json

filename = "rl_vs_non_rl_evaluation.jsonl"

rl_scores = []
non_rl_scores = []

with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        rl_scores.append(data["rl_score"]["score"])
        non_rl_scores.append(data["non_rl_score"]["score"])

def average(scores):
    return sum(scores) / len(scores)

print("🔍 Offline Evaluation Results:")
print(f"- Avg RL Score:     {average(rl_scores):.2f}")
print(f"- Avg Non-RL Score: {average(non_rl_scores):.2f}")
