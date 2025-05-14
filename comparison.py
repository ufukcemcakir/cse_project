import json
import matplotlib.pyplot as plt

# === Config ===
INPUT_FILE = "comparison_scores.jsonl"

# === Load Data ===
titles = []
rl_scores = []
non_rl_scores = []

rl_higher_count = 0
non_rl_higher_count = 0
tie_count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            entry = json.loads(line)
            title = entry.get("abstract_title", "[Untitled]")
            rl = entry.get("rl_score")
            non_rl = entry.get("non_rl_score")

            if rl is not None and non_rl is not None:
                titles.append(title[:30] + "..." if len(title) > 30 else title)
                rl_scores.append(rl)
                non_rl_scores.append(non_rl)

                if rl > non_rl:
                    rl_higher_count += 1
                elif non_rl > rl:
                    non_rl_higher_count += 1
                else:
                    tie_count += 1

        except json.JSONDecodeError:
            continue

# === Stats ===
avg_rl = sum(rl_scores) / len(rl_scores) if rl_scores else 0
avg_non_rl = sum(non_rl_scores) / len(non_rl_scores) if non_rl_scores else 0

print(f"📊 Average RL Score:     {avg_rl:.2f}")
print(f"📊 Average Non-RL Score: {avg_non_rl:.2f}")
print(f"🏆 RL better:     {rl_higher_count} times")
print(f"🏆 Non-RL better: {non_rl_higher_count} times")
print(f"🤝 Tie:           {tie_count} times")

# === Plot ===
plt.figure(figsize=(12, 6))
x = list(range(len(titles)))
plt.plot(x, rl_scores, label="RL Score", marker="o")
plt.plot(x, non_rl_scores, label="Non-RL Score", marker="s")
plt.xticks(x, titles, rotation=45, ha="right")
plt.xlabel("Abstracts")
plt.ylabel("Mistral Score (0–10)")
plt.title("Comparison of RL vs Non-RL Reading Path Scores")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
