from collections import defaultdict
import json

file_path = "metadata.jsonl"
field_counts = defaultdict(int)

with open(file_path, "r") as f:
    for line in f:
        paper = json.loads(line)
        fields = paper.get("mag_field_of_study", [])
        for field in fields:
            field_counts[field] += 1

# Print the distribution
for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{field}: {count} papers")
