import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import json

# Load training data
with open("llm_training_data.jsonl", "r", encoding="utf-8") as f:
    training_data = [json.loads(line) for line in f if line.strip()]

embedder = SentenceTransformer("all-MiniLM-L6-v2")

class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

model = PolicyNet(input_dim=384)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 5
for epoch in range(EPOCHS):
    total_loss = 0
    for example in training_data:
        papers = example["papers"]
        reward = float(example["score"])
        if len(papers) < 2:
            continue

        titles = [p["title"] for p in papers]
        embeddings = torch.tensor(embedder.encode(titles), dtype=torch.float)

        scores = model(embeddings).squeeze()
        probs = torch.softmax(scores, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        loss = -log_prob * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {total_loss / len(training_data):.4f}")

torch.save(model.state_dict(), "reinforce_policy_model.pt")
