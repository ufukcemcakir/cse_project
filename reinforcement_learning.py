import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
from torch.distributions import Categorical
from sentence_transformers import SentenceTransformer

# --- Config ---
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 384
HIDDEN_DIM = 128
EPOCHS = 5
LR = 1e-4
TOP_K = 5

# --- Model ---
class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # score for each paper
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)  # (batch,)

# --- Load Data ---
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return entries

# --- Training ---
def train_policy_net(data_path):
    data = load_data(data_path)
    model = PolicyNet(INPUT_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Normalize rewards
    rewards = [e["score"] for e in data]
    mean_r, std_r = sum(rewards)/len(rewards), (sum((r - sum(rewards)/len(rewards))**2 for r in rewards)/len(rewards))**0.5

    for epoch in range(1, EPOCHS+1):
        total_loss = 0
        random.shuffle(data)

        for entry in data:
            papers = entry["papers"]
            titles = [p["title"] for p in papers]
            reward = entry["score"]
            norm_reward = (reward - mean_r) / (std_r + 1e-8)

            with torch.no_grad():
                embeddings = torch.tensor([EMBEDDER.encode(t) for t in titles], dtype=torch.float32).to(DEVICE)

            logits = model(embeddings)
            dist = Categorical(logits=logits)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # Option A: normalized reward
            loss = -log_prob * norm_reward - 0.01 * entropy.mean()

            # Option B (alternative): advantage-based reward (use instead of norm_reward)
            # baseline = 6.0
            # advantage = reward - baseline
            # loss = -log_prob * advantage - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Debug
            print(f"[E{epoch}] ➤ Picked: {titles[action]} | Reward: {reward} | Loss: {loss.item():.2f}")

        avg_loss = total_loss / len(data)
        print(f"📉 Epoch {epoch}/{EPOCHS} - Avg Loss: {avg_loss:.4f}\n")

    torch.save(model.state_dict(), "policy_net_rl.pt")
    print("✅ Model saved to policy_net_rl.pt")

# --- Run ---
if __name__ == "__main__":
    train_policy_net("llm_training_data.jsonl")  # or "llm_training_data_enriched.jsonl"
