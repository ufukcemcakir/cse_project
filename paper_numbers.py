import requests

def count_papers_in_field(field):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={field}&fields=title&limit=1"
    response = requests.get(url)
    data = response.json()
    return data.get("total", 0)

# Example usage
fields = ["Climate"]
for field in fields:
    count = count_papers_in_field(field)
    print(f"{field}: {count} papers")
