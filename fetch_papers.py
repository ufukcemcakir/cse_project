import requests
import json
import time

def get_papers_by_field(field, max_results):
    papers_list = []
    page = 1
    per_page = 100

    while len(papers_list) < max_results:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={field}&fields=title,abstract,authors,year,venue,citationCount,fieldsOfStudy,openAccessPdf,url&limit={per_page}&offset={(page-1)*per_page}"
        
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
        
        data = response.json()
        
        if "data" not in data or not data["data"]:
            print("No more papers found.")
            break
        
        papers_list.extend(data["data"])
        print(f"Retrieved {len(papers_list)} papers...")
        page += 1
        time.sleep(2)
    
    if papers_list:
        with open(f"{field.replace(' ', '_')}_papers.json", "w") as f:
            json.dump(papers_list, f, indent=4)
        print(f"Saved {len(papers_list)} papers in {field.replace(' ', '_')}_papers.json")
    else:
        print("No papers found.")

get_papers_by_field("Climate", max_results=10)
