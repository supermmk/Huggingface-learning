from datasets import load_dataset

issues_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")

sample = issues_dataset.shuffle(seed=42).select(range(3))

for url, pr in zip(sample["html_url"], sample["pull_request"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n}")

issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)


import requests

def get_comments(issue_number):
    url = f"https://api.github.com/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]

issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)