# ============================================================
# 数据清洗完整流程
# 任务：清洗 GitHub issues 数据集
# ============================================================

# ---------- 1. 加载已抓取的数据 ----------
from datasets import load_dataset

issues_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")


# ---------- 2. 随机抽样，查看数据规律 ----------
# shuffle + select：随机抽 3 条看看
sample = issues_dataset.shuffle(seed=666).select(range(3))

# zip 同时遍历两列，对比 URL 和 pull_request
for url, pr in zip(sample["html_url"], sample["pull_request"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n")

# 发现规律：
#   PR   → pull_request 是字典
#   Issue → pull_request 是 None


# ---------- 3. 加新列：is_pull_request ----------
# map + lambda：根据 pull_request 是否为 None 判断
issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)


# ---------- 4. 扩充数据：给每条 issue 加评论 ----------
import requests

# 4.1 定义函数：调 GitHub API 拿评论
def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)          # headers 需要认证
    return [r["body"] for r in response.json()]            # 只取 body 字段

# 4.2 测试函数
print(get_comments(2792))

# 4.3 用 map 给每条 issue 加 comments 列
# ⚠️ 每条 issue 都要发一次 API 请求，可能几分钟
issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)


# ---------- 5. （可选）保存清洗后的数据 ----------
# issues_with_comments_dataset.to_json("datasets-issues-with-comments.jsonl")