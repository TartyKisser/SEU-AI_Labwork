from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
# 下载Llama-3.2-1B-Instruct模型到指定目录
# 注意：需要提供有效的token
snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B-Instruct", 
    repo_type="model",
    token="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",  # 使用你的实际token
    cache_dir="/mnt/e/autoplan/ChatEV-main/model"  # 选择一个有足够空间的目录
)