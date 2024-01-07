import os 
from huggingface_hub import hf_hub_download  # Load model directly

# 添加镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.system('export HF_ENDPOINT=https://hf-mirror.com')
saved_dir = "/root/code/InternLM/homework/course_2/internlm-20b"
# hf_hub_download(repo_id="internlm/internlm-20b", subfolder=saved_dir, filename="config.json")

os.system('export HF_ENDPOINT=https://hf-mirror.com && huggingface-cli download --resume-download internlm/internlm-20b --local-dir "{}"'.format(saved_dir))
