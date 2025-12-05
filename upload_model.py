from huggingface_hub import HfApi, create_repo, upload_folder, login
import os

repo = "zjhhhh/Llama-3.2-3B-Instruct_multi_armo_2reward_SFT"
folder = "./checkpoint/llama3-3B-sft"
token = os.environ.get("HF_TOKEN")

if token:
    login(token=token)

api = HfApi()
create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
upload_folder(repo_id=repo, repo_type="model", folder_path=folder)