export HUGGINGFACE_HUB_CACHE=/tmp/.cache
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -e .
pip install -e ".[vllm]"
echo "$@"
python start.py "$@"
# lm_eval --model=vllm --model_args=pretrained=epfl-llm/meditron-7b,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,download_dir=/home/ec2-user/SageMaker/huggingface --tasks=mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_medicine,mmlu_medical_genetics --batch_size=auto