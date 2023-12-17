export HUGGINGFACE_HUB_CACHE=/tmp/.cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export NUMEXPR_MAX_THREADS=96
pip install -e .
pip install -e ".[vllm]"
echo "$@"
python start.py "$@"