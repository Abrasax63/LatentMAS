

# CUDA_VISIBLE_DEVICES=0 python run.py --task gsm8k --method latent_mas --model_name Qwen/Qwen3-4B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt sequential --latent_steps 20   > 0.txt &
# CUDA_VISIBLE_DEVICES=1 python run.py --task gsm8k --method latent_mas --model_name Qwen/Qwen3-4B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt hierarchical --latent_steps 20 > 1.txt &
# CUDA_VISIBLE_DEVICES=2 python run.py --task gsm8k --method latent_mas --model_name Qwen/Qwen3-8B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt sequential --latent_steps 20   > 2.txt &
# CUDA_VISIBLE_DEVICES=3 python run.py --task gsm8k --method latent_mas --model_name Qwen/Qwen3-8B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt hierarchical --latent_steps 20 > 3.txt &

# CUDA_VISIBLE_DEVICES=4 python run.py --task gsm8k --method latent_mas --model_name Qwen/Qwen3-4B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt sequential --latent_steps 20   --latent_space_realign > re_0.txt &
# CUDA_VISIBLE_DEVICES=5 python run.py --task gsm8k --method latent_mas --model_name Qwen/Qwen3-4B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt hierarchical --latent_steps 20 --latent_space_realign > re_1.txt &
# CUDA_VISIBLE_DEVICES=6 python run.py --task gsm8k --method latent_mas --model_name Qwen/Qwen3-8B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt sequential --latent_steps 20   --latent_space_realign > re_2.txt &
# CUDA_VISIBLE_DEVICES=7 python run.py --task gsm8k --method latent_mas --model_name Qwen/Qwen3-8B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt hierarchical --latent_steps 20 --latent_space_realign > re_3.txt &

# CUDA_VISIBLE_DEVICES=0 python run.py --task gsm8k --method text_mas --model_name Qwen/Qwen3-4B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt sequential   > text_mas_0.txt &
# CUDA_VISIBLE_DEVICES=1 python run.py --task gsm8k --method text_mas --model_name Qwen/Qwen3-4B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt hierarchical > text_mas_1.txt &
# CUDA_VISIBLE_DEVICES=2 python run.py --task gsm8k --method text_mas --model_name Qwen/Qwen3-8B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt sequential   > text_mas_2.txt &
# CUDA_VISIBLE_DEVICES=3 python run.py --task gsm8k --method text_mas --model_name Qwen/Qwen3-8B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt hierarchical > text_mas_3.txt &

# CUDA_VISIBLE_DEVICES=4 python run.py --task gsm8k --method text_mas --model_name Qwen/Qwen3-4B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt sequential   --use_vllm > text_mas_vllm_0.txt &
# CUDA_VISIBLE_DEVICES=5 python run.py --task gsm8k --method text_mas --model_name Qwen/Qwen3-4B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt hierarchical --use_vllm > text_mas_vllm_1.txt &
# CUDA_VISIBLE_DEVICES=6 python run.py --task gsm8k --method text_mas --model_name Qwen/Qwen3-8B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt sequential   --use_vllm > text_mas_vllm_2.txt &
# CUDA_VISIBLE_DEVICES=7 python run.py --task gsm8k --method text_mas --model_name Qwen/Qwen3-8B --max_samples 20 --generate_bs 20 --max_new_tokens 2048 --temperature 0.6 --prompt hierarchical --use_vllm > text_mas_vllm_3.txt &
