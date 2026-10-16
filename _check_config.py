from transformers import AutoConfig
config = AutoConfig.from_pretrained(r'D:\claudeAgent\RAG\florence-2-large', trust_remote_code=True)
print("_attn_implementation:", getattr(config, "_attn_implementation", "NOT SET"))
print("text_config._attn_implementation:", getattr(config.text_config, "_attn_implementation", "NOT SET"))
