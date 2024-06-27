from datasets import load_dataset
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("RLHFlow/LLaMA3-SFT")
model = AutoModelForCausalLM.from_pretrained("RLHFlow/LLaMA3-SFT")

ds = load_dataset("BAAI/Infinity-Instruct")

