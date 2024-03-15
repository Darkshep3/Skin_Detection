from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from transformers.models.llama import convert_llama_weights_to_hf
import torch
import os

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# model_id = "mistralai/Mistral-7B-v0.1"
model_id = "tiiuae/falcon-7b"
# model_id = "D:\Allen_2023\model_weights\llama-7b"

cache_dir="D:\Allen_2023\model_weights"

# convert downloaded llama weights into huggingface weights
# input_dir = "D:\Allen_2023\model_weights\llama"
# spm_path = os.path.join(input_dir, "tokenizer.model")
# convert_llama_weights_to_hf.write_model(
#             model_path=cache_dir,
#             input_base_path=input_dir,
#             model_size="7B",
#             safe_serialization=False,
#             tokenizer_path=spm_path,
#         )



tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             cache_dir=cache_dir,
                                             torch_dtype=torch.float16).to(device)

# text = """
# Pretending you are a doctor, please select what disease the patient has from the options: vasculitis, dermatofibroma, eczema. The patient narrative is: 
# "
# As a patient, I've been troubled by the emergence of small, round, or oval bumps on my skin. These bumps are typically less than 1 centimeter in diameter and exhibit varying hues, ranging from flesh-colored to brown or reddish-brown. Their appearance has left me concerned and uncertain about their cause, prompting me to seek medical evaluation and guidance.
# "
# """
text = "Yes or no: would a pear sink in water?"
encoded_input = tokenizer([text], return_tensors='pt').to(device)
output = model.generate(**encoded_input, max_new_tokens=200, do_sample=True)

output = tokenizer.batch_decode(output)[0]
print(output)
