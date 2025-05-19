import torch
from elastic_models.diffusers import FluxPipeline

mode_name = 'black-forest-labs/FLUX.1-dev'
hf_token = ''
device = torch.device("cuda")

pipeline = FluxPipeline.from_pretrained(
    mode_name,
    torch_dtype=torch.bfloat16,
    token=hf_token,
    # S, M, L, XL
    mode='S'
)
pipeline.to(device)

prompts = ["Kitten eating a banana"]
output = pipeline(prompt=prompts)

for prompt, output_image in zip(prompts, output.images):
    output_image.save((prompt.replace(' ', '_') + '.png'))
