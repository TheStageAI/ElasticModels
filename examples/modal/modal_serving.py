import modal

MODEL_TYPE = "dev"
app = modal.App(f"flux-{MODEL_TYPE}-thestage-blackwell-test")
# Use this image for H100 and L40s GPUs
IMG_NVIDIA_BASE = "public.ecr.aws/i3f7g5s7/thestage/elastic-models:0.1.2-diffusers-nvidia-24.09a"
# Use this image for B200 GPU
IMG_BLACKWELL = "public.ecr.aws/i3f7g5s7/thestage/elastic-models:0.1.2-diffusers-blackwell-24.09a"
HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
ENVS = {
    "MODEL_REPO": f"black-forest-labs/FLUX.1-{MODEL_TYPE}",
    "MODEL_BATCH": "4",
    "THESTAGE_AUTH_TOKEN": "",
    "HUGGINGFACE_ACCESS_TOKEN": "",
    "PORT": "80",
    "PORT_HEALTH": "80",
    "HF_HOME": "/cache/huggingface",
}
image = modal.Image.from_registry(
    IMG_BLACKWELL,
    add_python="3.11"
)\
    .env(ENVS)\
    .add_local_file("modal_start.sh", "/usr/local/bin/startup.sh", copy=True)\
    .add_local_file("supervisord.conf", "/etc/supervisor/supervisord.conf", copy=True)\
    .run_commands("chmod +x /usr/local/bin/startup.sh")\
    .entrypoint(["/usr/local/bin/startup.sh"])

@app.function(
    image=image,
    gpu="B200",
    min_containers=8,  
    max_containers=8,
    timeout=10000,
    ephemeral_disk=600 * 1024,
    volumes={"/opt/project/.cache": HF_CACHE},
    startup_timeout=60*20
)
@modal.web_server(
    80,
    label=f"flux-{MODEL_TYPE}-blackwell-test", 
    startup_timeout=60*20
)
def serve():
    pass
