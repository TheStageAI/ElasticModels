import modal

MODEL_TYPE = "dev"
app = modal.App(f"flux-{MODEL_TYPE}-thestage-blackwell-test")
# TheStage AI pre-built image (supports L40s, H100, B200)
IMG = "public.ecr.aws/i3f7g5s7/thestage/elastic-models:0.2.0-diffusers-24.09c"
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
    IMG,
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
