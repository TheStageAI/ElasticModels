<div align="center" id="sglangtop">
<img src="images/logo.png" alt="logo" width="400" margin="10px"></img>

<!-- [![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![](https://img.shields.io/badge/Gurubase-(experimental)-006BFF)](https://gurubase.io/g/sglang) -->

</div>

--------------------------------------------------------------------------------

| [**Hugging Face**]()
| [**TheStage AI Platform**](https://app.thestage.ai/)
| [**TheStage AI Website**](https://thestage.ai/)
| [**TheStage AI X**](https://https://x.com/TheStageAI)

---
# Elastic Models: Fast and Flexible Models for Self-Serving
Elastic models are the models produced by TheStage AI ANNA: Automated Neural Networks Accelerator. ANNA allows you to control model size, latency and quality with a simple slider movement. Elastic models:

* Represented by 4 tiers: S, M, L, XL. From fastest to slowest.

* __XL__: Mathematically equivalent neural network, optimized with our DNN compiler. 

* __L__: Near lossless model, with less than 1% degradation obtained on corresponding benchmarks.

* __M__: Faster model, with accuracy degradation less than 1.5%.

* __S__: The fastest model, with accuracy degradation less than 2%.

* Supports LLMs, VLMs, Diffusion models. All models provided in Hugging Face transformers and diffusers libraries. 

* Underlying inference engine supports fp16, bf16, int8, fp8, int4, 2:4 sparsity inference. To control quality of models we are using ANNA: Automated NNs Analyzer. For each point corresponding to number of bitops or model size ANNA finds the best quality solution using supported hardware acceleration techniques. Think of it like JPEG for DNNs.

* No dependecies with TensorRT-LLM, Sglang, vLLM. Simple setup through PyPi. 


### Goals

* Provide flexibility in cost vs quality selection for inference
* Provide clear quality and latency benchmarks
* Provide interface of HF libraries: transformers and diffusers with a single line of code
* Provide models supported on a wide range of hardware, which are pre-compiled and require no JIT.
* Provide the best models and service for self-hosting.
---

![](images/flux.jpeg)




<!-- <details>
<summary>More</summary>

- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details> -->

## Quick Start

__System requirements:__
* GPUs: H100, L40s
* CPU: AMD, Intel
* Python: 3.10-3.12


To work with our models just run these lines in your terminal:

```shell
pip install thestage
pip install elastic_models[nvidia]\
 --index-url https://thestage.jfrog.io/artifactory/api/pypi/pypi-thestage-ai-production/simple\
 --extra-index-url https://pypi.nvidia.com\
 --extra-index-url https://pypi.org/simple
pip install flash_attn==2.7.3 --no-build-isolation
pip uninstall apex
```

Then go to [app.thestage.ai](https://app.thestage.ai), login and generate API token from your profile page. Set up API token as follows:

```shell
thestage config set --api-token <YOUR_API_TOKEN>
```

Congrats, now you can use accelerated models!

## Current state

- **Hardware.** Nvidia H100, L40s. More GPUs are coming.
- **LLMs.** Llama3 1B, 8B instruct, Mistral 7B instruct, Qwen2.5 7B instruct, Deepseek R1: Llama 8B distill, Qwen2.5 7B distill. 
- **Text-to-Image.** FLUX.1-schnell, FLUX.1-dev.
- **VLMs.** Coming soon!
- **Context length.** Demo models support context lenght up to 8192 tokens and batch size up to 32 depending on GPU.
- **Image sizes.** Diffusion models currently supports image resolution up to 1280x1280.
- **Memory usage.** Currently inference engine preallocates memory for maximum possible size. For more precise memory control - contact us at contact@thestage.ai
- **Speed.** Models demonstrates world leading performance comparing to open benchmarks. For instnace, LLama3 8B gives ~195 tok/s with 100/300 input-output test and ~170 tok/s with 4096/1000 input-output test.

## Roadmap

- Release models serving
- VLMs releaase
- Text-to-video models release. 


## Production Cases
The project has been deployed to large-scale production, generating trillions of tokens every day.
It is supported by the following institutions: AMD, Atlas Cloud, Baseten, Cursor, DataCrunch, Etched, Hyperbolic, Iflytek, Jam & Tea Studios, LinkedIn, LMSYS, Meituan, Nebius, Novita AI, NVIDIA, Oracle, RunPod, Stanford, UC Berkeley, UCLA, xAI, and 01.AI.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us

For companies interested in deploying TheStage AI inference engine in their environment, application of ANNA for custom models or partnership please contact us at contact@thestage.ai.



# Aknowlegents