from __future__ import annotations

import json
import random
import time
from urllib.parse import urlparse

import requests
from locust import User, between, events, task

"""
Locust file to benchmark RunPod diffusion endpoint (/v1/images/generations).
All configuration is hardcoded.
"""

# Insert your inference URL here
INFERENCE_URL = "https://<your-inference-url>/v1/images/generations"
ENDPOINT_PATH = urlparse(INFERENCE_URL).path

# Insert your authorization token here, e.g. "Bearer <apitoken>"
# If you didn't set any authorization, set AUTHORIZATION = ""
AUTHORIZATION = ""
MODEL_NAME = "flux-1-schnell-s-bs4"
POS_PROMPT = "sunset"
ASPECT_RATIO = "1:1"
GUIDANCE = 6.5
STEPS = 4

MIN_WAIT = 0
MAX_WAIT = 0

MAX_RETRIES = 999999
RETRY_DELAY = 1.0
TIMEOUT = 300

HEADERS = {
    "Authorization": AUTHORIZATION,
    "Content-Type": "application/json",
    "X-Model-Name": MODEL_NAME,
}

class DiffusionUser(User):
    wait_time = between(MIN_WAIT, MAX_WAIT)

    @task
    def infer(self) -> None:
        seed = random.randint(0, 10000)

        for attempt in range(MAX_RETRIES + 1):
            start_time = time.perf_counter()
            exc = None
            try:
                payload = {
                    "prompt": POS_PROMPT,
                    "seed": seed,
                    "aspect_ratio": ASPECT_RATIO,
                    "guidance_scale": GUIDANCE,
                    "num_inference_steps": STEPS,
                }

                resp = requests.post(
                    INFERENCE_URL,
                    headers=HEADERS,
                    data=json.dumps(payload),
                    timeout=TIMEOUT,
                )

                response_time = time.perf_counter() - start_time
                response_time_ms = int(response_time * 1000)

                if resp.ok:
                    response_length = len(resp.content or b"")
                    events.request.fire(
                        request_type="inference",
                        name=ENDPOINT_PATH,
                        response_time=response_time_ms,
                        response_length=response_length,
                        exception=None,
                    )
                    break

                exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                raise exc

            except Exception as exc:
                response_time = time.perf_counter() - start_time
                response_time_ms = int(response_time * 1000)
                events.request.fire(
                    request_type="inference",
                    name=ENDPOINT_PATH,
                    response_time=response_time_ms,
                    response_length=0,
                    exception=exc,
                )
                if attempt >= MAX_RETRIES:
                    break
                time.sleep(RETRY_DELAY)
