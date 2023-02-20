import io
import os
import asyncio
import httpx
import base64
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from exceptions import RunpodStatusUnexpectedError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
# from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

# todo: if we don't need any pre processing apart from rate limiting, we can just use this service for rate limiting
# and not for the actual inference calls and response handling. When the client
# wants to run one of this service's endpoints, it will send a request that will be checked against the rate limit.
# If the rate limit is not exceeded, the client (and not this service) will initiate the inference request
# to the model's endpoint. Notice that this approach has some security implications (key stored in client, etc.),

src_path = Path('.')
env_path = src_path / '.env'
load_dotenv(dotenv_path=env_path)  # load environment variables from .env file into the os.environ object.

debug = os.environ.get("DEBUG")
runpod_status_url = os.environ.get("RUNPOD_STATUS_URL")
runpod_run_url = os.environ.get("RUNPOD_RUN_URL")
runpod_api_key = os.environ.get("RUNPOD_API_KEY")
redis_url = os.environ.get("REDIS_URL")
redis_db = os.environ.get("REDIS_DB")

redis_host = redis_url.split("//")[1].split(":")[0]
redis_port = redis_url.split("//")[1].split(":")[1].split("/")[0]

limiter_storage_uri = f"redis://{redis_host}:{redis_port}/{redis_db}"

limiter = Limiter(key_func=get_remote_address, storage_uri=limiter_storage_uri)

app = FastAPI()

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

delay = 1.5  # in seconds


async def runpod_status_output(
    run_id: str,
    num_errors: int = 0,
    status_url: str = runpod_status_url,
    runpod_key: str = runpod_api_key,
) -> dict or None:
    """
    Returns the output of the run with the given ID once it is completed.
    """
    # in progress run_response:
    # {
    #     "delayTime": 2624,
    #     "id": "c80ffee4-f315-4e25-a146-0f3d98cf024b",
    #     "input": {
    #         "prompt": "a cute magical flying dog, fantasy art drawn by disney concept artists"
    #     },
    #     "status": "IN_PROGRESS"
    # }
    # completed run_response:
    # {
    #   "delayTime": 123456, // (milliseconds) time in queue
    #   "executionTime": 1234, // (milliseconds) time it took to complete the job
    #   "gpu": "24", // gpu type used to run the job
    #   "id": "c80ffee4-f315-4e25-a146-0f3d98cf024b",
    #   "input": {
    #     "prompt": "a cute magical flying dog, fantasy art drawn by disney concept artists"
    #   },
    #   "output": [
    #     {
    #       "image": "https://job.results1",
    #       "seed": 1
    #     },
    #     {
    #       "image": "https://job.results2",
    #       "seed": 2
    #     }
    #   ],
    #   "status": "COMPLETED"
    # }
    num_retries_on_error = 2
    try:
        while True:
            async with httpx.AsyncClient() as client:
                status_call_response = await client.get(
                    status_url + run_id,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {runpod_key}",
                    },
                )

            http_status = status_call_response.status_code
            print(f"http status: {http_status}, status_url response: {status_call_response.json()}")

            if http_status == 500 and num_errors < num_retries_on_error:
                num_errors += 1
                await asyncio.sleep(delay)

            elif http_status == 200:
                status = status_call_response.json()["status"]
                if status == "IN_PROGRESS":
                    print("IN_PROGRESS")
                    await asyncio.sleep(delay)
                elif status == "IN_QUEUE":
                    print("IN_QUEUE")
                    await asyncio.sleep(delay)
                elif status == "COMPLETED":
                    print("COMPLETED")
                    return status_call_response.json()["output"]
                elif status == "FAILED":
                    print("FAILED")
                    return None
                else:
                    print(f"http_status: {http_status}, unexpected response: {status_call_response.json()}")
                    raise RunpodStatusUnexpectedError(f'unexpected response: {status_call_response.json()}')

            else:
                print(f"unexpected http status: {http_status}, unexpected response: {status_call_response.json()}")
                raise RunpodStatusUnexpectedError(f'http_status: {http_status}, unexpected response: {status_call_response.json()}')
    except Exception as e:
        print(f"Unexpected error: {e}")
        if num_errors < num_retries_on_error:
            num_errors += 1
            return await runpod_status_output(run_id, num_errors)
        else:
            print(f"Too many errors: {num_errors}")
            return None


async def txt2img_runpod(input_data, run_url=runpod_run_url, runpod_key=runpod_api_key):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {runpod_key}'
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(run_url, json=input_data, headers=headers)
        response_data = response.json()
        run_id = response_data["id"]
        try:
            base64_image_string = await runpod_status_output(run_id)
        except Exception as e:
            print(e)
            return None
        return base64_image_string


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/generate_image/", response_model=None)
@limiter.limit("10/minute")
async def generate_img(prompt: str, model: str, seed: int, height: int, width: int, guidance_scale: float,
                       num_inference_steps: int, request: Request):
    runpod_run_input_data = {
        'input': {
            'prompt': prompt,
            'model': model,
            'seed': seed,
            'height': height,
            'width': width,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps
        }
    }
    print(f'runpod_run_input_data: {runpod_run_input_data}')
    # todo: maybe stream chunks of the response as they come in to avoid storing the whole image in memory
    # todo: inference returns a base64encodedstring which can't be streamed with a StreamingResponse because it is
    # not an iterator. so we decode it and make it a file like object to stream it. Probably a better day to do this
    # is to send the array of bytes from inference and just resend them back to the web client without additional
    # processing
    base64_image_string = await txt2img_runpod(runpod_run_input_data)
    image_bytes = base64.b64decode(base64_image_string)
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
