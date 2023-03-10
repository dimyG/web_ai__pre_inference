import io
import os
# import asyncio
# import httpx
import base64
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
# from exceptions import RunpodStatusUnexpectedError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
# from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
import jwt
import logging
from logging.config import dictConfig
from config import LogConfig
from utils import User
from middleware import request_context_middleware, _request_ctx_var
from starlette.middleware.base import BaseHTTPMiddleware
from runpod_calls import txt2img_server_based, runpod_run, runpod_status

# todo: if we don't need any pre processing apart from rate limiting, we can just use this service for rate limiting
# and not for the actual inference calls and response handling. When the client
# wants to run one of this service's endpoints, it will send a request that will be checked against the rate limit.
# If the rate limit is not exceeded, the client (and not this service) will initiate the inference request
# to the model's endpoint. Notice that this approach has some security implications (key stored in client, etc.),

# Note: It is recommended to call the dictConfig(...) function before the FastAPI initialization.
dictConfig(LogConfig().dict())
logger = logging.getLogger("pre_inference")

src_path = Path('.')
env_path = src_path / '.env'
load_dotenv(dotenv_path=env_path)  # load environment variables from .env file into the os.environ object.

debug = os.environ.get("DEBUG")
redis_host = os.environ.get("REDIS_HOST")
redis_port = os.environ.get("REDIS_PORT")
redis_db = os.environ.get("REDIS_DB")
jwt_secret = os.environ.get("JWT_SECRET")

if redis_port:
    redis_port = int(redis_port)
if redis_db:
    redis_db = int(redis_db)


def limiter_key(request: Request) -> str:
    # slow api's get_remote_address returns request.client.host. But if the request is proxied through AWS API Gateway
    # then the client's ip address is in the forwarded header not in the host (the request.client.host doesn't exist).
    # So we need to check if the request is proxied and if so, get the client's ip address from the forwarded header.
    # The forwarded header is a comma separated list of proxies that the request has passed through.
    forwarded = request.headers.get('forwarded')
    # logger.debug(f'forwarded: {forwarded}')
    if forwarded:
        #  return the ip of the first proxy (the client) in the chain
        return forwarded.split(';')[0].split('=')[1]
    else:
        logger.error('forwarded header not found, a common limiter key will be used')
        # this would be the AWS API Gateway host (the last proxy in the chain) which is the same for all users
        return request.headers.get('host')


limiter_storage_uri = f"redis://{redis_host}:{redis_port}/{redis_db}"
limiter = Limiter(key_func=limiter_key, storage_uri=limiter_storage_uri)
origins = ["*"]

app = FastAPI()

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(BaseHTTPMiddleware, dispatch=request_context_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/preinference/")
async def read_root():
    return {"Hello": "World"}


def jwt_decode(request: Request, secret: str = jwt_secret):
    try:
        encoded_jwt = request.headers['Authorization'].split(' ')[1]
        decoded_jwt = jwt.decode(encoded_jwt, secret, algorithms=["HS256"])
    except KeyError:
        # anonymous user
        return
    except Exception:
        # if expired token, invalid token, etc. treat it the same as an anonymous user
        return
    return decoded_jwt


def get_limit_for_user() -> str:
    """ Return the rate limit for the current user.
     Have in mind that as of slowapi version 0.1.7, you can't pass the request object to a function that returns
     the rate limit and is called from the @limiter.limit decorator.
     See issue https://github.com/laurentS/slowapi/issues/41 and https://github.com/laurentS/slowapi/issues/13
     So the alternative is to add the current request to the global context using
     a middleware and then retrieve it from the global context in the function that returns the rate limit.
     This is what the request_context_middleware does."""
    request = _request_ctx_var.get()
    decoded_jwt = jwt_decode(request)
    # logger.debug(f'decoded_jwt: {decoded_jwt}')
    user = User.from_jwt(decoded_jwt)
    rate_limit = user.get_rate_limit()
    # logger.debug(f'Tier {user.tier} - rate_limit: {rate_limit}')
    return rate_limit


@app.post("/api/preinference/generate_image/", response_model=None)
@limiter.limit(get_limit_for_user)
async def generate_img(prompt: str, model: str, seed: int, height: int, width: int, guidance_scale: float,
                       num_inference_steps: int, request: Request):
    # In this version of the service, both the runpod run and the runpod status polling are done by the server
    # Not currently used
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
    # todo: return bytes from inference. Currently inference returns a base64encodedstring which can't be
    # streamed with a StreamingResponse because it is
    # not an iterator. so we must decode it and make it a file like object to stream it. Probably a better way to
    # do this is to send from inference the array of bytes directly and just resend them back to the web client without
    # additional need to decode the string into bytes.
    base64_image_string = await txt2img_server_based(runpod_run_input_data)
    # base64 decoding is cpu bound so even if it was async it would still block the event loop
    image_bytes = base64.b64decode(base64_image_string)
    # todo: maybe stream chunks of the response as they come in to avoid storing the whole image in memory
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


# Only the initiate run endpoint is currently used by the web client. Polling for status is done by the client.
@app.post("/api/preinference/initiate_run/", response_model=None)
@limiter.limit(get_limit_for_user)
async def initiate_run(prompt: str, model: str, seed: int, height: int, width: int, guidance_scale: float,
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
    run_id = await runpod_run(runpod_run_input_data)
    return {'run_id': run_id}


@app.get("/api/preinference/get_run_status/", response_model=None)
# @limiter.limit(get_limit_for_user)
async def get_run_status(run_id: str, request: Request):
    run_status, base64_image_string = await runpod_status(run_id)
    return {'run_status': run_status}


@app.get("/api/preinference/get_image/", response_model=None)
@limiter.limit(get_limit_for_user)
async def get_image(run_id: str, request: Request):
    # streams the image as a binary string
    run_status, base64_image_string = await runpod_status(run_id)
    if run_status == 'COMPLETED':
        image_bytes = base64.b64decode(base64_image_string)
        return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
    return {'run_status': run_status}
