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
from runpod_calls import txt2img_runpod

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
redis_url = os.environ.get("REDIS_URL")
redis_db = os.environ.get("REDIS_DB")
jwt_secret = os.environ.get("JWT_SECRET")

redis_host = redis_url.split("//")[1].split(":")[0]
redis_port = redis_url.split("//")[1].split(":")[1].split("/")[0]
limiter_storage_uri = f"redis://{redis_host}:{redis_port}/{redis_db}"
limiter = Limiter(key_func=get_remote_address, storage_uri=limiter_storage_uri)
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


@app.get("/")
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


@app.post("/generate_image/", response_model=None)
@limiter.limit(get_limit_for_user)
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
    # todo: return bytes from inference. Currently inference returns a base64encodedstring which can't be
    # streamed with a StreamingResponse because it is
    # not an iterator. so we must decode it and make it a file like object to stream it. Probably a better way to
    # do this is to send from inference the array of bytes directly and just resend them back to the web client without
    # additional need to decode the string into bytes.
    base64_image_string = await txt2img_runpod(runpod_run_input_data)
    # base64 decoding is cpu bound so even if it was async it would still block the event loop
    image_bytes = base64.b64decode(base64_image_string)
    # todo: maybe stream chunks of the response as they come in to avoid storing the whole image in memory
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
