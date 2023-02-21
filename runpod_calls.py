import asyncio
import httpx
import os
import logging
from exceptions import RunpodStatusUnexpectedError

logger = logging.getLogger("pre_inference")

runpod_status_url = os.environ.get("RUNPOD_STATUS_URL")
runpod_run_url = os.environ.get("RUNPOD_RUN_URL")
runpod_api_key = os.environ.get("RUNPOD_API_KEY")

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

            if http_status == 500 and num_errors < num_retries_on_error:
                num_errors += 1
                await asyncio.sleep(delay)

            elif http_status == 200:
                status = status_call_response.json()["status"]
                if status == "IN_PROGRESS":
                    logger.debug("IN_PROGRESS")
                    await asyncio.sleep(delay)
                elif status == "IN_QUEUE":
                    logger.debug("IN_QUEUE")
                    await asyncio.sleep(delay)
                elif status == "COMPLETED":
                    # logger.debug(f"status_url response: {status_call_response.json()}")
                    logger.debug("COMPLETED")
                    return status_call_response.json()["output"]
                elif status == "FAILED":
                    logger.debug("FAILED")
                    return None
                else:
                    logger.warning(f"http_status: {http_status}, unexpected response: {status_call_response.json()}")
                    raise RunpodStatusUnexpectedError(f'unexpected response: {status_call_response.json()}')

            else:
                logger.warning(f"unexpected http status: {http_status}, unexpected response: {status_call_response.json()}")
                raise RunpodStatusUnexpectedError(f'http_status: {http_status}, unexpected response: {status_call_response.json()}')
    except Exception as e:
        logger.debug(f"Unexpected error: {e}")
        if num_errors < num_retries_on_error:
            num_errors += 1
            return await runpod_status_output(run_id, num_errors)
        else:
            logger.warning(f"Too many errors: {num_errors}.")
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
            logger.warning(e)
            return None
        return base64_image_string
