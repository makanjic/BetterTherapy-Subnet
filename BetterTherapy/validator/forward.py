# The MIT License (MIT)
# Copyright © 2025 BetterTherapy

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time

import bittensor as bt
import numpy as np
import ulid

from BetterTherapy.protocol import InferenceSynapse
from BetterTherapy.utils.llm import generate_response
from BetterTherapy.utils.uids import get_available_uids
from BetterTherapy.validator.reward import get_rewards
from neurons import validator
import traceback
from BetterTherapy.db.query import (
    get_ready_requests,
    add_request,
    add_bulk_responses,
    delete_requests,
)
from BetterTherapy.db.models import MinerResponse
import json


async def forward(self: validator.Validator):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    try:
        miner_uids = get_available_uids(self, 256)
        # The dendrite client queries the network.
        prompt_for_vali = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>  
    You are a compassionate mental health assistant.  
    Generate both a mental health question and its empathetic answer.  
    Respond **only** with a VALID JSON object that:  
    • Begins with `{` and ends with `}`  
    • Contains exactly two keys: "question" and "answer"  
    • Includes no additional text, comments, or formatting  
    Example:{"question":"<mental health question>","answer":"<empathetic answer>"}  
    <|eot_id|>  
    <|start_header_id|>assistant<|end_header_id|>{ 
    """

        base_query_response = generate_response(
            prompt_for_vali, self.model, self.tokenizer
        )

        prompt = base_query_response.get("question", None)
        base_response = base_query_response.get("answer", None)
        if not prompt or not base_response:
            bt.logging.error(f"Invalid response format: {base_query_response}")
            return

        request_id = "btai_" + ulid.new().str
        bt.logging.info(f"Request ID: {request_id}")
        bt.logging.info(f"Prompt: {prompt}")
        bt.logging.info(f"Miner UIDs: {miner_uids}")
        bt.logging.info(
            f"Base Response: {base_response[:50] if len(base_response) > 50 else base_response}..."
        )

        responses = await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=InferenceSynapse(prompt=prompt, request_id=request_id),
            deserialize=True,
            timeout=500,
        )
        bt.logging.info(
            f"Received total responses: {len(responses)}, batching them and queueing them to openai"
        )
        if responses:
            batch_reponses, batch_metadata = self.batch_evals.create_batch(
                prompt, base_response, request_id, responses, miner_uids.tolist()
            )
            openai_batch_response = self.batch_evals.queue_batch(
                batch=batch_reponses, batch_metadata=batch_metadata
            )
            new_request = add_request(
                name=request_id,
                openai_batch_id=openai_batch_response.id,
                prompt=prompt,
                base_response=base_response,
            )
            miner_responses = []
            for resp, miner_uid in zip(responses, miner_uids.tolist()):

                miner_responses.append(
                    MinerResponse(
                        request_id=new_request.id,
                        miner_id=miner_uid,
                        response_text=resp.output,
                        response_time=resp.dendrite.process_time,
                    )
                )
            add_bulk_responses(responses=miner_responses)

        ready_requests = get_ready_requests()
        if ready_requests:
            processed_request_ids = []
            miner_scores = {}
            bt.logging.info(
                f"Found {len(ready_requests)} requests ready for processing."
            )
            for req in ready_requests:
                judged_responses = []
                miner_db_response = {
                    item.miner_id: {
                        "response_text": item.response_text,
                        "response_time": item.response_time,
                    }
                    for item in req.responses
                }

                bt.logging.info(
                    f"Processing batch request {req.openai_batch_id} created at {req.created_at} with prompt: {req.prompt}"
                )
                openai_batch, batch_info = self.batch_evals.query_batch(
                    batch_id=req.openai_batch_id
                )
                if openai_batch:
                    processed_request_ids.append(req.id)
                    for eval in openai_batch:
                        parsed_eval = json.loads(eval.strip())
                        custom_id = parsed_eval.get("custom_id", "")
                        if (
                            custom_id
                            and parsed_eval.get("response", "").get("status_code")
                            == 200
                        ):
                            parsed_miners = batch_info.metadata[custom_id].split(",")
                            miner_evaluation = (
                                parsed_eval.get("response", "")
                                .get("body")
                                .get("choices")[0]
                                .get("message")
                                .get("content")
                            )
                        try:
                            result = json.loads(miner_evaluation.strip())
                            scores = result.get(
                                "scores",
                                [0.0] * len(parsed_miners),
                            )
                            for score, miner_uid in zip(
                                scores,
                                parsed_miners,
                            ):
                                response_time_score = 0
                                miner_uid = int(miner_uid)
                                bounded_score = max(0.0, min(1.0, float(score)))
                                if bounded_score == 0.0:
                                    total_score = 0.0
                                elif bounded_score > 0.2:
                                    if miner_db_response.get(miner_uid) is None:
                                        bt.logging.warning(
                                            f"No response found for miner {miner_uid} in request {req.name}"
                                        )
                                        continue
                                    if (
                                        miner_db_response.get(miner_uid).get(
                                            "response_time", 500
                                        )
                                        < 10
                                    ):
                                        response_time_score = 100
                                    elif (
                                        miner_db_response.get(miner_uid).get(
                                            "response_time", 500
                                        )
                                        < 20
                                    ):
                                        response_time_score = 50
                                    elif (
                                        miner_db_response.get(miner_uid).get(
                                            "response_time", 500
                                        )
                                        < 30
                                    ):
                                        response_time_score = 20
                                    total_score = (
                                        bounded_score * 100 * 0.7
                                        + response_time_score * 0.3
                                    )
                                judged_responses.append(
                                    {
                                        "request_id": req.name,
                                        "miner_id": miner_uid,
                                        "hotkey": self.metagraph.hotkeys[miner_uid],
                                        "coldkey": self.metagraph.coldkeys[miner_uid],
                                        "prompt": req.prompt,
                                        "response": miner_db_response[miner_uid].get(
                                            "response_text", ""
                                        ),
                                        "base_response": req.base_response,
                                        "response_time": miner_db_response[
                                            miner_uid
                                        ].get("response_time", 500),
                                        "response_time_score": response_time_score,
                                        "quality_score": bounded_score * 100,
                                        "total_score": total_score,
                                    }
                                )

                                miner_scores[miner_uid] = (
                                    miner_scores.get(miner_uid, 0.0) + total_score
                                )
                        except Exception as e:
                            bt.logging.error(
                                f"Error parsing judge JSON: {e}, content: {parsed_eval}"
                            )
                            bt.logging.error(traceback.format_exc())
                if judged_responses:
                    self.wandb_logger.log_evaluation_round(
                        prompt, req.name, judged_responses
                    )
                    self.wandb_logger.create_summary_dashboard()
                else:
                    bt.logging.warning(f"No responses received for request {req.name}")

            if miner_scores:
                rewarded_miner_ids = list(miner_scores.keys())
                reward_scores = np.array(list(miner_scores.values()))
                self.update_scores(reward_scores, rewarded_miner_ids)
                bt.logging.info(
                    f"Updated scores for miners: keys: {rewarded_miner_ids}, values: {reward_scores.tolist()}"
                )
                delete_requests(request_ids=processed_request_ids)
                bt.logging.info(
                    f"Deleted processed requests with IDs: {processed_request_ids}"
                )
    except Exception as e:
        bt.logging.error(f"Error in forward pass: {e}")
        bt.logging.error(traceback.format_exc())
    finally:
        time.sleep(1 * 60 * 60)  # every 1 hr
