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


async def forward(self: validator.Validator):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    miner_uids = get_available_uids(self, k=256)

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

    base_query_response = generate_response(prompt_for_vali, self.model, self.tokenizer)
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

    bt.logging.info(f"Received total responses: {len(responses)}")

    rewards = get_rewards(self, prompt, base_response, responses=responses)
    responses_data = []
    full_rewards = []
    for resp, uid, reward in zip(responses, miner_uids, rewards, strict=False):
        if resp.output is None or resp.output == "" or reward is None or reward == 0:
            full_rewards.append(0)
            continue
        response_time_score = 0
        if reward > 0.2:
            if resp.dendrite.process_time < 10:
                response_time_score = 100
            elif resp.dendrite.process_time < 20:
                response_time_score = 50
            elif resp.dendrite.process_time < 30:
                response_time_score = 20
        response_time_score = response_time_score * 0.3  # 30% of the score
        quality_score = reward * 100 * 0.7  # 70% of the score
        total_score = response_time_score + quality_score
        full_rewards.append(total_score)
        responses_data.append(
            {
                "request_id": request_id,
                "miner_id": uid,
                "hotkey": self.metagraph.hotkeys[uid],
                "coldkey": self.metagraph.coldkeys[uid],
                "prompt": prompt,
                "response": resp.output,
                "base_response": base_response,
                "response_time": resp.dendrite.process_time,
                "response_time_score": response_time_score,
                "quality_score": quality_score,
                "total_score": total_score,
            }
        )
    if len(responses_data) > 0:
        self.wandb_logger.log_evaluation_round(prompt, request_id, responses_data)
        self.wandb_logger.create_summary_dashboard()
    else:
        bt.logging.warning(f"No responses received for request {request_id}")
    self.update_scores(np.array(full_rewards), miner_uids.tolist())
    time.sleep(10 * 60)  # every 5 minutes
