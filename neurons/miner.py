# The MIT License (MIT)
# Copyright © 2025 BetterTherapy

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import os
import time
import datetime
import typing

import bittensor as bt
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

# Bittensor Miner Template:
import BetterTherapy

# import base miner class which takes care of most of the boilerplate
from BetterTherapy.base.miner import BaseMinerNeuron
# from BetterTherapy.utils.llm import generate_response

import json


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)  # noqa: UP008
        self.setup_model()
        self.datadir = os.path.expanduser(self.config.datadir)
        self.temperature = os.getenv("OPENAI_TEMPERATURE", 0.7)
        bt.logging.info(f"Miner initialized with uid: {self.uid}")
        bt.logging.info(f"Data directory: {self.datadir}")
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)

    def setup_model(self):
        self.model_name = self.config.model.name
        self.model = OpenAI()
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # self.model.eval()
        # if torch.cuda.is_available():
        #     self.model.to("cuda")

    async def forward(
        self, synapse: BetterTherapy.protocol.InferenceSynapse
    ) -> BetterTherapy.protocol.InferenceSynapse:
        """
        Handles InferenceSynapse requests.
        """
        bt.logging.info(f"Forwarding request: {synapse}")
        if not synapse.dendrite.hotkey:
            bt.logging.error("Missing hotkey in dendrite.")
            return synapse

        validator_dir = os.path.join(self.datadir, synapse.dendrite.hotkey)
        if not os.path.exists(validator_dir):
            os.makedirs(validator_dir)

        synapse_file = os.path.join(validator_dir, synapse.request_id)
        if not os.path.exists(synapse_file):
            with open(synapse_file, "w") as f:
                json.dump({
                    "request_id": synapse.request_id
                    "prompt": synapse.prompt,
                    "output": ""
                }, f)
        else:
            mod_time = os.path.getmtime(synapse_file)
            mod_datetime = datetime.datetime.fromtimestamp(mod_time)
            now = datetime.datetime.now()
            threshold = now - datetime.timedelta(minutes=5)
            if mod_datetime < threshold:
                with open(synapse_file, "w") as f:
                    json.dump({
                        "request_id": synapse.request_id,
                        "prompt": synapse.prompt,
                        "output": ""
                    }, f)
            else:
                waited = 0
                while waited < 15 * 60:
                    content = ""
                    with open(synapse_file, "r") as f:
                        content = f.read()
                    if content.strip() != "":
                        try:
                            content_json = json.loads(content.strip())
                            request_id = content_json.get("request_id", "")
                            output = content_json.get("output", "")
                            if request_id != synapse.request_id:
                                break
                            if output.strip() != ""
                                synapse.output = output.strip()
                                return synapse
                        except:
                            break
                    time.sleep(5)
                    waited += 5

        output = self.generate_response(synapse.prompt)
        # output = generate_response(synapse.prompt, self.model, self.tokenizer, "miner")
        synapse.output = output

        with open(synapse_file, "w") as f:
            json.dump({
                "request_id": synapse.request_id,
                "prompt": synapse.prompt,
                "output": synapse.output
            }, f)

        return synapse

    def generate_response(self, prompt: str) -> typing.Optional[str]:
        """
        Generate a response to the prompt.
        """
        try:
            response = self.model.chat.completions.create(
                model=self.model_name,
                # model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a highly skilled, compassionate, and empathetic therapist specializing in mental health. "
                            "Your goal is to provide supportive, non-judgmental, and evidence-based responses that help users feel heard, understood, and empowered. "
                            "Always respond with warmth, validation, and curiosity. Ask gentle follow-up questions when appropriate, and encourage users to share more if they feel comfortable. "
                            "Avoid giving direct medical advice or making diagnoses. Focus on active listening, emotional support, and collaborative problem-solving."
                            
                            "RESPONSE Rules:\n"
                            "- Keep your response around 50-60 words\n"
                            "- Stay focused and concise while maintaining empathy prioritizing emotional validation and support"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            bt.logging.error(f"An error occurred in generate_response: {e}")
            return None

    async def blacklist(
        self, synapse: BetterTherapy.protocol.InferenceSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        registered = synapse.dendrite.hotkey in self.metagraph.hotkeys
        if self.config.blacklist.allow_non_registered and not registered:
            return False, "Allowing un-registered hotkey"
        elif not registered:
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: BetterTherapy.protocol.InferenceSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        registered = synapse.dendrite.hotkey in self.metagraph.hotkeys
        if not registered:
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
