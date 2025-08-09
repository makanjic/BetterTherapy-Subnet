import json

from openai import OpenAI
from .utils import count_words
import bittensor as bt


class OpenAIBatchLLMAsJudgeEval:
    def __init__(self, api_key, judge_model="gpt-4"):
        self.judge_client = OpenAI(api_key=api_key)
        self.judge_model = judge_model
        self.base_response = None

    def create_judge_prompt(
        self, prompt: str, base_response: str, responses: list[str]
    ) -> list[float]:
        """
        Use LLM-as-Judge to determine numerical scores for each miner's response compared to the base response.
        Returns a list of float scores (0-1).
        """
        numbered_responses = "\n".join(
            [
                f"Therapist {i + 1}: {resp if resp is not None else ''}"
                for i, resp in enumerate(responses)
            ]
        )
        judge_prompt = (
            "You are an expert evaluator. Given the following prompt, the base response, and a set of therapist responses, "
            "score each therapist's response on a scale from 0 to 1. "
            "If response is None or empty, score it as 0."
            "A score of 0.7 means the response is as good as the base response. Score higher if the response is better, lower if worse. "
            "Reply in the following format (JSON):\n"
            '{"scores": [score1, score2, ...]}\n\n'
            f"Prompt: {prompt}\n"
            f"Base Response: {base_response}\n\n"
            f"Therapist Responses:\n{numbered_responses}\n\n"
            "What are the scores for each response? (Output JSON only)"
        )

        return judge_prompt

    def create_batch(
        self,
        prompt: str,
        base_response: str,
        request_id: str,
        responses: list[str],
        miner_uids: list[int],
    ) -> list[dict]:
        """
        Create batches of requests for the LLM judge.
        Each batch will not exceed 5500 words total.
        Returns a list of batches, where each batch is a list of request dicts.
        """
        batch = []
        current_batch_responses = []
        current_batch_miner_uids = []
        current_word_count = 0
        max_words_per_batch = 5500
        batch_metadata = {}
        request_number = 1
        for i, (response, miner_uid) in enumerate(zip(responses, miner_uids)):
            response_word_count = count_words(response.output) if response.output else 0
            if (current_word_count + response_word_count > max_words_per_batch) or (
                current_word_count and i == min(len(responses) - 1, len(miner_uids) - 1)
            ):
                custom_id = f"{request_id}_{request_number}"
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.judge_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a strict and fair judge for therapy responses.",
                            },
                            {
                                "role": "user",
                                "content": self.create_judge_prompt(
                                    prompt, base_response, current_batch_responses
                                ),
                            },
                        ],
                        "max_tokens": 1000,
                    },
                }
                batch.append(request)
                batch_metadata[custom_id] = ",".join(
                    map(lambda x: str(x), current_batch_miner_uids)
                )
                current_batch_responses = []
                current_batch_miner_uids = []
                current_word_count = 0
                current_word_count = response_word_count
                request_number += 1

            else:
                current_batch_miner_uids.append(miner_uid)
                current_batch_responses.append(response.output)
                current_word_count += response_word_count
        return batch, batch_metadata

    def queue_batch(self, batch: list[dict], batch_metadata: dict):
        with open("batchinput.jsonl", "w", encoding="utf-8") as f:
            for obj in batch:
                json_str = json.dumps(obj, ensure_ascii=False)
                f.write(json_str + "\n")

        batch_input_file = self.judge_client.files.create(
            file=open("batchinput.jsonl", "rb"), purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        queue_response = self.judge_client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=batch_metadata,
        )
        bt.logging.info(f"Batch queued with ID: {queue_response.id}")
        return queue_response

    def query_batch(self, batch_id: str):
        batch = self.judge_client.batches.retrieve(batch_id)
        if batch.status == "completed":
            file_response = self.judge_client.files.content(batch.output_file_id)
            openai_batch_responses = file_response.text.splitlines()
            return openai_batch_responses, batch
        else:
            bt.logging.info(
                f"Error in openai batch processing with status:: {batch.status}\n Error: {batch.errors.to_json() if batch.errors else 'Unknown error'}"
            )
            return None, None
