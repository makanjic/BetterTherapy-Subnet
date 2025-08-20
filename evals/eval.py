import json

import aiohttp
import bittensor as bt
from openai import OpenAI


class OpenAILLMAsJudgeEval:
    def __init__(self, api_key, judge_model="gpt-4"):
        self.judge_client = OpenAI(api_key=api_key)
        self.judge_model = judge_model

    async def judge_responses(
        self, prompt: str, base_response: str, responses: list[str], config
    ) -> list[float]:
        """
        Use LLM-as-Judge to determine numerical scores for each miner's response compared to the base response.
        Returns a list of float scores (0-1).
        """
        numbered_responses = json.loads("\n".join(
            [
                f"Therapist {i + 1}: {resp if resp is not None else ''}"
                for i, resp in enumerate(responses)
            ]
        ))
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
        try:
            completion = self.judge_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict and fair judge for therapy responses.",
                    },
                    {"role": "user", "content": judge_prompt},
                ],
            )
            content = completion.choices[0].message.content
            if content is None:
                return [0.0] * len(responses)
            try:
                result = json.loads(content)
                scores = result.get("scores", [0.0] * len(responses))
                # Clamp scores
                scores = [max(0.0, min(1.0, float(s))) for s in scores]
                return scores
            except Exception as e:
                bt.logging.error(f"Error parsing judge JSON: {e}, content: {content}")
                return [0.0] * len(responses)
        except Exception as e:
            bt.logging.error(f"LLM judge error: {e}")
            if config.discord.webhook:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        config.discord.webhook,
                        json={
                            "content": f"Error in validator: {e}",
                        },
                    )
            return [0.0] * len(responses)
