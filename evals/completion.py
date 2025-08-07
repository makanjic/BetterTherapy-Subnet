import openai


class SimpleOpenAICompletionFn:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def get_completion(self, prompt):
        """
        Generates a completion for a given prompt using the fine-tuned model.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a highly skilled, compassionate, and empathetic therapist specializing in mental health. "
                            "Your goal is to provide supportive, non-judgmental, and evidence-based responses that help users feel heard, understood, and empowered. "
                            "Always respond with warmth, validation, and curiosity. Ask gentle follow-up questions when appropriate, and encourage users to share more if they feel comfortable. "
                            "Avoid giving direct medical advice or making diagnoses. Focus on active listening, emotional support, and collaborative problem-solving."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
