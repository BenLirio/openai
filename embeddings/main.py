import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.Engine(id="ada-search-query").embeddings(
        input="The food was delicious and the waiter..."
))
