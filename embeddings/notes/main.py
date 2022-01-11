import pandas as pd

df = pd.read_csv("~/datasets/Reviews.csv", index_col=0)
df = df[['Time', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']]
df = df.dropna()
df['combined'] = "title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()

df = df.tail(100)

df.sort_values('Time').tail(1_100)
df.drop('Time', axis=1, inplace=True)

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# remove reviews that are too long
df['n_tokens'] = df.combined.apply(lambda x: len(tokenizer.encode(x)))
df = df[df.n_tokens<2000]


from openai.embeddings_utils import get_embedding

df['babbage_similarity'] = df.combined.apply(lambda x: get_embedding(x, engine='babbage-similarity'))
df['babbage_search'] = df.combined.apply(lambda x: get_embedding(x, engine='babbage-search-document'))
df.to_csv('~/datasets/embedded_100_reviews.csv')
