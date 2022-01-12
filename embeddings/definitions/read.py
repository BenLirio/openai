import pandas as pd

for w in range(5):
    for x in range(5):
        for y in range(5):
            for z in range(5):
                df = pd.read_pickle(f"/home/ben/datasets/dict_word_embeddings_{w}_{x}_{y}_{z}.pkl")
                print("==============================================")
                for word in list(df.columns):
                    print(word)
                input()


