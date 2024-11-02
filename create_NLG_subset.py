import pandas as pd
(pd.read_csv("../full_dataset.csv", nrows=100)).to_csv("NLG_subset.csv")