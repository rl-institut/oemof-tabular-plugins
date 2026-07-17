import pandas as pd

df = pd.read_csv("mixer.csv", sep=";")

v = df.loc[0, "input_flow_share_config"]

print(type(v))
print(repr(v))