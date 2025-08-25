import pandas as pd
df = pd.read_parquet('/data2/tairan/workspace/BatchGen/test/r1_mmlu_pro_test/mmlu_pro_test.parquet')
print(df.head())
print(df.columns)
print(df.shape)
