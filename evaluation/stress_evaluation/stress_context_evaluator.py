import pandas as pd
df = pd.read_csv("stress_survey.csv")
df = df[['uid', 'datetime', 'timestamp', 'stress']]
df['binary_stress'] = df['stress'].apply(lambda x: 1 if x > 3 else 0)



df_high_stress = df[df['binary_stress'] == 1]
print(df_high_stress[df_high_stress['uid'] == 'u011'])

