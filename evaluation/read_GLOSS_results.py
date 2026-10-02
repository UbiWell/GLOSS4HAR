import pickle

subject_id = 'pilot2'  # or whatever the actual ID is
with open(f'evaluation/GLOSS4HAR_results_{subject_id}.pkl', 'rb') as f:
    results = pickle.load(f)

# print the hours
print("Hours:", list(results.keys()))
idx = 9

print("Understanding for hour " + str(list(results.keys())[idx]), results[list(results.keys())[idx]]["answer"])
# print our all results[hours]["answer"]
# for hour in results:
#     print(f"Answer: {results[hour]['understanding']}")
