import pickle

# Load from pickle file
with open('data.pkl', 'rb') as f:
    users = pickle.load(f)

print(users)
