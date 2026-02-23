import pickle
from surprise import Dataset, SVD
from surprise.model_selection import train_test_split

# Load data 
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

# Train SVD model
model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
model.fit(trainset) # Save model
with open('models/svd_model.pkl', 'wb') as f:
    pickle.dump(model, f)