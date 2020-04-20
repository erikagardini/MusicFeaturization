"""Save training and validation loss/accuracy curves from history".

References:
[1] Bahuleyan, H.
    Music Genre Classification using Machine Learning Techniques.
    arXiv:1804.01149
[2] https://github.com/HareeshBahuleyan/music-genre-classification
"""
import pickle
from matplotlib import pyplot as plt
import pandas as pd

# Load scores
with open('../models/history.pkl', 'rb') as f:
    scores = pickle.load(f, encoding='latin1')
print(scores.keys())
scores = pd.DataFrame(scores, index=range(1, 21))

plt.xticks(range(1,21))
plt.plot(scores['loss'], marker='o', label='training_loss')
plt.plot(scores['val_loss'], marker='d', label='validation_loss')
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Training Epochs', fontsize=12)
plt.grid()
plt.legend()
plt.savefig('../results/loss.svg')
plt.close()

plt.xticks(range(1,21))
plt.plot(scores['categorical_accuracy'], marker='o', label='training_accuracy')
plt.plot(scores['val_categorical_accuracy'], marker='d', label='validation_accuracy')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Training Epochs', fontsize=12)
plt.grid()
plt.legend()
plt.savefig('../results/accuracy.svg')