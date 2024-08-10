import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Determine the maximum length for padding
max_length = max(len(item) for item in data_dict['data'])

# Pad sequences to ensure uniform length
padded_data = pad_sequences(data_dict['data'], maxlen=max_length, padding='post', dtype='float32')

# Convert data and labels to NumPy arrays
data = np.asarray(padded_data)  # Use padded data
labels = np.asarray(data_dict['labels'])

# Print shapes for verification
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred,y_test)
print("Accuracy:", accuracy*100)
f=open('model.pickle','wb')
pickle.dump({'model':model},f)
f.close()