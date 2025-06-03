import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt


with open('model/sequence_data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

X = np.array(data_dict['data'])  # (num_samples, 30, 63)
y = np.array(data_dict['labels'])


lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)
pickle.dump(lb, open('label_binarizer.p', 'wb'))


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y)


model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 63)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=35, batch_size=16, validation_data=(X_test, y_test))


loss, acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {acc * 100:.2f}%')


model.save('model.keras')


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Gesture Classifier Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('training_accuracy_plot.png')