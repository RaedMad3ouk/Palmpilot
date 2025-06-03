import os
import numpy as np
import pickle

SEQUENCE_DATA_DIR = './data/sequences'  # directory with .npy sequence files
EXPECTED_SHAPE = (60, 63)  # <-- Update if your sequences change again

sequences = []
labels = []
label_names = []  # To keep string-to-index mapping
skipped_files = 0

# Sort gesture folders alphabetically
for class_label in sorted(os.listdir(SEQUENCE_DATA_DIR)):
    class_path = os.path.join(SEQUENCE_DATA_DIR, class_label)
    if not os.path.isdir(class_path):
        continue

    if class_label not in label_names:
        label_names.append(class_label)

    for seq_file in os.listdir(class_path):
        if not seq_file.endswith('.npy'):
            continue

        seq_path = os.path.join(class_path, seq_file)

        try:
            sequence = np.load(seq_path)

            # Skip invalid shapes
            if sequence.shape != EXPECTED_SHAPE:
                print(f"[WARNING] Skipped {seq_file} (shape: {sequence.shape})")
                skipped_files += 1
                continue

            sequences.append(sequence)
            labels.append(label_names.index(class_label))

        except Exception as e:
            print(f"[ERROR] Failed to load {seq_file}: {e}")
            skipped_files += 1

# Convert to numpy arrays
X = np.array(sequences)
y = np.array(labels)

print("Dataset loaded successfully.")
print("Total valid sequences:", len(X))
print("Total skipped files due to shape or error:", skipped_files)
print("Final shape:", X.shape)

# Save dataset
with open('sequence_data.pickle', 'wb') as f:
    pickle.dump({'data': X, 'labels': y}, f)

# Save label map (index to string)
with open('gesture_labels.pickle', 'wb') as f:
    pickle.dump(label_names, f)

print("Saved: sequence_data.pickle and gesture_labels.pickle")
