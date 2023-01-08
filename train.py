import os
import pickle as pk

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from utils import load_data, split_data,  compute_weight, create_model_age, create_model_gender

is_age = True
# load the dataset
X, y = load_data(csv_name="balanced_all.csv",is_age=is_age)
# split the data into training, validation and testing sets
data = split_data(X, y, test_size=0.1, valid_size=0.1)
# construct the model

sc = MinMaxScaler()
data["X_train"] = sc.fit_transform(data["X_train"])
data["X_test"] = sc.transform(data["X_test"])
data["X_valid"] = sc.transform(data["X_valid"])
pca = PCA(n_components=0.95)
data["X_train"] = pca.fit_transform(data["X_train"])
data["X_test"] = pca.transform(data["X_test"])
data["X_valid"] = pca.transform(data["X_valid"])

if not os.path.isdir("tools"):
    os.mkdir("tools")
if is_age:
    pk.dump(pca, open("tools/PCA_age.pkl", "wb"))
    pk.dump(sc, open("tools/MinMaxScaler_age.pkl", "wb"))
    model = create_model_age(data["X_train"].shape[1])
    age_weights = compute_weight(data["y_train"])
    tensorboard = TensorBoard(log_dir="logs")  # use tensorboard to view metrics
    early_stopping = EarlyStopping(mode="min", patience=10,
                                   restore_best_weights=True)  # early stop after 10 non-improving epochs

    batch_size = 400
    epochs = 100

    # train the model using the training set and validating using validation set
    history = model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size, class_weight=age_weights,
              validation_data=(data["X_valid"], data["y_valid"]), callbacks=[tensorboard, early_stopping])
    model.save("results/model_age.h5")
else:
    pk.dump(pca, open("tools/PCA_gender.pkl", "wb"))
    pk.dump(sc, open("tools/MinMaxScaler_gender.pkl", "wb"))
    model = create_model_gender(data["X_train"].shape[1])
# use tensorboard to view metrics
    age_weights = compute_weight(data["y_train"])
    tensorboard = TensorBoard(log_dir="logs")
    # define early stopping to stop training after 5 epochs of not improving
    early_stopping = EarlyStopping(mode="min", patience=15, restore_best_weights=True)
    batch_size = 64
    epochs = 100
    # train the model using the training set and validating using validation set
    history = model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size, class_weight=age_weights,
              validation_data=(data["X_valid"], data["y_valid"]))
    # save the model to a file
    model.save("results/model_gender.h5")

# evaluating the model using the testing set
print(f"Evaluating the model using {len(data['X_test'])} samples...")
loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Set figure size.
plt.figure(figsize=(12, 8))

# Generate line plot of training, testing loss over epochs.
#plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
plt.plot(val_accuracy, label='Validation Accuracy', color='orange')

# Set title
plt.title('Training Accuracy by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)
plt.xticks(range(0,100,5), range(0,100,5))

plt.legend(fontsize = 18)
plt.show()
