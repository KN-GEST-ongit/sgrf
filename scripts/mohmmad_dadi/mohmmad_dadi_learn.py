import os
import cv2
import numpy as np
import pickle

from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH, TRAINED_MODELS_PATH
from bdgs.classifier import process_image
from scripts.common.crop_image import crop_image
from bdgs.models.image_payload import ImagePayload
from bdgs.data.algorithm import ALGORITHM
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def learn():
    processed_images = []
    etiquettes = []
    images = get_learning_files(limit=1000, shuffle=True)
    for image, hand_recognition_data, _ in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, image))
        hand_image = cv2.imread(image_path)
        processed_image = (process_image(
            algorithm=ALGORITHM.MOHMMAD_DADI,
            payload=ImagePayload(image=hand_image)
        )).flatten()
        processed_images.append(processed_image)
        etiquettes.append(int(hand_recognition_data.split(" ")[0]) - 1)

    X = np.array(processed_images)
    y = np.array(etiquettes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pca = PCA(n_components=50)  # PCA can be replaced by LDA
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)
    knn_accuracy = knn.score(X_test_pca, y_test)
    print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")

    svm = SVC(kernel='linear')
    svm.fit(X_train_pca, y_train)
    svm_accuracy = svm.score(X_test_pca, y_test)
    print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")

    model_path = os.path.join(TRAINED_MODELS_PATH, 'mohmmad_dadi_pca.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pca, f)
    model_path = os.path.join(TRAINED_MODELS_PATH, 'mohmmad_dadi_knn.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)
    model_path = os.path.join(TRAINED_MODELS_PATH, 'mohmmad_dadi_svm.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(svm, f)

if __name__ == "__main__":
    learn()
