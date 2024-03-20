import cv2
import os
import numpy as np
import preprocessor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_data(folder):
    print("Pribiranje na podatocite...")
    dataset = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        if img is not None:
            data = np.array(preprocessor.preprocess(img))
            data = np.reshape(data, (901, 1))
            data = data.flatten()
            result = 1 if "genuine" in filename else 0
            # result = np.array(result)
            # result = np.reshape(result, (2, 1))
            dataset.append((data, result))
    return dataset


def get_accuracy(true_data, predictions):
    sum = 0
    for true, pred in zip(true_data, predictions):
        if true == pred:
            sum += 1
    acc = sum / len(true_data)
    return acc


def process_data(classifier, train_data, test_data):
    print("Procesiranje na podatocite...")
    train_x = [row[0] for row in train_data]
    train_y = [row[1] for row in train_data]
    test_x = [row[0] for row in test_data]
    test_y = [row[1] for row in test_data]
    classifier.fit(train_x, train_y)
    predictions_train = classifier.predict(train_x)
    predictions_test = classifier.predict(test_x)
    accuracy_train = get_accuracy(train_y, predictions_train)
    print(f"Tochnost nad trenirachkoto mnozhestvo: {accuracy_train}")
    accuracy_test = get_accuracy(test_y, predictions_test)
    print(f"Tochnost nad testirachkoto mnozhestvo: {accuracy_test}")
    if accuracy_train / accuracy_test > 1.15:
        print("Se sluchuva overfitting")
    else:
        print("Ne se sluchuva overfitting")
    return


def learn(author, current_dir):
    training_folder = os.path.join(current_dir, 'data/training/', author)
    test_folder = os.path.join(current_dir, 'data/test/', author)
    training_data = get_data(training_folder)
    test_data = get_data(test_folder)
    model = input("Izberete model od supervised learning za ispituvanje na verodostojnosta.\n"
                  "[Vnesete reden broj]\n"
                  "1. Nevronska mrezha\n"
                  "2. Naiven baesov klasifikator\n"
                  "3. Drvo na odluchuvanje\n"
                  "4. Shuma od drva na odluchuvanje\n")
    classifier = None
    if model == "1":
        classifier = MLPClassifier(hidden_layer_sizes=20, random_state=0, learning_rate_init=0.001, activation="relu",
                                   max_iter=150)
    elif model == "2":
        classifier = GaussianNB()
    elif model == "3":
        classifier = DecisionTreeClassifier(criterion="gini", random_state=0,)
    elif model == "4":
        classifier = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=0)
    else:
        print("Vnesete validen broj!")
        return
    process_data(classifier, training_data, test_data)
    inp = input("Vnesi slika od test folderot (na izbraniot avtor)\n")
    img = cv2.imread(os.path.join(test_folder, inp), 0)
    data = np.array(preprocessor.preprocess(img))
    data = np.reshape(data, (901, 1))
    data = data.flatten()
    prediction = classifier.predict([data])[0]
    if prediction == 0:
        print("Ovoj potpis e falsifikuvan")
    else:
        print("Ovoj potpis e verodostoen")
    print("Imajte vo predvid deka ovaa pretpostavka ne e celosno tochna i tochnosta na ovoj model ja pishuva pogore")
