import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pandas.plotting import scatter_matrix


def load_dataset(filename="./Dry_Bean_Dataset.csv"):
    """
    :param filename: Path to Bean Data set csv file.
    :return: Pandas Dataframe containing the Dry Bean attributes.
    """
    out = []
    try:
        with open(filename) as f:
            reader = csv.reader(f, delimiter=",", quotechar="\"")
            for row in reader:
                out.append(row)
        out = np.array(out)
        df = pd.DataFrame(out[1:, :], columns=out[0, :])
        return df

    except IOError:
        print(f"File: {filename} not found.")
        return 0


def show_variance_components(data):
    """
    Shows the variance graph of how including more features increases information.

    :param data: Dataframe containing Labeled features
    :return: None, shows matplotlib graph showing how increasing the number of included attributes does/does not
    increase information gained.
    """

    pca = PCA().fit(data)

    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (12, 6)

    fig, ax = plt.subplots()
    xi = np.arange(1, 17, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0, 1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 17, step=1))  # change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('Number of components to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')

    ax.grid(axis='x')
    plt.show()


def split_train_test(dataset):
    """
    Split the dataset in to a train and test set
    :param dataset:
    :return:
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=11)
    for train_index, test_index in split.split(d, d["Class"]):
        train_set = d.loc[train_index]
        test_set = d.loc[test_index]

    return train_set, test_set


def split_target(d):
    return d.drop("Class", axis=1), d[["Class"]]


def do_one_hot(d):
    """
    Applies onehot encoding to the Class feature of the input dataframe d.
    :param d: Dataframe of dataset
    :return: One hot encoded class feature
    """
    one_hot = OneHotEncoder()
    bean_class_oh = one_hot.fit_transform(d[["Class"]])
    return bean_class_oh


def do_cross_validation(mlp_, X, y):
    scores = cross_val_score(mlp_, X, y, cv=10)
    print(scores)


def do_grid_search(X, y):
    """
    Runs a grid search to identify improved parameters for MLP Classifier
    :return: mlp with optimal parameters found applied
    """
    parameters = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    mlp_ = MLPClassifier(hidden_layer_sizes=100, random_state=11, max_iter=700, verbose=False)
    clf = GridSearchCV(mlp_, parameters, n_jobs=-1, cv=3)
    clf.fit(X, y)
    print(clf.best_params_)
    return clf


def do_scatter_matrix_plot(d_):
    """
    :param d: Dataset dataframe. e.g Dry Bean Dataframe
    :return: None, displays scatter plot
    """
    attributes = ["Area","Perimeter","MajorAxisLength","MinorAxisLength","AspectRation","Eccentricity","ConvexArea",
                  "EquivDiameter","Extent","Solidity","roundness","Compactness","ShapeFactor1","ShapeFactor2",
                  "ShapeFactor3","ShapeFactor4"]
    scatter_matrix(d_[attributes], figsize=(12, 8))


def do_confusion_matrix(mlp_, X, y_true):
    plot_confusion_matrix(mlp_, X, y_true)
    plt.show()


def do_accuracy_score(mlp_, X, y):
    p_test = mlp_.predict(X)
    ac_s = accuracy_score(y, p_test)
    return ac_s


if __name__ == "__main__":
    d = load_dataset("Dry_Bean_Dataset.csv")

    train, test = split_train_test(d)
    train_x, train_y = split_target(train)
    test_x, test_y = split_target(test)

    train_x = train_x.astype(float)
    train_y = np.ravel(train_y)

    num_pipeline = Pipeline([
        ("min_max_scaler", MinMaxScaler()),
        #("pca", PCA(n_components=0.95))
    ])
    train_x_tr = num_pipeline.fit_transform(train_x)
    test_x_tr = num_pipeline.fit_transform(test_x)

    # do_grid_search(train_x_tr, train_y)

    mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50), random_state=11, max_iter=700, verbose=False,
                        activation='tanh', solver='adam', learning_rate='constant', alpha=0.0001)
    mlp.fit(train_x_tr, train_y)
    print(do_accuracy_score(mlp, test_x_tr, test_y)) #0.9103929489533603

    # do_confusion_matrix(mlp, train_x_tr, train_y)
    # do_confusion_matrix(mlp, test_x_tr, test_y)
