# importing all necessary libarries
from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# app ki heading

st.write("""
# Explore differen ML models and datasets
Dekhty hain knsa best ha in main sai
""")

# dataset ka name aek box main daal kr side p laga do
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Iris", "Breast Cancer", "Wine")
)

# or esi k nechy classifier ka name aek dabby main daal do
classifier_nname = st.sidebar.selectbox(
    "Select Classifier",
    ("KNN", "SVM", "Random Forest")
)

# ab hum n aek function define krna hay dataset ko load krny k leye
def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x =  data.data
    y = data.target
    return x, y
# ab is function ko call krwa lain gy or X, y variable k equal rkh lain gay
X, y = get_dataset(dataset_name)

# ab hum apny dataset ki shape ko app pr print kr dain gay
st.write("Shape of dataset: ", X.shape)
st.write("Number of classes: ", len(np.unique(y)))

# next hum 3 classifier k parameter ko user input may add kr dain gay
def add_parameter_ui(Classifier_name):
    params = dict()     # create an empty dictionary
    if classifier_nname == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C     # its the degree of correct classification
    elif classifier_nname == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K     # its the number of nearest neighbours
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth     # depth of every tree that grew in random forest
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators       # number of trees
    return params

# ab es function ko call kr lain gay or params variable k equal rkh lain gay
params = add_parameter_ui(classifier_nname)

# ab hum classifier bnayen gay base on classifier_names and params
def get_classifer(classifier_name, params):
    clf = None
    if classifier_nname == "SVM":
        clf = SVC(C = params["C"])
    elif classifier_nname == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["K"])
    else:
        clf = RandomForestClassifier(n_estimators = params["n_estimators"], max_depth = params["max_depth"], random_state = 1234)
    return clf

# ab es function ko bula lain gay or clf variable k equal rkh lain gay
clf = get_classifer(classifier_nname, params)

# ab hum apny dataset ko split kr lain gay
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

# ab hum n apny classifier ki training krni hai
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# model ka accuracy score check kr laina hay or esy ap n print kr daina hay
acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_nname}")
st.write(f"Accuracy = {acc}")

### PLOT DATASET ###
# ab hum apny sary features of 2 dimensional plot p draw kr dain gay using pca
pca = PCA(2)
X_projected = pca.fit_transform(X)

# ab hum apna data 0 and 1 dimension main slice kr dain gay
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()

plt.scatter(x1, x2, c=y, alpha = 0.8, cmap = "viridis")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# plt show
st.pyplot(fig)




