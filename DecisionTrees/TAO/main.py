
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.datasets import load_iris, load_digits
from ClassificationTree import TreeNode, ClassificationTree
from algorithms import TAO, LocalSearch
from sklearn.datasets import fetch_openml


dataset = load_digits()
idx = np.random.permutation(len(dataset.data))
data = dataset.data[idx]
label = dataset.target[idx]

# Load data from https://www.openml.org/d/554
#data = np.load('mnist_train.npy')
#label = np.load('mnist_label.npy')

clf = DecisionTreeClassifier(random_state=0, max_depth=4, min_samples_leaf=4)
clf.fit(data, label)
T = ClassificationTree()
T.initialize_from_CART(data, label, clf)

#x = data[8]
#print (T.predict_label(x.reshape((1, -1)), 0))
#print (clf.predict(x.reshape((1, -1))))
#print ("x--->", x)
#print(T.get_path_to(x, 0))
T.print_tree_structure()

print ("T acc -> ", 1-T.misclassification_loss(data, label, 0))
print ("clf acc -> ", clf.score(data, label))
#node_id = 4
#x = data[T.tree[node_id].data_idxs]
#print (x)
#print (T.predict_label(x, node_id))
#for (id, node) in T.tree.items():
    #print("Prima: node ", id, " items -->", node.data_idxs)
tao = TAO(T)
tao.evolve(data, label)


L = ClassificationTree()
L.initialize_from_CART(data, label, clf)
ls = LocalSearch(L)
to_delete = ls.evolve(data, label)
#T.print_tree_structure()
#for (id, node) in T.tree.items():
    #print("Dopo: node ", id, " items -->", node.data_idxs)

print ("LS acc -> ", 1-L.misclassification_loss(data, label, 0))
print ("TAO acc -> ", 1-T.misclassification_loss(data, label, 0))
print ("CART acc -> ", clf.score(data, label))

L.print_tree_structure()
#T.print_tree_structure()
test = data[50:51]
lab = label[50:51]
print(to_delete)
#print (lab)
#print (L.predict_label(test, 0))
#print (to_delete)
#VISUALIZE THE TREE
#tree.plot_tree(clf)
#plt.show()
