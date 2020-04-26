
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.datasets import load_iris, load_digits
from DecisionTree import TreeNode, ClassificationTree
from algorithms import TAO, LocalSearch



dataset = load_iris()
data = dataset.data
label = dataset.target

n_trial = 5
val_split = 0.2
clf_train_score, tao_train_score, ls_train_score = 0, 0, 0
clf_valid_score, tao_valid_score, ls_valid_score = 0, 0, 0

#data = dataset.data[idx]
#label = dataset.target[idx]





for i in range(n_trial):
    data = np.load('cancer_train.npy')
    label = np.load('cancer_label.npy')


    idx = np.random.permutation(len(data))
    data = data[idx]
    label = label[idx]
    #data = dataset.data[idx]
    #label = dataset.target[idx]
    valid_id = int(len(data)*(1-val_split))
    X_train = data[0:valid_id]
    y_train = label[0:valid_id]

    X_valid = data[valid_id:]
    y_valid = label[valid_id:]




    clf = DecisionTreeClassifier(random_state=0, max_depth=3, min_samples_leaf=4)
    clf.fit(X_train, y_train)
    T = ClassificationTree(oblique = False)
    T.initialize_from_CART(X_train, y_train, clf)
    #tao_train_score+=1-T.misclassification_loss(X_train, y_train, T.tree[0])
    #print ("score before: ", tao_train_score)
#x = data[8]
#print (T.predict_label(x.reshape((1, -1)), 0))
#print (clf.predict(x.reshape((1, -1))))
#print ("x--->", x)
#print(T.get_path_to(x, 0))
#T.print_tree_structure()

#print ("T acc -> ", 1-T.misclassification_loss(data, label, T.tree[0]))
#print ("clf acc -> ", clf.score(data, label))
#node_id = 4
#x = data[T.tree[node_id].data_idxs]
#print (x)
#print (T.predict_label(x, node_id))
#for (id, node) in T.tree.items():
    #print("Prima: node ", id, " items -->", node.data_idxs)
    tao = TAO(T)
    tao.evolve(X_train, y_train)


    L = ClassificationTree(oblique = False)
    L.initialize_from_CART(X_train, y_train, clf)
    ls = LocalSearch(L)
    ls.evolve(X_train, y_train, alfa=1000000, max_iteration=10)
    clf_train_score+=clf.score(X_train, y_train)
    tao_train_score+=1-T.misclassification_loss(T.tree[0], X_train, y_train, range(len(X_train)), T.oblique)
    ls_train_score+=1-L.misclassification_loss(L.tree[0], X_train, y_train, range(len(X_train)), L.oblique)
    clf_valid_score+=clf.score(X_valid, y_valid)
    tao_valid_score+=1-T.misclassification_loss(T.tree[0], X_valid, y_valid, range(len(X_valid)), T.oblique)
    ls_valid_score+=1-L.misclassification_loss(L.tree[0], X_valid, y_valid, range(len(X_valid)), L.oblique)

#to_delete = ls.evolve(data, label)
#T.print_tree_structure()
#for (id, node) in T.tree.items():
    #print("Dopo: node ", id, " items -->", node.data_idxs)

print ("LS train acc -> ", ls_train_score/n_trial, "Depth: ", L.depth)
print ("TAO train acc -> ", tao_train_score/n_trial, "Depth: ", T.depth)
print ("CART train acc -> ", clf_train_score/n_trial)

L.print_tree_structure()
T.print_tree_structure()
print ("LS valid acc -> ", ls_valid_score/n_trial)
print ("TAO valid acc -> ", tao_valid_score/n_trial)
print ("CART valid acc -> ", clf_valid_score/n_trial)
#print(to_delete)
#print (lab)
#print (L.predict_label(test, 0))
#print (to_delete)
#VISUALIZE THE TREE
#tree.plot_tree(clf)
#plt.show()
