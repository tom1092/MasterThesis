from DecisionTree import TreeNode, ClassificationTree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import spearmanr, pearsonr



def zero_one_loss(node, data, targets):
    loss = 0
    #Per ogni punto del nodo verifico dove questo lo inoltrerebbe
    #se il punto viene inoltrato su un figlio che porta a misclassificazione
    #è errore
    predictions = ClassificationTree.predict_label(data, node, False)
    n_misclassified = np.count_nonzero(targets-predictions)
    return n_misclassified


def best_split(node, X, labels):
    error_best =  np.inf
    j = node.feature

    vals = {}
    best_node = None
    for point_idx in node.data_idxs:
        vals[point_idx] = X[point_idx, j]

    values = sorted(vals.items(), key=lambda x: x[1])
    sorted_indexes = [tuple[0] for tuple in values]
    #print ("len sort indx: ", len(sorted_indexes))
    thresh = 0.5*X[sorted_indexes[0], j]
    node.threshold = thresh
    ClassificationTree.create_new_children(node, X, labels, node.id, j, thresh, False)

    actual_loss = zero_one_loss(node, X, labels)
    #Ciclo su ogni valore di quella componente e provo tutti gli split
    #possibili
    i = 0
    while i < len(sorted_indexes):
        if i < len(sorted_indexes)-1:
            thresh = 0.5*(X[sorted_indexes[i], j]+X[sorted_indexes[i+1], j])
        else:
            thresh = 1.5*X[sorted_indexes[i], j]
        node.threshold = thresh
        ClassificationTree.create_new_children(node, X, labels, node.id, j, thresh, False)

        actual_loss = zero_one_loss(node, X, labels)
        if actual_loss < error_best:
            error_best = actual_loss
            best_left = node.left_node
            best_right = node.right_node
            best_t = thresh
        i+=1

    return best_t, best_left, best_right



#X = np.random.uniform(size = (500, 10))
#labels = np.random.randint(0, high=2, size= (500, ))
data = np.load('banknote_train.npy')
y = np.load('banknote_label.npy')
clf_train, clf_valid, spear_train, spear_valid = 0, 0, 0, 0
for i in range(1):
    idx = np.random.permutation(len(data))
    data = data[idx]
    y = y[idx]
    val_split = 0.2
    #data = dataset.data[idx]
    #label = dataset.target[idx]
    valid_id = int(len(data)*(1-val_split))
    X = data[0:valid_id]
    labels = y[0:valid_id]

    X_valid = data[valid_id:]
    y_valid = y[valid_id:]
    depth = 3
    to_optimize = []
    node = TreeNode(0, 0, None, None, None, None, 0, 0, 0, None)
    node.data_idxs = range(len(X))
    to_optimize.append(node)
    while to_optimize:
        actual_node = to_optimize.pop()
        if actual_node.depth <= depth-1 and len(actual_node.data_idxs) > 0:
            actual_node.is_leaf = False
            try:
                best_feature = np.nanargmax([np.abs(spearmanr(X[actual_node.data_idxs, j], labels[actual_node.data_idxs])[0]) for j in range(len(X[0]))])
            except:
                best_feature = np.random.randint(0, len(X[0]))
            #print("feature: ", best_feature)
            #print("actual node ", actual_node.data_idxs)
            actual_node.feature = best_feature
            actual_node.threshold, actual_node.left_node, actual_node.right_node = best_split(actual_node, X, labels)

            #ClassificationTree.build_idxs_of_subtree(X, actual_node.data_idxs, actual_node, False)
            to_optimize.append(actual_node.left_node)
            to_optimize.append(actual_node.right_node)


    stack = [node]
    while stack:
        actual_node = stack.pop()
        if actual_node.is_leaf:
            print("%snode=%s is child of node %s. It's a leaf node. Value: %s" % (actual_node.depth * "\t", actual_node.id, actual_node.parent_id, actual_node.value))
        else:
            print("%snode=%s is child of node %s. It's a test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (actual_node.depth * "\t",
                     actual_node.id,
                     actual_node.parent_id,
                     actual_node.left_node_id,
                     actual_node.feature,
                     actual_node.threshold,
                     actual_node.right_node_id,
                     ))
            stack.append(actual_node.left_node)
            stack.append(actual_node.right_node)


    spear_train+=1-zero_one_loss(node, X, labels)/len(labels)
    spear_valid+=1-zero_one_loss(node, X_valid, y_valid)/len(y_valid)

    clf = DecisionTreeClassifier(random_state=0, max_depth=3, min_samples_leaf=4)
    clf.fit(X, labels)
    clf_train += clf.score(X, labels)
    clf_valid += clf.score(X_valid, y_valid)

    L = ClassificationTree(oblique = False)
    L.initialize_from_CART(X, labels, clf)

    L.print_tree_structure()
print("clf train: ", clf_train/30)
print("spearman train: ", spear_train/30)
print("clf valid: ", clf_valid/30)
print("spearman valid: ", spear_valid/30)
