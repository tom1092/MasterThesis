from DecisionTree import TreeNode, ClassificationTree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from algorithms import TAO

def genetic_tree_optimization(n_trees, n_iter, depth, X, labels, oblique, CR):
    cart_trees = [DecisionTreeClassifier(random_state=0, max_depth=depth, min_samples_leaf=4) for i in range(n_trees)]
    trees = []
    for tree in cart_trees:
        print("Preoptimizing tree...")
        tree.fit(X, labels)
        T = ClassificationTree(oblique = oblique)
        T.initialize_from_CART(X, labels, tree)
        for (node_id, node) in T.tree.items():
            p = np.random.uniform()
            if p < CR and not node.is_leaf:
                node.feature = np.random.randint(0, len(X[0]))
                node.threshold = np.random.uniform(np.min(X[node.feature]),np.max(X[node.feature]))
        ClassificationTree.build_idxs_of_subtree(X, range(len(labels)), T.tree[0], oblique)
        tao = TAO(T)
        tao.evolve(X, labels)
        trees.append(T)

    best_loss = np.inf
    best_tree = None
    for i in range(n_iter):
        print("Iter: ", i)
        for tree in trees:
            partner = trees[np.random.randint(0, n_trees)]
            optimize_crossover(tree, partner, depth, X, labels, oblique)
            #tao = TAO(tree)
            #tao.evolve(X, labels)
            loss = ClassificationTree.misclassification_loss(tree.tree[0], X, labels, range(len(labels)), oblique)
            if loss < best_loss:
                best_loss = loss
                tao = TAO(tree)
                tao.evolve(X, labels)
                #best_loss = ClassificationTree.misclassification_loss(tree.tree[0], X, labels, range(len(labels)), oblique)
                best_tree = tree
    #tao = TAO(best_tree)
    #tao.evolve(X, labels)
    return best_tree, best_loss


def optimize_crossover(tree_a, tree_b, depth, X, labels, oblique):
    for d in reversed(range(depth)):
        nodes_a = ClassificationTree.get_nodes_at_depth(d, tree_a)
        nodes_a = [node for node in nodes_a if not node.is_leaf]
        all_nodes = ClassificationTree.get_nodes_at_depth(d, tree_a)
        all_nodes.extend(ClassificationTree.get_nodes_at_depth(d, tree_b))
        all_nodes = [node for node in all_nodes if not node.is_leaf]
        #print (node.id for node in nodes_a)
        #worst = nodes_a[np.argmax([ClassificationTree.misclassification_loss(node, X, labels, node.data_idxs, oblique) for node in nodes_a])]

        for node in nodes_a:
            best = find_best_branch(all_nodes, node.data_idxs, X, labels, oblique)
            best.data_idxs = node.data_idxs
            if best.feature != node.feature:
                ClassificationTree.replace_node(node, best, tree_a)

        #best = find_best_branch(all_nodes, worst.data_idxs, X, labels, oblique)
        #best.data_idxs = worst.data_idxs
        #ClassificationTree.replace_node(worst, best, tree_a)



def find_best_branch(nodes, idxs, X, labels, oblique):
    losses = [ClassificationTree.misclassification_loss(node, X, labels, idxs, oblique) for node in nodes]
    return nodes[np.argmin(losses)]


depth = 3
oblique = False
n_trees = 50
n_iter = 10
data = np.load('cancer_train.npy')
y = np.load('cancer_label.npy')
#idx = np.random.permutation(len(data))
#data = data[idx]
#y = y[idx]
val_split = 0.2
#data = dataset.data[idx]
#label = dataset.target[idx]
valid_id = int(len(data)*(1-val_split))
X = data[0:valid_id]
labels = y[0:valid_id]

X_valid = data[valid_id:]
y_valid = y[valid_id:]

clf = DecisionTreeClassifier(random_state=0, max_depth=depth, min_samples_leaf=4)
clf.fit(X, labels)
best_t, best_loss = genetic_tree_optimization(n_trees, n_iter, depth, X, labels, oblique, 0.95)
print("best score train: ", 1-best_loss)
print("cart score train:", clf.score(X, labels))
print("best score valid: ", 1-ClassificationTree.misclassification_loss(best_t.tree[0], X_valid, y_valid, range(len(y_valid)), oblique))
print("cart score valid:", clf.score(X_valid, y_valid))
best_t.print_tree_structure()
