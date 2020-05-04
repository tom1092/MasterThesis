from DecisionTree import TreeNode, ClassificationTree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from algorithms import TAO
import random

def genetic_tree_optimization(n_trees, n_iter, depth, X, labels, oblique, X_valid, y_valid, CR=0.95, n_good=20):
    clf = RandomForestClassifier(n_estimators = n_trees, max_depth=depth, random_state=0, min_samples_leaf= 4)
    clf.fit(X, labels)
    random_trees = clf.estimators_
    trees = []
    for tree in random_trees:
        T = ClassificationTree(oblique = oblique)
        T.initialize_from_CART(X, labels, tree)
        ClassificationTree.build_idxs_of_subtree(X, range(len(labels)), T.tree[0], oblique)
        tao = TAO(T)
        tao.evolve(X, labels, min_size_prune = 1)
        trees.append(T)

    best_loss = np.inf
    best_tree = None

    for i in range(n_iter):
        print("Iter: ", i)
        for tree in trees:
            trial = ClassificationTree.copy_tree(tree)
            nodes = random.sample(list(trial.tree.values()), 1)
            p = np.random.uniform()
            for node in nodes:
                if p < CR and not node.is_leaf:
                    node.feature = np.random.randint(0, len(X[0]))
                    node.threshold = np.random.uniform(np.min(X[node.feature]),np.max(X[node.feature]))
            tao = TAO(tree)
            ClassificationTree.build_idxs_of_subtree(X, range(len(labels)), trial.tree[0], oblique)
            partners = random.sample(trees, 2)
            optimize_crossover(trial, ClassificationTree.copy_tree(partners[0]), ClassificationTree.copy_tree(partners[1]), depth, X, labels, oblique)
            tao = TAO(trial)
            tao.evolve(X, labels, min_size_prune = 1)
            loss = regularized_loss(trial.tree[0], X, labels, range(len(labels)), oblique)
            #loss = ClassificationTree.misclassification_loss(trial.tree[0], X_valid, y_valid, range(len(y_valid)), oblique)
            #Se il nuovo individuo è migliore del padre li scambio
            if loss < regularized_loss(tree.tree[0], X, labels, range(len(labels)), oblique):
            #if loss < ClassificationTree.misclassification_loss(tree.tree[0], X_valid, y_valid, range(len(y_valid)), oblique):
                tree = trial

            if loss < best_loss:
                best_loss = loss
                best_tree = tree
                print ("best loss: ", best_loss)
                print("loss train best: ", 1-ClassificationTree.misclassification_loss(best_tree.tree[0], X, labels, range(len(labels)), oblique))
                print("loss valid: ", 1-ClassificationTree.misclassification_loss(best_tree.tree[0], X_valid, y_valid, range(len(y_valid)), oblique))

    '''
    good_pop = []
    losses = []
    for tree in trees:
        loss = regularized_loss(tree.tree[0], X, labels, range(len(labels)), oblique)
        good_pop.append(tree)
        losses.append(loss)

    good_pop = np.array(good_pop)
    good_pop = good_pop[np.argsort(losses)]
    #print(type(np.argsort(losses)))
    #good_pop = good_pop[]
    good_pop = good_pop[0: n_good]
    #for tree in good_pop:
        #tao = TAO(tree)
        #print("Preoptimizing tree...")
        #tao.evolve(X, labels, min_size_prune = 1)

    best_loss = np.inf
    for i in range(n_iter):
        print("Iter: ", i)
        for tree in good_pop:
            partners = random.sample(trees, 2)
            optimize_crossover(tree, partners[0], partners[1], depth, X, labels, oblique)

            loss = regularized_loss(tree.tree[0], X, labels, range(len(labels)), oblique)
            #tao = TAO(tree)
            #tao.evolve(X, labels, min_size_prune = 1)
            if loss < best_loss:

                best_loss = loss
                #tao = TAO(tree)
                #tao.evolve(X, labels, min_size_prune = 1)
                #best_loss = ClassificationTree.misclassification_loss(tree.tree[0], X, labels, range(len(labels)), oblique)
                best_tree = tree
    #tao = TAO(best_tree)
    #tao.evolve(X, labels)
    '''
    print("ritorno loss train best: ", 1-ClassificationTree.misclassification_loss(best_tree.tree[0], X, labels, range(len(labels)), oblique))
    print("ritono loss valid: ", 1-ClassificationTree.misclassification_loss(best_tree.tree[0], X_valid, y_valid, range(len(y_valid)), oblique))
    return best_tree, best_loss


def optimize_crossover(tree_a, tree_b, tree_c, depth, X, labels, oblique):
    for d in reversed(range(depth)):
        nodes_a = ClassificationTree.get_nodes_at_depth(d, tree_a)
        nodes_a = [node for node in nodes_a if not node.is_leaf]
        all_nodes = ClassificationTree.get_nodes_at_depth(d, tree_a)
        all_nodes.extend(ClassificationTree.get_nodes_at_depth(d, tree_b))
        all_nodes.extend(ClassificationTree.get_nodes_at_depth(d, tree_c))
        all_nodes = [node for node in all_nodes if not node.is_leaf]
        #print (node.id for node in nodes_a)
        #worst = nodes_a[np.argmax([ClassificationTree.misclassification_loss(node, X, labels, node.data_idxs, oblique) for node in nodes_a])]

        node = random.choice(nodes_a)
        for node in nodes_a:
            if len(node.data_idxs) > 0:
                best = find_best_branch(all_nodes,  node.data_idxs, X, labels, oblique)
                best.data_idxs = node.data_idxs
                ClassificationTree.replace_node(node, best, tree_a)

        #best = find_best_branch(all_nodes, worst.data_idxs, X, labels, oblique)
        #best.data_idxs = worst.data_idxs
        #ClassificationTree.replace_node(worst, best, tree_a)



def find_best_branch(nodes, idxs, X, labels, oblique):
    losses = [regularized_loss(node, X, labels, idxs, oblique) for node in nodes]
    return nodes[np.argmin(losses)]


def gini(node, X, labels, idxs, oblique, n_classes = 2):

    if idxs:
        g= np.sum([(np.count_nonzero(np.array(y[idxs]) == c)/len(idxs))**2 for c in range(n_classes)])
        impurity = 1-g
        if node.is_leaf:
            return len(idxs)/len(labels) * impurity

        else:
            left_idxs = []
            right_idxs = []
            for i in idxs:
                if X[i, node.feature] < node.threshold:
                    left_idxs.append(i)
                else:
                    right_idxs.append(i)
            return len(idxs)/len(labels) * (impurity - len(left_idxs)/len(idxs)*gini(node.left_node, X, labels, left_idxs, oblique, n_classes = 2) - len(right_idxs)/len(idxs)*gini(node.right_node, X, labels, right_idxs, oblique, n_classes = 2))
    else:
        return 0



#Loss 0/1 regolarizzata con gini index del sottoalbero
def regularized_loss(node, X, labels, idxs, oblique, n_classes = 2, l = 0):
    #print(len(node.data_idxs))
    return ClassificationTree.misclassification_loss(node, X, labels, idxs, oblique) + l*gini(node, X, labels, idxs, oblique, n_classes)


depth = 3
oblique = False
n_trees = 10
n_iter = 10
data = np.load('cancer_train.npy')
y = np.load('cancer_label.npy')
cart_train, tao_train, global_train, cart_valid, tao_valid, global_valid = 0,0,0,0,0,0
n_trial = 3
for i in range(n_trial):
    idx = np.random.permutation(len(data))
    data = data[idx]
    y = y[idx]
    val_split = 0.35
    #data = dataset.data[idx]
    #label = dataset.target[idx]
    valid_id = int(len(data)*(1-val_split))
    X = data[0:valid_id]
    labels = y[0:valid_id]

    X_valid = data[valid_id:]
    y_valid = y[valid_id:]

    clf = DecisionTreeClassifier(random_state=0, max_depth=depth, min_samples_leaf=4)
    clf.fit(X, labels)
    T = ClassificationTree(oblique = oblique)
    T.initialize_from_CART(X, labels, clf)
    tao = TAO(T)
    tao.evolve(X, labels)
    best_t, best_loss = genetic_tree_optimization(n_trees, n_iter, depth, X, labels, oblique, X_valid, y_valid)
    cart_train += clf.score(X, labels)
    tao_train += 1-ClassificationTree.misclassification_loss(T.tree[0], X, labels, range(len(labels)), oblique)
    global_train += 1-ClassificationTree.misclassification_loss(best_t.tree[0], X, labels, range(len(labels)), oblique)
    cart_valid += clf.score(X_valid, y_valid)
    tao_valid += 1-ClassificationTree.misclassification_loss(T.tree[0], X_valid, y_valid, range(len(y_valid)), oblique)
    global_valid += 1-ClassificationTree.misclassification_loss(best_t.tree[0], X_valid, y_valid, range(len(y_valid)), oblique)

print("global score train: ", global_train/n_trial)
print("cart score train:", cart_train/n_trial)
print("cart + tao train: ", tao_train/n_trial)
print("global score valid: ", global_valid/n_trial)
print("cart score valid:", cart_valid/n_trial)
print("cart + tao valid: ", tao_valid/n_trial)
    #best_t.print_tree_structure()
