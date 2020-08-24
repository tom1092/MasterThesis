from DecisionTree import TreeNode, ClassificationTree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from algorithms import TAO, LocalSearch
from sklearn.svm import LinearSVC
import random
from joblib import Parallel, delayed
from trial import mutation

def genetic_tree_optimization(n_trees, n_iter, depth, X, labels, oblique, X_valid, y_valid, CR=0.95, l = 1):
    clf = RandomForestClassifier(n_estimators = n_trees, max_depth=depth, random_state=0, min_samples_leaf= 4)
    clf.fit(X, labels)
    random_trees = clf.estimators_
    trees = []
    best_loss = np.inf
    best_tree = None
    for tree in random_trees:
        T = ClassificationTree(oblique = oblique)
        T.initialize_from_CART(X, labels, tree)
        tao_opt(T, X, labels)
        trees.append(T)
        ClassificationTree.build_idxs_of_subtree(X, range(len(labels)), T.tree[0], oblique)
        #ClassificationTree.restore_tree(T)
    #multi_optimize_tao(trees, X, labels)

    best_loss = np.inf
    best_tree = None
    for i in range(n_iter):
        print("Iter: ", i)
        #multi_optimize_evolution(trees, X, labels, CR)

        for tree in trees:
            #optimize_evolution(tree, trees, X, labels, X_valid, y_valid, CR)
            trial = mutation(tree, trees, CR, X, labels, depth)
            tao_opt(trial, X, labels)
            trial_loss = regularized_loss(trial.tree[0], X, labels, X_valid, y_valid, range(len(labels)), oblique, l=l)
            loss = regularized_loss(tree.tree[0], X, labels, X_valid, y_valid, range(len(labels)), oblique, l=l)
            if trial_loss < loss:
                tree = trial
                #print("migliore")
            if loss < best_loss:
                best_loss = loss
                best_tree = tree
                print ("best loss: ", best_loss)
                print("loss train best: ", 1-ClassificationTree.misclassification_loss(best_tree.tree[0], X, labels, range(len(labels)), oblique))
                print("loss valid: ", 1-ClassificationTree.misclassification_loss(best_tree.tree[0], X_valid, y_valid, range(len(y_valid)), oblique))

    print("ritorno loss train best: ", 1-ClassificationTree.misclassification_loss(best_tree.tree[0], X, labels, range(len(labels)), oblique))
    print("ritono loss valid: ", 1-ClassificationTree.misclassification_loss(best_tree.tree[0], X_valid, y_valid, range(len(y_valid)), oblique))
    return best_tree, best_loss


def optimize_evolution(tree, trees, X, labels, X_valid, y_valid, CR):
    trial = ClassificationTree.copy_tree(tree)
    '''
    n_nodes = len(X[0])/(2**tree.depth - 1)
    if n_nodes < 1:
        n_nodes = 1


    p = np.random.uniform()
    for node in nodes:
        if p < CR and not node.is_leaf:
            node.feature = np.random.randint(0, len(X[0]))
            node.threshold = np.random.uniform(np.min(X[node.feature]),np.max(X[node.feature]))
    '''
    crossover(trial, trees, CR, X)
    ClassificationTree.build_idxs_of_subtree(X, range(len(labels)), trial.tree[0], oblique)
    #partners = random.sample(trees, 2)
    #optimize_crossover(trial, ClassificationTree.copy_tree(partners[0]), ClassificationTree.copy_tree(partners[1]), depth, X, labels, oblique)
    tao_opt(trial, X, labels)
    loss = regularized_loss(trial.tree[0], X, labels, X_valid, y_valid, range(len(labels)), oblique)
    #Se il nuovo individuo è migliore del padre li scambio
    if loss < regularized_loss(tree.tree[0], X, labels, X_valid, y_valid, range(len(labels)), oblique):
    #if loss < ClassificationTree.misclassification_loss(tree.tree[0], X, labels, range(len(labels)), oblique):
        tree = trial


def crossover(tree, trees, CR, X):
    partners = random.sample(trees, 3)
    partners = [ClassificationTree.copy_tree(partners[0]), ClassificationTree.copy_tree(partners[1]), ClassificationTree.copy_tree(partners[2])]
    tree_nodes = list(tree.tree.values())
    d = random.choice(range(1, depth))
    #for d in reversed(range(tree.depth-1)):
    nodes_at_depth = ClassificationTree.get_nodes_at_depth(d, tree)
    other_nodes_at_depth = ClassificationTree.get_nodes_at_depth(d, partners[0])
    #other_nodes_at_depth.extend(ClassificationTree.get_nodes_at_depth(d, partners[1]))
    #other_nodes_at_depth.extend(ClassificationTree.get_nodes_at_depth(d, partners[2]))
    #for node in nodes_at_depth:
    node = random.choice(nodes_at_depth)
    p = np.random.uniform()
    if p < CR:
        choice = random.choice(other_nodes_at_depth)

        if choice.left_node != None and not choice.left_node.is_leaf and choice.right_node != None and not choice.right_node.is_leaf:
            p2 = np.random.uniform()
            if p2 < 0.5:
                node.feature = choice.left_node.feature
                node.threshold = choice.left_node.threshold
                #ClassificationTree.replace_node(node, choice.left_node, tree)
            else:
                node.feature = choice.right_node.feature
                node.threshold = choice.right_node.threshold
                #ClassificationTree.replace_node(node, choice.right_node, tree)

        else:
            #ClassificationTree.replace_node(node, choice, tree)
            node.feature = choice.feature
            node.threshold = choice.threshold
        '''
        if choice.parent_id != -1:
            node.feature = tree.tree[choice.parent_id].feature
            node.threshold = tree.tree[choice.parent_id].threshold
        '''

    else:
        node.feature = random.choice(range(len(X[0])))
        node.threshold = 0



def multi_optimize_evolution(trees, X, labels, CR, depth):
    Parallel(n_jobs=8)(delayed(optimize_evolution)(tree, trees, CR, X, labels, depth) for tree in trees)


def multi_optimize_tao(trees, X, labels):
    Parallel(n_jobs=8)(delayed(tao_opt)(tree, X, labels) for tree in trees)


def tao_opt(tree, X, labels):
    tao = TAO(tree, train_on_all_features = True)
    tao.evolve(X, labels, min_size_prune = 1)


def optimize_crossover(tree_a, tree_b, tree_c, depth, X, labels, oblique):
    d = random.choice(range(depth))
    #for d in reversed(range(depth)):
    nodes_a = ClassificationTree.get_nodes_at_depth(d, tree_a)
    nodes_a = [node for node in nodes_a if not node.is_leaf]
    all_nodes = ClassificationTree.get_nodes_at_depth(d, tree_a)
    all_nodes.extend(ClassificationTree.get_nodes_at_depth(d, tree_b))
    all_nodes.extend(ClassificationTree.get_nodes_at_depth(d, tree_c))
    all_nodes = [node for node in all_nodes if not node.is_leaf]
    #print (node.id for node in nodes_a)
    #worst = nodes_a[np.argmax([ClassificationTree.misclassification_loss(node, X, labels, node.data_idxs, oblique) for node in nodes_a])]
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

    '''
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
            return impurity*(gini(node.left_node, X, labels, left_idxs, oblique, n_classes = 2) + gini(node.right_node, X, labels, right_idxs, oblique, n_classes = 2))
    else:
        return 0
    '''

    stack = [node]
    loss = 0
    while stack:
        actual = stack.pop()
        idxs = actual.data_idxs
        if len(idxs) > 0 and not actual.is_leaf:
            loss += (2 - len(idxs)/len(labels))*(1-np.sum([(np.count_nonzero(np.array(labels[idxs]) == c)/len(idxs))**2 for c in range(n_classes)]))
            stack.append(actual.left_node)
            stack.append(actual.right_node)
    return loss




#Loss 0/1 regolarizzata con gini index del sottoalbero
def regularized_loss(node, X, labels, X_valid, y_valid, idxs, oblique, n_classes = 2, l = 0.9):
    #print(len(node.data_idxs))
    return ClassificationTree.misclassification_loss(node, X, labels, idxs, oblique) + l*ClassificationTree.misclassification_loss(node, X_valid, y_valid, range(len(y_valid)), oblique)#+ l*gini(node, X, labels, idxs, oblique, n_classes)





def multi_test(ls_train, ls_test, svm_train, svm_test, random_train, random_test, cart_train, tao_train, global_train, cart_test, tao_test, global_test, n_trial):
    Parallel(n_jobs=n_trial)(delayed(test)(iter, ls_train, ls_test, svm_train, svm_test, random_train, random_test, cart_train, tao_train, global_train, cart_test, tao_test, global_test) for iter in range(n_trial))


def test(n_runs, ls_train, ls_test, svm_train, svm_test, random_train, random_test, cart_train, tao_train, global_train, cart_test, tao_test, global_test):
    for run in range(n_runs):
        depth = 3
        oblique = False
        n_trees = 200
        n_iter = 5
        data = np.load('cancer_train.npy')
        y = np.load('cancer_label.npy')
        print ("Run -> ", run)
        idx = np.random.permutation(len(data))
        data = data[idx]
        y = y[idx]
        train_split = 0.50
        valid_split = 0.75
        #data = dataset.data[idx]
        #label = dataset.target[idx]
        train_id = int(len(data)*train_split)
        valid_id = int(len(data)*valid_split)
        X = data[0:train_id]
        labels = y[0:train_id]

        X_valid = data[train_id:valid_id]
        y_valid = y[train_id:valid_id]

        X_test = data[valid_id:]
        y_test = y[valid_id:]

        #CART
        clf = DecisionTreeClassifier(random_state=0, max_depth=depth, min_samples_leaf=4)
        clf.fit(X, labels)


        #TAO
        T = ClassificationTree(oblique = oblique)
        T.initialize_from_CART(X, labels, clf)
        tao = TAO(T)
        tao.evolve(X, labels)

        T.print_tree_structure()
        #LS

        '''
        L = ClassificationTree(oblique = oblique)
        L.initialize_from_CART(X, labels, clf)
        ls = LocalSearch(L)
        ls.evolve(X, labels, alfa=1000000, max_iteration=10)
        '''


        #SVM
        svm = LinearSVC(tol=1e-6, max_iter=10000, dual=False)
        svm.fit(X, labels)

        #RandomForest
        random_for = RandomForestClassifier(n_estimators = n_trees, max_depth=depth, random_state=0, min_samples_leaf= 4)
        random_for.fit(X, labels)

        #Genetic
        best_t, best_loss = genetic_tree_optimization(n_trees, n_iter, depth, X, labels, oblique, X_valid, y_valid, CR = 0, l = 0)
        #best_t.print_tree_structure()

        best_t.print_tree_structure()
        #Train Score
        cart_train.append(clf.score(X, labels))
        #ls_train.append(1-ClassificationTree.misclassification_loss(L.tree[0], X, labels, range(len(labels)), oblique))
        tao_train.append(1-ClassificationTree.misclassification_loss(T.tree[0], X, labels, range(len(labels)), oblique))
        global_train.append(1-ClassificationTree.misclassification_loss(best_t.tree[0], X, labels, range(len(labels)), oblique))
        svm_train.append(svm.score(X, labels))
        random_train.append(random_for.score(X, labels))

        #Test Score
        cart_test.append(clf.score(X_test, y_test))
        #ls_test.append(1-ClassificationTree.misclassification_loss(L.tree[0], X_test, y_test, range(len(y_test)), oblique))
        tao_test.append(1-ClassificationTree.misclassification_loss(T.tree[0], X_test, y_test, range(len(y_test)), oblique))
        global_test.append(1-ClassificationTree.misclassification_loss(best_t.tree[0], X_test, y_test, range(len(y_test)), oblique))
        svm_test.append(svm.score(X_test, y_test))
        random_test.append(random_for.score(X_test, y_test))

ls_train, ls_test, svm_train, svm_test, random_train, random_test, cart_train, tao_train, global_train, cart_test, tao_test, global_test = [],[],[],[],[],[],[],[],[],[],[],[]
n_runs = 1

test(n_runs, ls_train, ls_test, svm_train, svm_test, random_train, random_test, cart_train, tao_train, global_train, cart_test, tao_test, global_test)
print()
print()
print("CART train:", np.mean(cart_train), " +- ", np.var(cart_train)**0.5)
#print("LS train:", np.mean(ls_train), " +- ", np.var(ls_train)**0.5)
print("TAO train: ", np.mean(tao_train), " +- ", np.var(tao_train)**0.5)
print("SVM train: ", np.mean(svm_train), " +- ", np.var(svm_train)**0.5)
print("RandomForest train: ", np.mean(random_train), " +- ", np.var(random_train)**0.5)
print("Genetic train: ", np.mean(global_train), " +- ", np.var(global_train)**0.5)
print()
print("/*--------*/")
print("   ----   ")
print()
print("CART test:", np.mean(cart_test), " +- ", np.var(cart_test)**0.5)
#print("LS test:", np.mean(ls_test), " +- ", np.var(ls_test)**0.5)
print("TAO test: ", np.mean(tao_test), " +- ", np.var(tao_test)**0.5)
print("SVM test: ", np.mean(svm_test), " +- ", np.var(svm_test)**0.5)
print("RandomForest test: ", np.mean(random_test), " +- ", np.var(random_test)**0.5)
print("Genetic test: ", np.mean(global_test)," +- ", np.var(global_test)**0.5)
