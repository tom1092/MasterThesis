from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.datasets import load_iris
from numba import jitclass, boolean, int32, float32


class TreeNode:
    def __init__(self, id, depth, left_node_id, right_node_id, left_node, right_node, feature, threshold, is_leaf, value):
        self.id = id
        self.depth = depth
        self.left_node_id = left_node_id
        self.right_node_id = right_node_id
        self.left_node = left_node
        self.right_node = right_node
        self.feature = feature
        self.threshold = threshold
        self.is_leaf = is_leaf
        self.parent_id = -1
        self.value = value
        self.data_idxs = []
        self.weights = None
        self.intercept = None


    #Dato un nodo ne restituisce uno nuovo con stessi attributi
    @staticmethod
    def copy_node(node):
        new = TreeNode(node.id, node.depth, node.left_node_id, node.right_node_id, node.left_node, node.right_node, node.feature, node.threshold, node.is_leaf, node.value)
        new.parent_id = node.parent_id
        new.data_idxs = node.data_idxs
        return new


class ClassificationTree:

    def __init__(self, min_samples=None, oblique = False):
        self.tree = {}
        self.min_samples = min_samples
        self.depth = None
        self.n_leaves = None
        self.oblique = oblique

    #Crea l'albero iniziale usando CART
    def initialize_from_CART(self, data, label, clf):
        self.depth = clf.max_depth
        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold
        value = clf.tree_.value

        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
                self.tree[node_id] = TreeNode(node_id, node_depth[node_id], children_left[node_id], children_right[node_id], None, None, feature[node_id], threshold[node_id], False, -1)
                if self.oblique:
                    ej = np.zeros(len(data[0]))
                    ej[feature[node_id]] = 1
                    self.tree[node_id].weights = ej
                    self.tree[node_id].intercept = -threshold[node_id]
            else:
                is_leaves[node_id] = True
                self.tree[node_id] = TreeNode(node_id, node_depth[node_id], -1, -1, None, None, feature[node_id], threshold[node_id], True, np.argmax(value[node_id]))

        #Imposto i padri ogni figlio
        for i in range(len(self.tree)):
            #Verifico se è un branch
            if self.tree[i].left_node_id != self.tree[i].right_node_id:
                #In tal caso setto i relativi padri
                self.tree[self.tree[i].left_node_id].parent_id = i
                self.tree[self.tree[i].right_node_id].parent_id = i
                self.tree[self.tree[i].id].left_node = self.tree[self.tree[i].left_node_id]
                self.tree[self.tree[i].id].right_node = self.tree[self.tree[i].right_node_id]

        #Costruisco indici elementi del dataset associati ad ogni nodo
        self.build_idxs_of_subtree(data, range(len(data)), self.tree[0], oblique = self.oblique)


    #Dato un albero ne crea uno nuovo con stessi parametri
    @staticmethod
    def copy_tree(tree):
        new = ClassificationTree(min_samples=tree.min_samples, oblique = tree.oblique)
        new.depth = tree.depth
        new.n_leaves = tree.n_leaves
        for (node_id, node) in tree.tree.items():
            new.tree[node_id] = TreeNode(node_id, node.depth, node.left_node_id, node.right_node_id, None, None, node.feature, node.threshold, node.is_leaf, node.value)
            new.tree[node_id].parent_id = node.parent_id
            new.tree[node_id].data_idxs = node.data_idxs
            new.tree[node_id].weights = node.weights
            new.tree[node_id].intercept = node.intercept

        #Ora che ho istanziato tutti i nodi vado a settare i puntatori ai figli per ogni nodo
        #Uso una BFS
        stack = [new.tree[0]]
        while stack:
            actual = stack.pop()
            if not actual.is_leaf:
                actual.left_node = new.tree[actual.left_node_id]
                actual.right_node = new.tree[actual.right_node_id]
                stack.append(actual.left_node)
                stack.append(actual.right_node)

        return new


    #Ritorna la lista dei nodi alla profondità desiderata
    @staticmethod
    def get_nodes_at_depth(depth, tree):
        nodes = []
        for (id, node) in tree.tree.items():
            if node.depth == depth:
                nodes.append(node)
        return nodes


    #Stampa la struttura dell'albero in modo interpretabile
    def print_tree_structure(self):
        print("The binary tree structure has %s nodes and has "
              "the following tree structure:"
              % len(self.tree))

        for i in self.tree.keys():
            if self.tree[i].is_leaf:
                print("%snode=%s is child of node %s. It's a leaf node." % (self.tree[i].depth * "\t", i, self.tree[i].parent_id))
            else:
                print("%snode=%s is child of node %s. It's a test node: go to node %s if X[:, %s] <= %s else to "
                      "node %s."
                      % (self.tree[i].depth * "\t",
                         i,
                         self.tree[i].parent_id,
                         self.tree[i].left_node_id,
                         self.tree[i].feature,
                         self.tree[i].threshold,
                         self.tree[i].right_node_id,
                         ))


    #Costruisce la lista degli indici del dataset associati ad ogni nodo del sottoalbero
    @staticmethod
    def build_idxs_of_subtree(data, idxs, root_node, oblique):
        #Prima svuoto tutte le liste dei nodi del sottoalbero
        stack = [root_node]

        #Finchè non ho esplorato tutto il sottoalbero
        while(len(stack) > 0):
            actual_node = stack.pop()
            #print(actual_node.id)
            #Se il nodo attuale non è una foglia
            #print(actual_id)
            actual_node.data_idxs = []
            if actual_node.left_node and actual_node.right_node:
                #Svuoto la lista sua e dei figli
                actual_node.left_node.data_idxs = []
                actual_node.right_node.data_idxs = []
                stack.append(actual_node.left_node)
                stack.append(actual_node.right_node)

        #Guardando il path per ogni elemento del dataset aggiorno gli indici di
        #ogni nodo
        for i in idxs:
            path = ClassificationTree.get_path_to(data[i], root_node, oblique)
            for node in path:
                node.data_idxs.append(i)


    #Restituisce la lista degli id dei nodi appartenenti al percorso di decisione per x
    #nel sottoalbero con radice in root_node.
    @staticmethod
    def get_path_to(x, root_node, oblique):

        #Parto dalla root e definisco il percorso
        actual_node = root_node
        path = [actual_node]
        if oblique:
            #Finchè non trovo una foglia
            while(not actual_node.is_leaf):
                #Decido quale sarà il prossimo figlio
                weights = actual_node.weights
                intercept = actual_node.intercept
                if np.dot(x, weights) + intercept <= 0:
                    actual_node = actual_node.left_node
                else:
                    actual_node = actual_node.right_node
                path.append(actual_node)
        else:
            #Finchè non trovo una foglia
            while(not actual_node.is_leaf):
                #Decido quale sarà il prossimo figlio
                feature = actual_node.feature
                thresh = actual_node.threshold
                if x[feature] <= thresh:
                    actual_node = actual_node.left_node
                else:
                    actual_node = actual_node.right_node
                path.append(actual_node)

        return path



    #Predice la label degli elementi data nel sottoalbero con radice root_node
    @staticmethod
    def predict_label(data, root_node, oblique):

        predictions = [root_node.value if root_node.is_leaf else ClassificationTree.get_path_to(x, root_node, oblique)[-1].value for x in data[:,]]

        return predictions


    #Restituisce id della foglia del sottoalbero con radice in root_node che predice x
    @staticmethod
    def predict_leaf(x, root_node):
        path = ClassificationTree.get_path_to(x, root_node)
        return path[-1].id


    #Restituisce la loss del sottoalbero con radice in root_node
    @staticmethod
    def misclassification_loss(root_node, data, target, indexes, oblique):
        #data = data[self.tree[root_id].data_idxs]
        #target = target[self.tree[root_id].data_idxs]
        if len(indexes) > 0:
            n_misclassified = np.count_nonzero(target[indexes]-ClassificationTree.predict_label(data[indexes], root_node, oblique))
            return n_misclassified/len(indexes)
        else:
            return 0


    #Rimette apposto la struttura dati a dizionario usando una DFS
    @staticmethod
    def restore_tree(tree):
        T = tree.tree
        root_node = T[0]
        T.clear()
        stack = [root_node]
        ids = []
        depth = 0
        leaves = []
        while(len(stack) > 0):
            actual_node = stack.pop()
            if actual_node.depth > depth:
                depth = actual_node.depth

            T[actual_node.id] = actual_node
            ids.append(actual_node.id)
            if not actual_node.is_leaf:
                stack.append(actual_node.right_node)
                stack.append(actual_node.left_node)
            else:
                leaves.append(actual_node.id)
        tree.depth = depth
        tree.n_leaves = len(leaves)
        return ids


    #Crea due nuove foglie al nodo, le ottimizza per maggioranza e ritorna il nuovo sottoalbero
    @staticmethod
    def create_new_children(node, X, y, max_id, feature, threshold, oblique = False, weights=None, intercept=None):

        node.is_leaf = False
        node.feature = feature
        node.threshold = threshold
        if oblique:
            node.weights = weights
            node.intercept = intercept
        #print(self.max_id)
        id_left = max_id+1
        id_right = max_id+2
        left_child_node = TreeNode(id_left, node.depth+1, -1, -1, None, None, None, None, True, None)
        right_child_node = TreeNode(id_right, node.depth+1, -1, -1, None, None, None, None, True, None)
        left_child_node.parent_id = node.id
        right_child_node.parent_id = node.id
        node.left_node_id = id_left
        node.right_node_id = id_right
        node.left_node = left_child_node
        node.right_node = right_child_node

        ClassificationTree.build_idxs_of_subtree(X, node.data_idxs, node, oblique)
        bins = np.bincount(y[left_child_node.data_idxs])
        best_class_left = -1
        best_class_right = -1
        if len(bins > 0):
            best_class_left = bins.argmax()
        bins = np.bincount(y[right_child_node.data_idxs])
        if len(bins > 0):
            best_class_right = bins.argmax()
        left_child_node.value = best_class_left
        right_child_node.value = best_class_right


    #Elimina il nodo da un albero
    @staticmethod
    def delete_node(node_id, tree):
        T = tree.tree
        stack = [node_id]
        while (len(stack) > 0):
            actual_node = stack.pop()
            if not T[actual_node].is_leaf:
                stack.append(T[actual_node].left_node_id)
                stack.append(T[actual_node].right_node_id)
            T.pop(actual_node)


    #Mette il sottoalbero con radice in node_B, al posto di quello con radice
    #in node_A
    @staticmethod
    def replace_node(node_A, node_B, tree):
        tree = tree.tree

        #Salvo id del padre di A per potergli dire chi è il nuovo figlio
        parent_A_id = node_A.parent_id
        if parent_A_id != -1:
            parent_A = tree[parent_A_id]
            #Occorre capire se A era figlio destro o sinistro
            if parent_A.left_node_id == node_A.id:
                parent_A.left_node_id = node_B.id
                parent_A.left_node = node_B

            elif parent_A.right_node_id == node_A.id:
                parent_A.right_node_id = node_B.id
                parent_A.right_node = node_B

        node_B.parent_id = parent_A_id

        #Rimetto apposto le depth nel sottoalbero con root node_B
        node_B.depth = node_A.depth
        stack = [node_B]
        while (len(stack) > 0):
            actual_node = stack.pop()
            if not actual_node.is_leaf:
                actual_node.left_node.depth = actual_node.depth + 1
                actual_node.right_node.depth = actual_node.depth + 1
                stack.append(actual_node.left_node)
                stack.append(actual_node.right_node)


    #Ritona la lista delle feature sulle quali avvengono gli split nell'albero
    @staticmethod
    def get_features(T):
        features = []
        stack = [T.tree[0]]
        while stack:
            actual = stack.pop()
            if not actual.is_leaf:
                features.append(actual.feature)
                stack.append(actual.left_node)
                stack.append(actual.right_node)
        return features




#VISUALIZE THE TREE
#tree.plot_tree(clf)
#plt.show()
