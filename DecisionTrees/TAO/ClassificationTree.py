from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.datasets import load_iris
from numba import jitclass, boolean, int32, float32

'''
JitNode = [
    ('id', int32),
    ('depth', int32),
    ('left_node_id', int32),
    ('right_node_id', int32),
    ('feature', int32),
    ('threshold', float32),
    ('is_leaf', boolean),
    ('parent_id', int32),
    ('value', int32),
    ('data_idxs', int32[:]),
]
'''
#@jitclass(JitNode)
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


    #Dato un nodo ne restituisce uno nuovo con stessi attributi
    @staticmethod
    def copy_node(node):
        new = TreeNode(node.id, node.depth, node.left_node_id, node.right_node_id, node.left_node, node.right_node, node.feature, node.threshold, node.is_leaf, node.value)
        new.parent_id = node.parent_id
        new.data_idxs = node.data_idxs
        return new


class ClassificationTree:

    def __init__(self, min_samples=None):
        self.tree = {}
        self.tree_depth = -1
        self.min_samples = min_samples
        self.depth = None
        self.n_leaves = None

    #Crea l'albero iniziale usando CART
    def initialize_from_CART(self, data, label, clf):
        self.tree_depth = clf.max_depth
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
        self.build_idxs_of_subtree(data, range(len(data)), self.tree[0])



    #Costruisce la lista degli indici del dataset associati ad ogni nodo del sottoalbero
    def build_idxs_of_subtree(self, data, idxs, root_node):
        #Prima svuoto tutte le liste dei nodi del sottoalbero
        stack = [root_node]
        #Finchè non ho esplorato tutto il sottoalbero
        while(len(stack) > 0):
            actual_node = stack.pop()
            #Se il nodo attuale non è una foglia
            #print(actual_id)
            actual_node.data_idxs = []
            if not actual_node.is_leaf:
                #Svuoto la lista sua e dei figli
                actual_node.left_node.data_idxs = []
                actual_node.right_node.data_idxs = []
                stack.append(actual_node.left_node)
                stack.append(actual_node.right_node)

        #Guardando il path per ogni elemento del dataset aggiorno gli indici di
        #ogni nodo
        for i in idxs:
            path = self.get_path_to(data[i], root_node)
            for node in path:
                node.data_idxs.append(i)


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


    #Restituisce la lista degli id dei nodi appartenenti al percorso di decisione per x
    #nel sottoalbero con radice in root_node.
    def get_path_to(self, x, root_node):

        #Parto dalla root e definisco il percorso
        actual_node = root_node
        path = [actual_node]
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


    #Ritorna la lista degli id dei nodi alla profondità desiderata
    def get_nodes_at_depth(self, depth):
        nodes = []
        for (id, node) in self.tree.items():
            if node.depth == depth:
                nodes.append(node)
        return nodes


    #Predice la label degli elementi data nel sottoalbero con radice root_node
    def predict_label(self, data, root_node):

        predictions = []
        for x in data[:,]:
            #Se è una foglia ritorno il suo valore
            if root_node.is_leaf:
                predictions.append(root_node.value)
            else:
                path = self.get_path_to(x, root_node)
                leaf_node = path[-1]
                label = leaf_node.value
                predictions.append(label)
        return predictions


    #Restituisce id della foglia del sottoalbero con radice in root_node che predice x
    def predict_leaf(self, x, root_node):
        path = self.get_path_to(x, root_node)
        return path[-1].id


    #Restituisce la loss del sottoalbero con radice in root_node
    def misclassification_loss(self, data, target, root_node):
        #data = data[self.tree[root_id].data_idxs]
        #target = target[self.tree[root_id].data_idxs]
        if len(data) > 0:
            n_misclassified = np.count_nonzero(target-self.predict_label(data, root_node))
            return n_misclassified/len(data)
        else:
            return 0




#VISUALIZE THE TREE
#tree.plot_tree(clf)
#plt.show()
