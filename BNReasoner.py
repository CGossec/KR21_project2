import copy
import os
from typing import List, Union

from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def pruning(self, X: List[str], Y: List[str], Z: List[str]) -> BayesNet:
        """
        Given a graph G and disjoint set of nodes X, Y, Z, return a pruned graph G' such that:
            * There are no leaf nodes that do not belong to X, Y, Z
            * All edges outgoing from nodes in Z are deleted

        :param X: List of nodes
        :param Y: List of nodes
        :param Z: List of nodes
        :return: Copy of self.net, with appropriate nodes and edge removed
        """
        pruned_graph = copy.copy(self.bn)
        modified = True
        while modified:
            modified = False
            for var in pruned_graph.get_all_variables():
                if pruned_graph.get_children(var) == [] and var not in X and var not in Y and var not in Z:
                    pruned_graph.del_var(var)
                    modified = True
            for var in Z:
                for child in pruned_graph.get_children(var):
                    pruned_graph.del_edge((var, child))
                    modified = True
        return pruned_graph


if __name__ == "__main__":
    bifxml_path = os.getcwd() + "/testing/dog_problem.BIFXML"
    bnr = BNReasoner(bifxml_path)
    bnr.bn.draw_structure()
    pruned = bnr.pruning(["light-on"], ["family-out"], ["dog-out"])
    pruned.draw_structure()
