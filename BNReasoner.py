import copy
import os
from typing import List, Union, Dict

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

    def pruning(self, query: List[str], evidence: Dict[str, bool]) -> BayesNet:
        """
        Given a graph G and disjoint set of nodes in the query and an evidence, return a pruned graph G' such that:
            * There are no leaf nodes that do not belong to Query + evidence
            * All edges outgoing from nodes in evidence are deleted

        :param query: List of nodes
        :param evidence: Dictionary of variables with their assigned truth values
        :return: Copy of self.net, with appropriate nodes and edge removed, and CPTs updated
        """
        pruned_graph = copy.copy(self.bn)
        modified = True
        while modified:
            modified = False
            for var in pruned_graph.get_all_variables():
                if pruned_graph.get_children(var) == [] and var not in query and var not in evidence:
                    pruned_graph.del_var(var)
                    modified = True
            CPTs = pruned_graph.get_all_cpts()
            for given in evidence:
                for variable in CPTs:
                    cpt = CPTs[variable]
                    if given not in cpt.columns:
                        continue
                    if cpt.columns[-2] == given:
                        continue
                    indices_to_drop = cpt[cpt[given] == (not evidence[given])].index
                    new_cpt = cpt.drop(indices_to_drop)
                    new_cpt = new_cpt.drop(given, axis = 1)
                    pruned_graph.update_cpt(variable, new_cpt)
                for child in pruned_graph.get_children(given):
                    pruned_graph.del_edge((given, child))
                    modified = True
        return pruned_graph

    def marginal_distributions(self, query: List[str], evidence: Optional[Dict[str, bool]]) -> pd.DataFrame:
        """
        Computes the marginal distribution of the given query w.r.t. evidence.

        :param query: List of nodes
        :param evidence: if None, this computes the prior marginal. Otherwise, this is a dictionary of values with
            respective truth assignments.
        :return: a pandas DataFrame containing the CPT of the given query
        """
        pass


if __name__ == "__main__":
    bifxml_path = os.getcwd() + "/testing/dog_problem.BIFXML"
    bnr = BNReasoner(bifxml_path)
    print(bnr.bn.get_all_cpts())
    pruned = bnr.pruning(["light-on", "family-out"], {"dog-out" : False})
    print(pruned.get_all_cpts())
