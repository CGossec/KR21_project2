import copy
import os
from typing import List, Union, Dict, Optional, Tuple
import pandas as pd

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

    def min_degree(self) -> List[Tuple[str, int]]:
        """
        Ordering nodes on the basis of the smallest degree.

        :return: a list the variables and their degree ordered according to the heuristic
        """
        interaction_graph = self.bn.get_interaction_graph()
        degrees = sorted(list(interaction_graph.degree()), key=lambda x: x[1])
        return degrees

    def min_fill(self) -> List[Tuple[str, int]]:
        """
        Ordering nodes on the basis where elimination leads to smallest number of edges.

        :return: a list the variables and their degree ordered according to the heuristic
        """
        interaction_graph = self.bn.get_interaction_graph()
        added_edges = []
        for node in interaction_graph:
            count = 0
            for neighbor in interaction_graph.neighbors(node):
                for far_neighbor in interaction_graph.neighbors(neighbor):
                    if far_neighbor not in interaction_graph.neighbors(node):
                        count += 1
            added_edges.append((node, count))
        return sorted(added_edges, key=lambda x: x[1])

    def marginal_distributions(self, query: List[str], evidence: Optional[Dict[str, bool]], ordering: List[Tuple[str, int]]) -> pd.DataFrame:
        """
        Computes the marginal distribution of the given query w.r.t. evidence.

        :param query: List of nodes
        :param evidence: if None, this computes the prior marginal. Otherwise, this is a dictionary of values with
            respective truth assignments.
        :return: a pandas DataFrame containing the CPT of the given query
        """
        S = self.bn.get_all_cpts()
        factors = {}
        print([S[cpt] for cpt in [_ for _ in S]], "\n=======================================\n")
        for (node, _) in ordering:
            factor = None
            for variable in S:
                cpt = S[variable]
                if node in cpt.columns:
                    if factor is None:
                        factor = cpt
                        continue
                    if len(cpt) > len(factor):
                        tmp = copy.copy(factor)
                        tmp_var = tmp.columns[0]
                        factor = cpt
                        for _, row in tmp.iterrows():
                            truth_value = row[tmp_var]
                            factor[factor[tmp_var] == truth_value] *= row["p"]
                    else:
                        for _, row in cpt.iterrows():
                            truth_value = row[variable]
                            factor[factor[variable] == truth_value] *= row["p"]
            columns_to_keep = list(factor.columns)
            columns_to_keep.remove(node)
            columns_to_keep.remove("p")
            print(factor)
            summed_out_factor = pd.pivot_table(factor, index=columns_to_keep, values="p", aggfunc="sum")
            summed_out_factor = summed_out_factor.reset_index()
            print(summed_out_factor)
            factors[node] = summed_out_factor
            for variable in S:
                if node in S[variable].columns:
                    S[variable] = summed_out_factor
        return factors

if __name__ == "__main__":
    bifxml_path = os.getcwd() + "/testing/dog_problem.BIFXML"
    bnr = BNReasoner(bifxml_path)
    print(bnr.marginal_distributions(["dog-out"], None, bnr.min_fill()))
