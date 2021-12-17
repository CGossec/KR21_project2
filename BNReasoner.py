import copy
import os
from typing import List, Union, Dict, Optional, Tuple
import pandas as pd

from BayesNet import BayesNet
import networkx as nx
import matplotlib.pyplot as plt


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
            print(net)

    def determine_reach(self, reachable_nodes, seen):
        new_reachable_nodes = list(set(reachable_nodes) - set(seen))
        for node in new_reachable_nodes:
            for child in self.bn.get_children(node):
                if child not in reachable_nodes:
                    reachable_nodes.append(child)
            seen.append(node)
        if list(set(reachable_nodes) - set(seen)) != []:
            #Recursive part: Find the children of the children of the current set
            reachable_nodes = self.determine_reach(reachable_nodes, seen)
        return reachable_nodes

    def d_separation(self , X, Y, Z):
        #Prune the graph
        all_variables = (self.bn.get_all_variables())

        for variable in all_variables:
            if variable not in (X + Y +Z) and self.bn.get_children(variable) == []:
                self.bn.del_var(variable)
        for variable in Z:
            children = self.bn.get_children(variable)
            for child in children:
                self.bn.del_edge([variable, child])
        #Find the reachable nodes from X
        X_reachables = []
        for variable in X:
            reachable_nodes = []
            seen = []
            reachable_nodes.extend(self.bn.get_children(variable))
            seen.extend(variable)
            reachable_nodes = self.determine_reach(reachable_nodes, seen)
            X_reachables.append(reachable_nodes)
        for x_reach in X_reachables:
            for y in Y:
                if y in x_reach:
                    print(X, 'and', Y, 'are not d-separated by', Z)
                    return False
        print(X, 'and', Y, 'are d-separated by', Z)
        return True

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
                    indices_to_drop = cpt[cpt[given] == (not evidence[given])].index
                    new_cpt = cpt.drop(indices_to_drop)
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


    def marginal_distributions(self, query: List[str], evidence: Optional[Dict[str, bool]], ordering: List[Tuple[str, int]]) -> Dict[str, pd.DataFrame]:
        """
        Computes the marginal distribution of the given query w.r.t. evidence.

        :param query: List of nodes
        :param evidence: if None, this computes the prior marginal. Otherwise, this is a dictionary of values with
            respective truth assignments.
        :return: a dictionary matching the variables of the query with their respective CPTs as pd.DataFrame
        """
        if evidence is None:
            evidence = {}
        pruned_graph = self.pruning(bnr.bn.get_all_variables(), evidence=evidence)
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
                    S[variable] = multiply_factors(factor, cpt)
            columns_to_keep = list(factor.columns)
            columns_to_keep.remove(node)
            columns_to_keep.remove("p")
            if columns_to_keep != []:
                summed_out_factor = pd.pivot_table(factor, index=columns_to_keep, values="p", aggfunc="sum")
                summed_out_factor = summed_out_factor.reset_index()
            else:
                summed_out_factor = factor
            factors[node] = summed_out_factor
            for variable in S:
                if S[variable].empty:
                    continue
                if S[variable].columns[-2] == node:
                    S[variable] = pd.DataFrame()
                    continue
                if node in S[variable].columns:
                    S[variable] = sum_out_variable(S[variable].drop(node, axis=1), variable)
        return {var: self.normalize_with_evidence(factors[var], evidence) for var in query}

    def normalize_with_evidence(self, cpt: pd.DataFrame, evidence: Dict[str, bool]) -> pd.DataFrame:
        res = cpt.copy()
        for var in evidence:
            evidence_cpt = self.bn.get_cpt(var)
            evidence_cpt = sum_out_variable(evidence_cpt, var)
            proba_of_evidence = float(evidence_cpt[evidence_cpt[var] == evidence[var]]["p"])
            res["p"] = res["p"] / proba_of_evidence
        return res


def multiply_factors(cpt1: pd.DataFrame, cpt2: pd.DataFrame) -> pd.DataFrame:
    res = cpt2 if len(cpt2) > len(cpt1) else cpt1
    other = cpt1 if len(cpt2) > len(cpt1) else cpt2
    for var in other.columns[:-1]:
        for _, row in other.iterrows():
            truth_value = row[var]
            res.loc[res[var] == truth_value, "p"] *= row["p"]
    return res

def sum_out_variable(cpt: pd.DataFrame, variable: str) -> pd.DataFrame:
    res = pd.DataFrame({variable: [True, False], "p": [0, 0]})
    for _, row in cpt.iterrows():
        res.loc[res[variable] == row[variable], "p"] += row["p"]
    return res

if __name__ == "__main__":
    bifxml_path = os.getcwd() + "/testing/lecture_example2.BIFXML"
    bnr = BNReasoner(bifxml_path)
    #print(bnr.marginal_distributions(["C"], {}, [('A', 123123), ('B', 123), ('C', 123)]))
    manual_order = [('J', 1), ('I', 2), ('Y', 3), ('X', 4), ('O', 5)]
    print(bnr.marginal_distributions('O', {}, manual_order))
