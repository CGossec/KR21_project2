import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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
                for variable in pruned_graph.get_all_cpts():
                    cpt = pruned_graph.get_all_cpts()[variable]
                    if given not in cpt.columns:
                        continue
                    cpt.loc[cpt[given] == (not evidence[given]), "p"] = 0
                    pruned_graph.update_cpt(variable, cpt)
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
            neighbors = interaction_graph.neighbors(node)
            for neighbor in neighbors:
                for elm in neighbors:
                    if neighbor is elm:
                        continue
                    if elm not in interaction_graph.neighbors(neighbor):
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
        for elm in query:
            for tup in ordering:
                if elm in tup:
                    ordering.remove(tup)
        if evidence is None:
            evidence = {}
        pruned_graph = self.pruning(bnr.bn.get_all_variables(), evidence=evidence)
        S = pruned_graph.get_all_cpts().copy()
        factors = {}
        # print([S[cpt] for cpt in S], "\n=======================================\n")
        for (node, _) in ordering:
            factors_to_mult = []
            indices = []
            for index, variable in enumerate(S):
                cpt = S[variable]
                if node in cpt.columns:
                    factors_to_mult.append(cpt)
                    indices.append(index)
            factor = multiply_factors(factors_to_mult)
            factors[node] = sum_out_variable(factor, node)
            for index in indices:
                S[list(S.keys())[index]] = factors[node]
        final_factor = multiply_factors(list(factors.values()))
        return self.normalize_with_evidence(final_factor, evidence, factors)

    def normalize_with_evidence(self, cpt: pd.DataFrame, evidence: Dict[str, bool], factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        For every value in the result, we normalize according to the given evidence to reflect proper probabilities.

        :param cpt: the dataframe we normalize. Modification is not done inplace.
        :param evidence: A dictionary of the variables and their assigned truth values
        :param factors: the factors we computed. I'm not sure whether this could be changed or not.
        :return: A new CPT, normalized according to the evidence.
        """
        res = cpt.copy()
        for var in evidence:
            evidence_cpt = self.bn.get_cpt(var).copy()
            if np.array([col in evidence_cpt.columns[:-1] for col in np.array(res.columns[:-1])]).any():
                continue
            proba_of_evidence = float(evidence_cpt[evidence_cpt[var] == evidence[var]]["p"])
            res["p"] = res["p"] / proba_of_evidence
        return res

    def reduce_cpts(self, cpts, evidence):
        if not evidence:
            return cpts

        for cpt in cpts:
            for ev in evidence:
                if ev in cpts[cpt]:
                    for i in range(len(cpts[cpt].index)-1,-1,-1):
                        # cpts[cpt] = cpts[cpt][cpts[cpt].columns[ev] != evidence[ev]]
                        if cpts[cpt].loc[i,ev] == (not evidence[ev]):
                            cpts[cpt] = cpts[cpt].drop(labels=i, axis=0)
                            cpts[cpt] = cpts[cpt].reset_index(drop=True)
            # self.bn.update_cpt(cpt, cpts[cpt])
        return cpts

    def map(self, variables, evidence):
        res, cpts = {} , self.bn.get_all_cpts()
        cpts = self.reduce_cpts(cpts, evidence)
        pos_marg = self.marginal_distributions(variables, evidence, self.min_fill())
        idx = pos_marg['p'].idxmax()
        print(variables)
        print(pos_marg)
        for var in pos_marg:
            if var in variables:
                res[var] = pos_marg.iloc[idx][var]
        return res

    def mpe(self, evidence):
        result = evidence.copy()
        self.bn = self.pruning(self.bn.get_all_variables(),evidence)
        el_order = self.min_fill()
        el_order = [x for (x,_) in el_order]
        cpts = self.reduce_cpts(self.bn.get_all_cpts(), evidence)
        cpts_l = [ cpts[x] for x in cpts]
        factors = multiply_factors(cpts_l)
        factors = self.reduce_cpts({'Factors':factors}, evidence)['Factors']
        factors_copy = factors.copy()
        prob = None
        for i in el_order:
            factors_copy, prob = max_out_variable(factors_copy, i)
        sol = factors.loc[factors['p'] == prob]
        for var in sol:
            if var is not 'p':
                result[var] = sol.iloc[0][var]
        return result


def create_cpt(vars: List[str]) -> pd.DataFrame:
    """
    Creates a CPT with proba 1 for every value.
    The CPT contains all possible variable instantiations.

    :param vars: List of variables to instantiate in the CPT
    :return: A joint CPT for all possible variable instantiations, every row has proba 1
    """
    res = None
    for var in vars:
        if res is None:
            res = pd.DataFrame({var: [True, False], "p": [1, 1]})
            continue
        res = pd.DataFrame(list(res.values) * 2, columns=res.columns)
        res[var] = [True] * (len(res) // 2) + [False] * (len(res) // 2)
        columns = list(res.columns[-1:]) + list(res.columns[:-1])
        res = res[columns]
    return res

def max_out_variable(cpt: pd.DataFrame, variable: str) -> pd.DataFrame:

    def find_complementary_row(cpt: pd.DataFrame, entry_row: pd.Series, index_to_switch: int) -> Tuple[pd.Series, int]:
        complementary_values = list(entry_row)[:-1]
        complementary_values[index_to_switch] = not complementary_values[index_to_switch]
        row_matches_conditions = [row.all() # every condition needs to evaluate to true
            for row in np.array( # We use numpy so can transpose
                [cpt[cpt.columns[i]] == complementary_values[i] # Compare cpt[SOME_COLUMN] with complementary_value[SOME_COLUMN]
                for i in range(len(complementary_values))] # for every column that we want to match
                ).T]
        try:
            index = row_matches_conditions.index(True)
            complementary_row = cpt.iloc[index]
            return complementary_row, index
        except:
            return None, None

    #print(f"Summing out variable {variable} from \n{cpt}")
    res = cpt.copy()
    cols = list(res.columns)
    var_to_remove = cols.index(variable)
    indices_to_drop = []
    sol = None
    for ii, row in res.iterrows():
        # print("ii, row", ii, row)
        if row[variable] == True:
            opposite_row, index = find_complementary_row(cpt, row, var_to_remove)
            if opposite_row is not None:
                res.loc[ii, "p"] = max(res.loc[ii, "p"], opposite_row["p"])
                sol = res.loc[ii, "p"]
                indices_to_drop.append(index)
    if not sol:
        sol = res.loc[ii, "p"]
    res = res.drop(indices_to_drop)
    res = res.reset_index(drop=True)
    if res.columns[-2] != variable:
        res = res.drop(columns=[variable])
    #print(f"End result is: \n{res}\n====================================")
    return res, sol
def multiply_factors(factors: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Multiplies CPTs, taking the common variables and multiplying the values where it can.

    :param factors: A list of all the factors to multiply with each other
    :return: a CPT containing the multiplication result of all CPTs
    """
    res = factors[0]
    for factor in factors[1:]:
        res = res.merge(factor, how="outer")
    res = res.drop(columns=["p"])
    columns = list(res.columns)
    res = create_cpt(columns)
    keys = list(res.columns[:-1])
    for row_index, row in res.iterrows():
        for i in range(len(factors)):
            cols_to_match = list(factors[i].columns[:-1])
            cond = [row.all() for row in np.array([[factors[i][col] == row[col]] for col in cols_to_match]).T]
            try:
                index = cond.index(True)
            except:
                index = None
            if index is not None:
                res.loc[row_index, "p"] *= factors[i].iloc[index]["p"]
    return res

def sum_out_variable(cpt: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Given a CPT, we want to remove a variable from it by summing all values opposed with said variable.
    If the table looks like:
        A      B      p
    0  False False  0.015
    1  False True   0.485
    2  True  False  0.495
    3  True  True   0.005
    Then we sum out lines 0 and 2, and lines 1 and 3 to get
        B      p
    0  False  0.51
    1  True   0.49
    We summed every line where B was False, 0.015 + 0.495 = 0.51, and did similarly for B is True, disregarding
    the value of variable A.

    :param cpt: the conditional probability table that we want to simplify through a summing-out
    :param variable: The variable that will be removed through summing out
    :return: the new conditional probability table
    """
    def find_complementary_row(cpt: pd.DataFrame, entry_row: pd.Series, index_to_switch: int) -> Tuple[pd.Series, int]:
        """
        Featuring the most complex list comprehension I have ever written, without a doubt.

        This function creates an array out of the values of the current row, excluding the probability row;
        it then switches the value of the variable to remove, before individually checking every row of the
        given CPT for the row that matches every value (there can only be one).
        It returns the appropriate row, as well as the index of that row, to easily drop it in the main function.

        :param cpt: the CPT through which we iterate to find the complementary value
        :param entry_row: the row of which we want to find the complementary value
        :param index_to_switch: the column index of the row we want to switch. In example above, this would be 0,
            since we want to add up probabilities for switched values of A
        :return: the complementary row and its index, or None if there were no complementaries (so the variable to sum out was given
            as evidence)
        """
        complementary_values = list(entry_row)[:-1]
        complementary_values[index_to_switch] = not complementary_values[index_to_switch]
        row_matches_conditions = [row.all() # every condition needs to evaluate to true
            for row in np.array( # We use numpy so can transpose
                [cpt[cpt.columns[i]] == complementary_values[i] # Compare cpt[SOME_COLUMN] with complementary_value[SOME_COLUMN]
                for i in range(len(complementary_values))] # for every column that we want to match
                ).T]
        try:
            index = row_matches_conditions.index(True)
            complementary_row = cpt.iloc[index]
            return complementary_row, index
        except:
            return None, None

    # print(f"Summing out variable {variable} from \n{cpt}")
    res = cpt.copy()
    cols = list(res.columns)
    var_to_remove = cols.index(variable)
    indices_to_drop = []
    for ii, row in res.iterrows():
        if row[variable] == True:
            opposite_row, index = find_complementary_row(cpt, row, var_to_remove)
            if opposite_row is not None:
                res.loc[ii, "p"] += opposite_row["p"]
                indices_to_drop.append(index)
    res = res.drop(indices_to_drop)
    res = res.reset_index(drop=True)
    res = res.drop(columns=[variable])
    # print(f"End result is: \n{res}\n====================================")
    return res

if __name__ == "__main__":
    bifxml_path = os.getcwd() + "/testing/lecture_example.BIFXML"
    bnr = BNReasoner(bifxml_path)
    query = ['Sprinkler?', 'Rain?']
    evidence = {"Winter?": True}#, "Slippery Road?": False}
    res = (bnr.marginal_distributions(query, evidence, bnr.min_degree()))
    print(res)
