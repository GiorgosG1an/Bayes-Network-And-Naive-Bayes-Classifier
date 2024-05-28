from typing import List, Optional, Union
from bayes_networks.bayes_node import BayesNode

class BayesNet:
    """Bayesian network containing only boolean-variable nodes."""

    def __init__(self, node_specs: Optional[List[Union[str, List[str], float, dict]]] = None):
        """Nodes must be ordered with parents before children."""
        self.nodes: List[BayesNode] = []
        self.variables: List[str] = []
        node_specs = node_specs or []
        for node_spec in node_specs:
            self.add(node_spec)

    def add(self, node_spec: Union[str, List[str], float, dict]):
        """Add a node to the net. Its parents must already be in the
        net, and its variable must not."""
        node = BayesNode(*node_spec)
        assert node.variable not in self.variables
        assert all((parent in self.variables) for parent in node.parents)
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_node(self, var: str) -> BayesNode:
        """Return the node for the variable named var.
        >>> burglary.variable_node('Burglary').variable
        'Burglary'"""
        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: {}".format(var))

    def variable_values(self, var: str) -> List[bool]:
        """Return the domain of var."""
        return [True, False]

    def __repr__(self) -> str:
        return 'BayesNet({0!r})'.format(self.nodes)