"""Transparent fixed C-topology control for the constrained prototype."""
from smartcfd.baselines.rule_partition import rule_graph
def template_partition(domain):
    return rule_graph(domain,variant=0)
