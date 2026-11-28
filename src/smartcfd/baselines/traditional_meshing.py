"""Same mesh generator and resolution for the fixed-topology baseline."""
from smartcfd.core.types import Failure,MeshSpec
from smartcfd.baselines.template_partition import template_partition
def traditional_mesh(domain,spec=None):
    from smartcfd.meshing.generator import generate_mesh
    graph=template_partition(domain)
    if isinstance(graph,Failure):
        return graph
    return generate_mesh(domain,graph,spec or MeshSpec())
