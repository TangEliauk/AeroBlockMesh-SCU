"""Bounded potential shaping with explicit finite-budget terminal outcomes."""
from smartcfd.core.types import EnvConfig
def topology_defect(graph):
    return sum(abs(len(face.vertices)-4) for face in graph.faces.values() if not face.protected)
def potential(graph):
    defect=topology_defect(graph)
    return -defect/(defect+4.0)
def transition_reward(before,after,config:EnvConfig,*,terminal=False,success=False,invalid=False,quality=0.):
    # Terminal potential is zero, including failure; bounded Phi prevents reward hacking.
    value=-config.step_cost+config.gamma*(0. if terminal else potential(after))-potential(before)
    if invalid and not terminal:
        value+=config.failure_reward
    if terminal:
        value+=(config.success_reward+max(0.,min(float(quality),1.))) if success else config.failure_reward
    return float(value)
