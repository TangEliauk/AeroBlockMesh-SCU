"""Reproducible teacher in the same action space, with two real coarse topologies.

The scope is deliberately limited: four outer strips, or four outer strips
subdivided by one additional radial layer. The immutable band is identical.
The teacher does not inject completed graphs or mesh coordinates into the env.
"""
from __future__ import annotations
import numpy as np
from smartcfd.core.types import Failure
from smartcfd.partition.topology import initial_graph,half_edges,all_quadrilateral
from smartcfd.partition.transition import apply_action

def _equal(a,b):
    return bool(np.allclose(a,b,rtol=0.,atol=1e-12))

def _matching_action(env,kind,a,b):
    for edge in env.registry:
        if env.graph.faces[edge.face_id].protected:
            continue
        for local in range(4):
            action=edge.slot*4+local
            desc=env.describe_action(action)
            if desc["kind"]!=kind:
                continue
            if ((_equal(desc["source_uv"],a) and _equal(desc["target_uv"],b)) or
                    (_equal(desc["source_uv"],b) and _equal(desc["target_uv"],a))):
                if env.action_mask[action]:
                    return action
    return None

def _program(variant):
    chord=lambda a,b:("insert_chord",a,b)
    midpoint=lambda a,b:("insert_edge_midpoint",a,b)
    if variant==0:
        return [chord((u,.5),(u,1.)) for u in (.25,.5,.75)]
    result=[chord((.25,.5),(.25,1.)),
            midpoint((.25,.5),(.25,1.)),
            midpoint((0.,.5),(0.,1.)),
            chord((0.,.75),(.25,.75))]
    for left,right in ((.25,.5),(.5,.75)):
        result.extend([chord((right,.5),(right,1.)),
                       midpoint((right,.5),(right,1.)),
                       chord((left,.75),(right,.75))])
    result.extend([midpoint((1.,.5),(1.,1.)),
                   chord((.75,.75),(1.,.75))])
    return result

def solve(env,seed=42,max_nodes=2000):
    """Return actions replayable from env's current reset state, or a Failure.

    max_nodes is an explicit bound on rule attempts, not an unreported beam
    search. The variant is selected by seed parity. A failed requested variant
    is reported, never silently replaced by an easier topology or geometry.
    """
    if env.graph is None:
        return Failure("teacher_requires_reset")
    if env.terminated:
        return [] if env.success else Failure("teacher_received_failed_terminal")
    if max_nodes<1:
        return Failure("teacher_search_budget")
    variants=(int(seed)%2,)
    diagnostics=[]
    attempts=0
    for variant in variants:
        trial=env.clone()
        actions=[]
        failure=None
        for kind,a,b in _program(variant):
            attempts+=1
            if attempts>max_nodes:
                return Failure("teacher_search_budget",diagnostics={"attempts":attempts-1,"failures":diagnostics})
            if trial.terminated:
                failure=Failure("teacher_premature_terminal")
                break
            action=_matching_action(trial,kind,a,b)
            if action is None:
                failure=Failure("teacher_rule_not_legal",diagnostics={"kind":kind,"endpoints":[a,b]})
                break
            try:
                _,_,done,_,info=trial.step(action)
            except (ValueError,IndexError,KeyError,RuntimeError) as error:
                failure=Failure("teacher_replay_error",str(error))
                break
            actions.append(int(action))
            if done:
                if info["success"]:
                    return actions
                failure=Failure("teacher_environment_failure",str(info.get("reason")),
                                diagnostics=info.get("failure",{}))
                break
        if failure is None and trial.success:
            return actions
        diagnostics.append({"variant":variant,"steps":len(actions),
                            "reason":failure.reason if failure else "teacher_did_not_finish",
                            "detail":failure.detail if failure else "",
                            "diagnostics":failure.diagnostics if failure else {}})
    return Failure("teacher_no_valid_rule_partition",diagnostics={"attempts":attempts,"failures":diagnostics})

def rule_graph(domain=None,variant=0):
    """Geometry-free graph fixture built by the very same transactional edits."""
    graph=initial_graph(domain)
    for kind,a,b in _program(variant):
        chosen=None
        for edge in half_edges(graph):
            for local in range(4):
                from smartcfd.partition.actions import describe_action
                action=edge.slot*4+local
                desc=describe_action(graph,action)
                if desc["kind"]==kind and (
                    (_equal(desc["source_uv"],a) and _equal(desc["target_uv"],b)) or
                    (_equal(desc["source_uv"],b) and _equal(desc["target_uv"],a))):
                    candidate=apply_action(graph,action)
                    if not isinstance(candidate,Failure):
                        chosen=candidate
                        break
            if chosen is not None:
                break
        if chosen is None:
            return Failure("rule_graph_construction_failed",diagnostics={"kind":kind,"endpoints":[a,b]})
        graph=chosen
    return graph if all_quadrilateral(graph) else Failure("rule_graph_unresolved")
