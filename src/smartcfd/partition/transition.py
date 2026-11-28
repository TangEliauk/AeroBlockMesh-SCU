"""Transactional edits; failures leave the original graph untouched."""
from smartcfd.core.types import BlockGraph,FaceBlock,Failure
from smartcfd.partition.actions import describe_action
from smartcfd.partition.constraints import validate_graph,valid_chord
from smartcfd.partition.topology import edge_incidence
def apply_action(graph:BlockGraph,action:int,max_blocks=32,max_half_edges=256):
    try:
        desc=describe_action(graph,action)
    except ValueError as error:
        return Failure("action_out_of_range",str(error))
    if desc["kind"]=="padding":
        return Failure("padding_action")
    if desc["protected_face"]:
        return Failure("protected_band_action")
    candidate=graph.clone();face=candidate.faces[desc["face_id"]];cycle=list(face.vertices);i=desc["edge_index"]
    if desc["kind"]=="insert_chord":
        j=(i+desc["forward_steps"])%len(cycle)
        if not valid_chord([candidate.vertices[v] for v in cycle],i,j):
            return Failure("invalid_chord")
        rotated=cycle[i:]+cycle[:i];k=desc["forward_steps"]
        candidate.faces[face.id]=FaceBlock(face.id,tuple(rotated[:k+1]),False)
        new_id=candidate.next_face_id;candidate.next_face_id+=1
        candidate.faces[new_id]=FaceBlock(new_id,tuple(rotated[k:]+rotated[:1]),False)
    else:
        if desc["protected_edge"]:
            return Failure("protected_interface_action")
        a,b=desc["source"],desc["target"];owners=edge_incidence(candidate)[tuple(sorted((a,b)))]
        if any(candidate.faces[fid].protected for fid,_ in owners):
            return Failure("protected_interface_action")
        new_id=candidate.next_vertex_id;candidate.next_vertex_id+=1
        candidate.vertices[new_id]=tuple((x+y)/2 for x,y in zip(candidate.vertices[a],candidate.vertices[b]))
        for fid,index in owners:
            old=candidate.faces[fid];nodes=list(old.vertices);nodes.insert(index+1,new_id)
            candidate.faces[fid]=FaceBlock(fid,tuple(nodes),old.protected)
    invalid=validate_graph(candidate,graph,max_blocks,max_half_edges)
    return invalid if invalid else candidate
