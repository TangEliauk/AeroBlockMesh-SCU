"""Finite-budget coarse-block MDP on a deliberately restricted C parameter domain."""
from __future__ import annotations
import copy
from dataclasses import asdict
from numbers import Integral
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from smartcfd.core.types import Domain,EnvConfig,Failure,MeshSpec,ENVIRONMENT_VERSION
from smartcfd.partition.actions import ACTION_COUNT,ACTION_SCHEMA,MAX_HALF_EDGES,describe_action
from smartcfd.partition.constraints import validate_graph
from smartcfd.partition.topology import initial_graph,half_edges,all_quadrilateral
from smartcfd.partition.transition import apply_action
from smartcfd.rl.state import observation,OBSERVATION_SCHEMA
from smartcfd.rl.reward import transition_reward

class BlockEnv(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self,domains:list[Domain],config:EnvConfig|None=None):
        if not domains:
            raise ValueError("Pass an explicit nonempty domain list, respecting train/validation/test separation")
        self.domains=list(domains);self.config=config or EnvConfig()
        if self.config.environment_version != ENVIRONMENT_VERSION:
            raise ValueError('BlockEnv only executes the current final-mesh environment contract')
        if not (1<=self.config.max_steps<=128 and 5<=self.config.max_blocks<=32 and 26<=self.config.max_half_edges<=256 and self.config.local_actions==4):
            raise ValueError("Prototype limits: 128 steps, 32 blocks, 256 halfedges and four local actions")
        self.action_space=spaces.Discrete(ACTION_COUNT)
        self.observation_space=spaces.Dict({
            "features":spaces.Box(-np.inf,np.inf,(256,24),dtype=np.float32),
            "next":spaces.Box(-1,255,(256,),dtype=np.int64),
            "previous":spaces.Box(-1,255,(256,),dtype=np.int64),
            "twin":spaces.Box(-1,255,(256,),dtype=np.int64),
            "mask":spaces.Box(0,1,(1024,),dtype=np.bool_),
            "global":spaces.Box(-np.inf,np.inf,(8,),dtype=np.float32)})
        self.graph=None;self.domain=None;self.steps=0;self.terminated=False;self.success=False
        self.mesh=None;self.last_failure=None;self._cache={};self._mask=np.zeros(ACTION_COUNT,dtype=bool)
    @property
    def sample_id(self):
        return self.domain.sample_id if self.domain is not None else None
    @property
    def last_mesh(self):
        return self.mesh
    @property
    def mesh_spec(self):
        return self.config.mesh_spec
    @property
    def environment_context(self):
        settings = {(float(d.farfield), float(d.stretch)) for d in self.domains}
        if len(settings) != 1:
            raise ValueError('A checkpoint context requires consistent domain parameters')
        farfield, stretch = next(iter(settings))
        return {'environment_version': ENVIRONMENT_VERSION, 'env_config': asdict(self.config),
                'mesh_spec': asdict(self.mesh_spec), 'domain_config': {'farfield': farfield, 'stretch': stretch}}
    @property
    def registry(self):
        return half_edges(self.graph)
    @property
    def num_steps(self):
        return self.steps
    @property
    def action_mask(self):
        return self._mask.copy()
    def clone(self):
        other=copy.copy(self)
        other.graph=self.graph.clone() if self.graph is not None else None
        other._cache=dict(self._cache);other._mask=self._mask.copy()
        other._np_random=copy.deepcopy(self.np_random)
        return other
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        options=options or {}
        if "domain_index" in options:
            index=int(options["domain_index"])
        elif "sample_id" in options:
            found=[i for i,d in enumerate(self.domains) if d.sample_id==options["sample_id"]]
            if len(found)!=1:
                raise ValueError("sample_id must identify exactly one supplied domain")
            index=found[0]
        else:
            index=int(self.np_random.integers(len(self.domains)))
        if not 0<=index<len(self.domains):
            raise ValueError("domain_index out of range")
        self.domain_index=index;self.domain=self.domains[index];self.graph=initial_graph(self.domain)
        invalid=validate_graph(self.graph,max_blocks=self.config.max_blocks,max_half_edges=self.config.max_half_edges)
        if invalid:
            raise ValueError(invalid.reason)
        self.steps=0;self.terminated=False;self.success=False;self.mesh=None;self.last_failure=None;self._cache={}
        self._refresh_mask()
        if not self._mask.any():
            self.terminated=True
        return self._observation(),self._info(reason="reset" if not self.terminated else "no_legal_actions")
    def _mesh_for(self,graph):
        from smartcfd.meshing.generator import generate_mesh
        result=generate_mesh(self.domain,graph,self.mesh_spec)
        if isinstance(result,Failure):
            return result
        if not bool(result.quality.get("valid",False)):
            return Failure("invalid_generated_mesh",diagnostics=dict(result.quality))
        quality=float(result.quality.get("q10_scaled_jacobian",0.))
        if not np.isfinite(quality) or quality<=0:
            return Failure("nonpositive_generated_mesh_quality")
        from smartcfd.cfd.solver import validate_su2
        try:
            validate_su2(result)
        except (ValueError, IndexError, KeyError) as error:
            return Failure('invalid_terminal_boundary', str(error))
        return result
    def _candidate(self,action):
        action=int(action)
        if action in self._cache:
            return self._cache[action]
        result=apply_action(self.graph,action,self.config.max_blocks,self.config.max_half_edges)
        mesh=None
        if not isinstance(result,Failure) and all_quadrilateral(result):
            mesh=self._mesh_for(result)
        pair=(result,mesh);self._cache[action]=pair
        return pair
    def _refresh_mask(self):
        self._mask=np.zeros(ACTION_COUNT,dtype=bool)
        if self.terminated:
            return
        for edge in self.registry:
            if self.graph.faces[edge.face_id].protected:
                continue
            for local in range(4):
                action=edge.slot*4+local
                result,_=self._candidate(action)
                self._mask[action]=not isinstance(result,Failure)
    def _observation(self):
        return observation(self.domain,self.graph,self._mask,self.steps,self.config)
    def _info(self,reason=None,**extra):
        result={"success":bool(self.success),"sample_id":self.domain.sample_id,
                "domain_index":self.domain_index,"steps":self.steps,"blocks":len(self.graph.faces),
                "half_edges":len(self.registry),"observation_schema":OBSERVATION_SCHEMA,
                "action_schema":ACTION_SCHEMA,"reason":reason,
                'environment_version':ENVIRONMENT_VERSION,'terminal_mesh_spec':asdict(self.mesh_spec)}
        if self.mesh is not None:
            result["quality"]=dict(self.mesh.quality)
        if self.last_failure is not None:
            result["failure"]={"reason":self.last_failure.reason,"detail":self.last_failure.detail,
                               "diagnostics":self.last_failure.diagnostics}
        result.update(extra)
        return result
    def valid_actions(self):
        return np.flatnonzero(self._mask).astype(int).tolist()
    def describe_action(self,action):
        return describe_action(self.graph,action)
    def step(self,action):
        if self.graph is None:
            raise RuntimeError("Call reset before step")
        if self.terminated:
            raise RuntimeError("Episode ended; reset before another step")
        if not isinstance(action,Integral) or not 0<=int(action)<ACTION_COUNT:
            raise ValueError("Action must be an integer in [0,1024)")
        action=int(action);before=self.graph
        self.steps+=1
        result,mesh=self._candidate(action)
        invalid=isinstance(result,Failure)
        if invalid:
            reason=result.reason
        else:
            self.graph=result;self._cache={}
            self.last_failure=mesh if isinstance(mesh,Failure) else None
            self.mesh=None if self.last_failure else mesh
            self.success=all_quadrilateral(self.graph) and self.mesh is not None
            reason=("mesh_generation_failed:"+self.last_failure.reason if self.last_failure else
                    ("generated_valid_multiblock_mesh" if self.success else "graph_edit"))
        self.terminated=bool(self.success or self.last_failure is not None or self.steps>=self.config.max_steps)
        if not self.success and self.steps>=self.config.max_steps:
            reason="operation_budget_exhausted"
        self._refresh_mask()
        if not self.terminated and not self._mask.any():
            self.terminated=True;reason="no_legal_actions";self._mask[:]=False
        reward=transition_reward(before,self.graph,self.config,terminal=self.terminated,
                    success=self.success,invalid=invalid,
                    quality=self.mesh.quality["q10_scaled_jacobian"] if self.mesh else 0.)
        return self._observation(),reward,bool(self.terminated),False,self._info(reason,invalid_action=invalid,action=action)
