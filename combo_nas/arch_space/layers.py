# -*- coding: utf-8 -*-
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import param_count
from .constructor import Slot

class DAGLayer(nn.Module):
    def __init__(self, n_nodes, chn_in, stride, 
                    allocator, merger_state, merger_out, enumerator, preproc,
                    edge_cls=Slot, edge_kwargs={}):
        super().__init__()
        self.n_nodes = n_nodes
        self.stride = stride
        self.chn_in = chn_in
        self.n_input = len(chn_in)
        self.n_states = self.n_input + self.n_nodes
        self.n_input_e = 1 if isinstance(edge_kwargs['chn_in'], int) else len(edge_kwargs['chn_in'])
        self.allocator = allocator(self.n_input, self.n_nodes)
        self.merger_state = merger_state()
        self.merger_out = merger_out(start=self.n_input)
        self.merge_out_range = self.merger_out.merge_range(self.n_states)
        self.enumerator = enumerator()

        chn_states = []
        if not preproc:
            self.preprocs = None
            chn_states.extend(chn_in)
        else:
            chn_cur = edge_kwargs['chn_in']
            chn_cur = chn_cur if self.n_input == 1 else chn_cur[0]
            self.preprocs = nn.ModuleList()
            for i in range(self.n_input):
                self.preprocs.append(preproc[i](chn_in[i], chn_cur))
                chn_states.append(chn_cur)

        self.fixed = False
        self.dag = nn.ModuleList()
        self.edges = []
        self.num_edges = 0
        for i in range(n_nodes):
            cur_state = self.n_input+i
            self.dag.append(nn.ModuleList())
            num_edges = self.enumerator.len_enum(cur_state, self.n_input_e)
            for sidx in self.enumerator.enum(cur_state, self.n_input_e):
                e_chn_in = self.allocator.chn_in([chn_states[s] for s in sidx], sidx, cur_state)
                edge_kwargs['chn_in'] = e_chn_in
                edge_kwargs['stride'] = stride if all(s < self.n_input for s in sidx) else 1
                e = edge_cls(**edge_kwargs)
                self.dag[i].append(e)
                self.edges.append(e)
            self.num_edges += num_edges
            chn_states.append(self.merger_state.chn_out([ei.chn_out for ei in self.dag[i]]))
            self.chn_out = self.merger_out.chn_out(chn_states)
        # logging.debug('DAGLayer: etype:{} chn_in:{} chn:{} #n:{} #e:{}'.format(str(edge_cls), self.chn_in, edge_kwargs['chn_in'][0],self.n_nodes, self.num_edges))
        # logging.debug('DAGLayer param count: {:.6f}'.format(param_count(self)))
        self.chn_out = self.merger_out.chn_out(chn_states)
        self.chn_states = chn_states

    def forward(self, x):
        if self.preprocs is None:
            states = [st for st in x]
        else:
            states = [self.preprocs[i](x[i]) for i in range(self.n_input)]

        for nidx, edges in enumerate(self.dag):
            res = []
            n_states = self.n_input + nidx
            topo = self.topology[nidx] if self.fixed else None
            for eidx, sidx in enumerate(self.enumerator.enum(n_states, self.n_input_e)):
                if not topo is None and not eidx in topo: continue
                e_in = self.allocator.alloc([states[i] for i in sidx], sidx, n_states)
                res.append(edges[eidx](e_in))
            s_cur = self.merger_state.merge(res)
            states.append(s_cur)
        
        out = self.merger_out.merge(states)
        return out
    
    def apply_edge(self, func, kwargs):
        return [func(**kwargs) for e in self.edges]
    
    def to_genotype(self, k):
        gene = []
        for nidx, edges in enumerate(self.dag):
            topk_genes = []
            n_states = self.n_input + nidx
            topo = self.topology[nidx] if self.fixed else None
            for eidx, sidx in enumerate(self.enumerator.enum(n_states, self.n_input_e)):
                if not topo is None and not eidx in topo: continue
                w_edge, g_edge_child = edges[eidx].to_genotype(k)
                if w_edge < 0: continue
                g_edge = (g_edge_child, sidx, n_states)
                if len(topk_genes) < k:
                    topk_genes.append((w_edge, g_edge))
                    continue
                for i in range(len(topk_genes)):
                    w, g = topk_genes[i]
                    if w_edge > w:
                        topk_genes[i] = (w_edge, g_edge)
                        break
            gene.append([g for w, g in topk_genes])
        return 0, gene
    
    def build_from_genotype(self, gene, *args, **kwargs):
        """ generate discrete ops from gene """
        chn_states = self.chn_states[:self.n_input]
        num_edges = 0
        self.topology = []
        for nidx, (edges, dag_rows) in enumerate(zip(gene, self.dag)):
            cur_state = self.n_input+nidx
            e_chn_out = []
            topo = []
            dag_topology = list(self.enumerator.enum(cur_state, self.n_input_e))
            for g_child, sidx, n_states in edges:
                eidx = dag_topology.index(sidx)
                topo.append(eidx)
                e = dag_rows[eidx]
                e.build_from_genotype(g_child, *args, **kwargs)
                num_edges += 1
                e_chn_out.append(e.chn_out)
            self.topology.append(topo)
            chn_states.append(self.merger_state.chn_out(e_chn_out))
        self.num_edges = num_edges
        self.chn_states = chn_states
        self.chn_out = self.merger_out.chn_out(chn_states)
        self.fixed = True
        # logging.debug('DAGLayer: etype:{} chn_in:{} #n:{} #e:{}'.format(str(edge_cls), self.chn_in, self.n_nodes, self.num_edges))
        # logging.debug('DAGLayer param count: {:.6f}'.format(param_count(self)))


class TreeLayer(nn.Module):
    def __init__(self, n_nodes, chn_in, stride, shared_a,
                    allocator, merger_out, preproc, 
                    child_cls, child_kwargs, edge_cls, edge_kwargs,
                    children=None, edges=None):
        super().__init__()
        self.edges = nn.ModuleList()
        self.subnets = nn.ModuleList()
        chn_in = (chn_in, ) if isinstance(chn_in, int) else chn_in
        self.n_input = len(chn_in)
        self.n_nodes = n_nodes
        self.n_states = self.n_input + self.n_nodes
        self.allocator = allocator(self.n_input, self.n_nodes)
        self.merger_out = merger_out(start=self.n_input)
        self.merge_out_range = self.merger_out.merge_range(self.n_states)

        chn_states = []
        if not preproc:
            self.preprocs = None
            chn_states.extend(chn_in)
        else:
            chn_cur = edge_kwargs['chn_in'][0]
            self.preprocs = nn.ModuleList()
            for i in range(self.n_input):
                self.preprocs.append(preproc(chn_in[i], chn_cur, 1, 1, 0, False))
                chn_states.append(chn_cur)
        
        sidx = range(self.n_input)
        for i in range(self.n_nodes):
            e_chn_in = self.allocator.chn_in([chn_states[s] for s in sidx], sidx, i)
            if not edges is None:
                self.edges.append(edges[i])
                c_chn_in = edges[i].chn_out
            elif not edge_cls is None:
                edge_kwargs['chn_in'] = e_chn_in
                edge_kwargs['stride'] = stride
                if 'shared_a' in edge_kwargs: edge_kwargs['shared_a'] = shared_a
                e = edge_cls(**edge_kwargs)
                self.edges.append(e)
                c_chn_in = e.chn_out
            else:
                self.edges.append(None)
                c_chn_in = e_chn_in
            if not children is None:
                self.subnets.append(children[i])
            elif not child_cls is None:
                child_kwargs['chn_in'] = c_chn_in
                self.subnets.append(child_cls(**child_kwargs))
            else:
                self.subnets.append(None)
        
        # logging.debug('TreeLayer: etype:{} ctype:{} chn_in:{} #node:{} #p:{:.6f}'.format(str(edge_cls), str(child_cls), chn_in, self.n_nodes, param_count(self)))
    
    def forward(self, x):
        x = [x] if not isinstance(x, list) else x
        if self.preprocs is None:
            states = [st for st in x]
        else:
            states = [self.preprocs[i](x[i]) for i in range(self.n_input)]

        n_states = self.n_input
        sidx = range(self.n_input)
        for edge, child in zip(self.edges, self.subnets):
            out = self.allocator.alloc([states[i] for i in sidx], sidx, n_states)
            if not edge is None:
                out = edge(out)
            if not child is None:
                out = child([out])
            states.append(out)
        
        out = self.merger_out.merge(states)
        return out
    
    def build_from_genotype(self, gene):
        pass