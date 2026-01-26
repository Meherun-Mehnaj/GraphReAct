#!/usr/bin/env python3
"""
AGoT research single-file implementation (real LLMs only)

Features:
 - T∅, T₀, Tₑ, C (numeric), Eval (with confidence), Φ (weighted)
 - adaptive edge selection via embeddings (sentence-transformers if available)
 - node pruning Ψ (top-k + duplicate removal)
 - caching & run metrics
 - supports HF local models (transformers) and Ollama/vLLM HTTP
 - exports JSON results into ./results/

Usage examples:
  # Ollama HTTP (recommended if you run a local server)
  python agot.py --backend ollama --model qwen2.5:7b --query "Explain quantum teleportation simply."

  # Hugging Face local model (may need GPU / lots of RAM)
  python agot.py --backend hf --model Qwen/Qwen2-1.8B-Instruct --query "..."

Notes:
 - If you want high-quality embeddings for similarity, `pip install sentence-transformers`
 - If you use HF backend, install: transformers, accelerate, torch
 - For Ollama, run ollama/vllm server locally and pass its base URL with --ollama_url
"""

import asyncio
import aiohttp
import argparse
import json
import math
import os
import time
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any

# -----------------------
# Utilities & helpers
# -----------------------
def now_ts():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def split_semicolon_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    s = s.strip()
    s = re.sub(r'(?i)^\s*(thoughts|subthoughts|follow-up|followups|follow-up:)\s*[:\-]?\s*', '', s)
    if ";" in s:
        parts = [p.strip() for p in s.split(";") if p.strip()]
        if parts:
            return parts
    lines = [re.sub(r'^[\-\•\d\.\)\s]+', '', l).strip() for l in s.splitlines() if l.strip()]
    if len(lines) > 1:
        return lines
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts or [s]

def simple_text_vector(text: str) -> Dict[str,int]:
    toks = re.findall(r"\w+", text.lower())
    return Counter(toks)

def cosine_similarity_counts(a: Dict[str,int], b: Dict[str,int]) -> float:
    common = set(a.keys()) & set(b.keys())
    num = sum(a[k]*b[k] for k in common)
    norma = math.sqrt(sum(v*v for v in a.values())) or 1.0
    normb = math.sqrt(sum(v*v for v in b.values())) or 1.0
    return max(0.0, min(1.0, num / (norma * normb)))

def jaccard_similarity(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

# -----------------------
# Data structures
# -----------------------
@dataclass
class Node:
    id: str
    thought: str
    strategy: str = ""
    answer: Optional[str] = None
    heritage: Tuple[Tuple[int,int], ...] = ()
    complex_score: float = 0.0
    state: str = "new"
    children: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d['heritage'] = [list(h) for h in d['heritage']]
        return d

@dataclass
class Graph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Tuple[str,str]] = field(default_factory=list)
    final_answer: Optional[str] = None

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_edge(self, a: str, b: str):
        self.edges.append((a,b))
        if a in self.nodes:
            self.nodes[a].children.append(b)

    def layer_nodes(self, layer_index: int) -> List[Node]:
        return [n for n in self.nodes.values() if any(h[0] == layer_index for h in n.heritage)]

    def summary(self, n_chars: int = 120) -> str:
        lines = []
        for node in self.nodes.values():
            ans = (node.answer[:80] + "...") if node.answer and len(node.answer) > 80 else (node.answer or "")
            lines.append(f"- {node.id[:8]} {node.heritage} {node.thought[:n_chars]} -> {ans} (s={node.score:.2f}, c={node.complex_score:.2f})")
        return "\n".join(lines)

    def to_dict(self):
        return {"nodes": {nid: n.to_dict() for nid,n in self.nodes.items()}, "edges": self.edges, "final_answer": self.final_answer}

# -----------------------
# LLM Adapters (real LLMs only)
# -----------------------
class BaseLLM:
    async def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        raise NotImplementedError
    async def embed(self, text: str) -> Optional[List[float]]:
        return None

class OllamaAdapter(BaseLLM):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-math-7b-rl", timeout: int = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout  # None means no timeout

    async def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model, 
            "prompt": prompt, 
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        
        # Retry logic with exponential backoff for connection issues
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Create new session for each request to prevent connection pooling issues
                connector = aiohttp.TCPConnector(
                    limit=100,  # Total connection pool size
                    limit_per_host=10,  # Per-host connection limit
                    keepalive_timeout=30,  # Keep connections alive for 30 seconds
                    enable_cleanup_closed=True  # Clean up closed connections
                )
                timeout = aiohttp.ClientTimeout(total=None)  # No timeout
                
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as sess:
                    async with sess.post(f"{self.base_url}/api/generate", json=payload) as resp:
                        if resp.status != 200:
                            raise Exception(f"Ollama API error: HTTP {resp.status}")
                        text = await resp.text()
                        try:
                            data = json.loads(text)
                            if isinstance(data, dict) and "response" in data:
                                return data["response"].strip()
                            else:
                                raise Exception(f"Unexpected Ollama response format: {text[:100]}")
                        except json.JSONDecodeError as e:
                            raise Exception(f"JSON parse error: {e}, raw response: {text[:200]}")
                            
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    if "Ollama" not in str(e):
                        raise Exception(f"Ollama connection error: {e}")
                    raise
                else:
                    # Wait before retry with exponential backoff
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️  Ollama request failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    continue

    async def embed(self, text: str) -> Optional[List[float]]:
        return None

class HFAdapter(BaseLLM):
    def __init__(self, model_name: str, device: Optional[str] = None, tokenizer_fast: bool = True):
        # lazy imports
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=tokenizer_fast)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map="auto" if device is None else {"": device},
            low_cpu_mem_usage=True
        )
        self.model.eval()
        self._device = next(self.model.parameters()).device
        self._torch = torch

    async def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        loop = asyncio.get_running_loop()
        def sync_generate():
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self._device)
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=float(temperature),
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
            out = self.tokenizer.decode(gen[0], skip_special_tokens=True)
            return out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
        return await loop.run_in_executor(None, sync_generate)

    async def embed(self, text: str) -> Optional[List[float]]:
        return None

# -----------------------
# Embedding helper: try sentence-transformers, else fallback to BoW
# -----------------------
class Embedder:
    def __init__(self):
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            # default model - you can pass a different name if desired
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            self.model = None

    async def embed(self, text: str) -> Optional[List[float]]:
        if self.model:
            loop = asyncio.get_running_loop()
            def sync_emb():
                vec = self.model.encode(text, normalize_embeddings=True)
                return vec.tolist() if hasattr(vec, "tolist") else list(map(float, vec))
            return await loop.run_in_executor(None, sync_emb)
        return None

# -----------------------
# Agents (T∅, T₀, Tₑ, C, Eval, Φ)
# -----------------------
class Agents:
    def __init__(self, llm: BaseLLM, embedder: Embedder, cache_enabled: bool = True):
        self.llm = llm
        self.embedder = embedder
        self.cache_enabled = cache_enabled
        self._cache: Dict[str,str] = {}
        self.metrics = {"llm_calls":0, "llm_time_s":0.0, "token_est":0}

    async def _generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        if self.cache_enabled and prompt in self._cache:
            return self._cache[prompt]
        t0 = time.time()
        self.metrics["llm_calls"] += 1
        resp = await self.llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        dt = time.time() - t0
        self.metrics["llm_time_s"] += dt
        self.metrics["token_est"] += len(resp.split())
        if self.cache_enabled:
            self._cache[prompt] = resp
        return resp

    async def T_initial(self, query: str, nmax: int = 3) -> Tuple[List[str], str]:
        prompt = (f"Generate up to {nmax} initial thoughts for the query below. "
                  "Return semicolon-separated short thought titles (no numbering).\n\n"
                  f"Query:\n{query}\n\nInstruction: (generate initial thoughts)")
        resp = await self._generate(prompt)
        parts = split_semicolon_list(resp)
        return parts[:nmax], "initial"

    async def T_nested_initial(self, complex_thought: str, parent_graph: Graph, nmax: int = 3) -> Tuple[List[str], str]:
        prompt = (f"Decompose the complex thought into up to {nmax} smaller focused thoughts for nested reasoning. "
                  "Return semicolon-separated items.\n\nThought:\n" + complex_thought
                  + "\n\nContext summary:\n" + (parent_graph.summary(120) if parent_graph else ""))
        resp = await self._generate(prompt)
        parts = split_semicolon_list(resp)
        return parts[:nmax], "nested"

    async def T_general(self, context: str, graph: Graph, nmax: int = 3) -> Tuple[List[str], str, List[Tuple[str,str]]]:
        prompt = (f"Given the problem statement and current graph summary, propose up to {nmax} follow-up thoughts "
                  "that help reach a solution. Return semicolon-separated thought titles.\n\n"
                  f"Context:\n{context}\n\nGraph summary:\n{graph.summary(120)}")
        resp = await self._generate(prompt)
        parts = split_semicolon_list(resp)
        return parts[:nmax], "general", []

    async def C_complexity_score(self, thought: str, graph: Graph) -> float:
        prompt = ( "Complexity check: Provide a numeric score between 0.0 (simple) and 1.0 (very complex). "
                   "Return only a float between 0 and 1 if possible.\n\n"
                   f"Thought:\n{thought}\n\nContext summary:\n{graph.summary(120)}")
        resp = await self._generate(prompt)
        m = re.search(r"(\d*\.\d+|\d+)", resp)
        if m:
            try:
                val = float(m.group(1))
                if val > 1:
                    if val <= 100:
                        val = min(1.0, val / 100.0)
                    else:
                        val = 1.0
                return max(0.0, min(1.0, val))
            except:
                pass
        heur = 0.0
        heur += min(1.0, len(thought) / 300.0)
        if any(k in thought.lower() for k in ("derive","prove","optimize","nontrivial","complex","intractable","simulate")):
            heur = min(1.0, heur + 0.35)
        return heur

    async def Eval(self, thought: str, graph: Graph) -> Tuple[str, float]:
        prompt = ( "Evaluate the thought and provide a concise, grounded result or sub-answer. "
                   "If numerical, compute or explain steps briefly. At the end, append '|| score:X' where X is a confidence between 0 and 1.\n\n"
                   f"Thought:\n{thought}\n\nContext summary:\n{graph.summary(120)}")
        resp = await self._generate(prompt)
        score = 0.5
        m = re.search(r"\|\|\s*score\s*[:=]\s*(\d*\.\d+|\d+)", resp, flags=re.IGNORECASE)
        if m:
            try:
                score = float(m.group(1))
                score = max(0.0, min(1.0, score))
                resp = re.sub(r"\|\|\s*score\s*[:=]\s*(\d*\.\d+|\d+)", "", resp, flags=re.IGNORECASE).strip()
            except:
                score = 0.5
        else:
            if len(resp.split()) < 10:
                score = min(0.9, score + 0.1)
            if any(w in resp.lower() for w in ("likely","probably","may","might","uncertain")):
                score = min(score, 0.6)
        return resp.strip(), score

    async def Phi(self, graph: Graph) -> str:
        lines = []
        for n in graph.nodes.values():
            lines.append(f"- ID:{n.id[:8]} score:{n.score:.3f} c:{n.complex_score:.3f} thought:{n.thought} answer:{(n.answer or '')}")
        prompt = ( "Synthesize a final concise solution from the following node answers. Weight node answers by their scores. "
                   "If there is conflict, highlight uncertainty and give the most supported answer. Keep output ≤ 400 tokens.\n\n"
                   "Nodes:\n" + "\n".join(lines))
        resp = await self._generate(prompt)
        return resp.strip()

    async def embed(self, text: str) -> Optional[List[float]]:
        emb = await self.embedder.embed(text)
        if emb:
            return emb
        try:
            return await self.llm.embed(text)
        except Exception:
            return None

# -----------------------
# Similarity utility (embeddings preferred)
# -----------------------
async def similarity_score(a_text: str, b_text: str, agents: Agents) -> float:
    vec_a = await agents.embed(a_text)
    vec_b = await agents.embed(b_text)
    if vec_a and vec_b:
        num = sum(x*y for x,y in zip(vec_a, vec_b))
        na = math.sqrt(sum(x*x for x in vec_a)) or 1.0
        nb = math.sqrt(sum(x*x for x in vec_b)) or 1.0
        return max(0.0, min(1.0, num / (na*nb)))
    va = simple_text_vector(a_text)
    vb = simple_text_vector(b_text)
    cos = cosine_similarity_counts(va, vb)
    jac = jaccard_similarity(a_text, b_text)
    return 0.6 * cos + 0.4 * jac

# -----------------------
# Pruning Ψ
# -----------------------
def prune_nodes(graph: Graph, keep_top_k: int = 6, similarity_threshold: float = 0.92) -> None:
    nodes = list(graph.nodes.values())
    nodes.sort(key=lambda n: n.score, reverse=True)
    kept = nodes[:keep_top_k]
    kept_ids = {n.id for n in kept}
    pruned_ids = []
    for n in nodes[keep_top_k:]:
        dup = False
        for k in kept:
            if jaccard_similarity(n.thought, k.thought) >= similarity_threshold:
                dup = True
                break
        if dup:
            pruned_ids.append(n.id)
        else:
            pruned_ids.append(n.id)
    for pid in pruned_ids:
        if pid in graph.nodes:
            del graph.nodes[pid]
    graph.edges = [(a,b) for (a,b) in graph.edges if a in graph.nodes and b in graph.nodes]

# -----------------------
# Core AGoT engine
# -----------------------
class AGoT:
    def __init__(self, agents: Agents, dmax: int = 2, lmax: int = 3, nmax: int = 3, concurrency: int = 6,
                 pruning_k: int = 8, prune_sim_threshold: float = 0.92, complexity_threshold: float = 0.5):
        self.agents = agents
        self.dmax = dmax
        self.lmax = lmax
        self.nmax = nmax
        self.semaphore = asyncio.Semaphore(concurrency)
        self.pruning_k = pruning_k
        self.prune_sim_threshold = prune_sim_threshold
        self.complexity_threshold = complexity_threshold
        self.metrics = {"nodes_created":0, "nested_graphs":0, "node_evals":0, "edges_created":0, "pruned_nodes":0, "start_time":None, "end_time":None}

    async def _evaluate_node(self, node: Node, graph: Graph):
        if node.state != "new":
            return
        node.state = "evaluating"
        node.complex_score = await self.agents.C_complexity_score(node.thought, graph)
        if node.complex_score >= 0.8:
            node.strategy = "refine"
        elif node.complex_score >= 0.4:
            node.strategy = "explore"
        else:
            node.strategy = "verify"

        if node.complex_score >= self.complexity_threshold and len(node.heritage) <= self.dmax:
            self.metrics["nested_graphs"] += 1
            nested_texts, nested_strategy = await self.agents.T_nested_initial(node.thought, graph, nmax=self.nmax)
            nested_graph = Graph()
            for idx, t in enumerate(nested_texts):
                nid = str(uuid.uuid4())
                nh = node.heritage + ((0, idx),)
                nn = Node(id=nid, thought=t, strategy=nested_strategy, heritage=nh)
                nested_graph.add_node(nn)
                self.metrics["nodes_created"] += 1
            await asyncio.gather(*(self._evaluate_node(nn, nested_graph) for nn in list(nested_graph.nodes.values())))
            node.answer = await self.agents.Phi(nested_graph)
            node.score = (sum(n.score for n in nested_graph.nodes.values()) / (len(nested_graph.nodes) or 1))
            node.state = "complex-evaluated"
            for nn in nested_graph.nodes.values():
                graph.add_node(nn)
                graph.add_edge(node.id, nn.id)
                self.metrics["edges_created"] += 1
        else:
            ans, sc = await self.agents.Eval(node.thought, graph)
            node.answer = ans
            node.score = sc
            node.state = "evaluated"
        self.metrics["node_evals"] += 1

    async def _evaluate_nodes_batch(self, nodes: List[Node], graph: Graph):
        async def worker(n):
            async with self.semaphore:
                await self._evaluate_node(n, graph)
        await asyncio.gather(*(worker(n) for n in nodes))

    async def run(self, query: str) -> Tuple[str, Graph, Dict[str,Any]]:
        self.metrics["start_time"] = time.time()
        graph = Graph()
        initial_texts, strat = await self.agents.T_initial(query, nmax=self.nmax)
        for idx, t in enumerate(initial_texts):
            nid = str(uuid.uuid4())
            node = Node(id=nid, thought=t, strategy=strat, heritage=((0, idx),))
            graph.add_node(node)
            self.metrics["nodes_created"] += 1

        for layer in range(self.lmax):
            layer_nodes = [n for n in list(graph.nodes.values()) if any(h[0] == layer for h in n.heritage)]
            if not layer_nodes:
                continue
            await self._evaluate_nodes_batch(layer_nodes, graph)
            prune_nodes(graph, keep_top_k=self.pruning_k, similarity_threshold=self.prune_sim_threshold)
            self.metrics["pruned_nodes"] = max(0, self.metrics["nodes_created"] - len(graph.nodes))
            candidates, estrat, _ = await self.agents.T_general(query, graph, nmax=self.nmax * 2)
            next_layer = layer + 1
            for idx, cand in enumerate(candidates[:self.nmax]):
                nid = str(uuid.uuid4())
                new_node = Node(id=nid, thought=cand, strategy=estrat, heritage=((next_layer, idx),))
                graph.add_node(new_node)
                self.metrics["nodes_created"] += 1
                sims = []
                for parent in layer_nodes:
                    sim = await similarity_score(new_node.thought, parent.thought, self.agents)
                    sims.append((parent.id, sim))
                sims.sort(key=lambda x: x[1], reverse=True)
                attached = False
                for pid, s in sims[:2]:
                    if s >= 0.15:
                        graph.add_edge(pid, new_node.id)
                        self.metrics["edges_created"] += 1
                        attached = True
                if not attached and layer_nodes:
                    pid, s = sims[0]
                    graph.add_edge(pid, new_node.id)
                    self.metrics["edges_created"] += 1

        final = await self.agents.Phi(graph)
        graph.final_answer = final
        self.metrics["end_time"] = time.time()
        all_metrics = {**self.metrics, **self.agents.metrics, "total_time_s": self.metrics["end_time"] - self.metrics["start_time"]}
        return final, graph, all_metrics

# -----------------------
# CLI runner
# -----------------------
async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ollama","hf"], required=True, help="LLM backend: ollama or hf")
    p.add_argument("--model", required=True, help="Model id (ollama: qwen2.5:7b, hf: Qwen/Qwen2-1.8B-Instruct etc.)")
    p.add_argument("--ollama_url", default="http://localhost:11434", help="Ollama base url")
    p.add_argument("--query", default="Explain quantum teleportation in simple steps and estimate feasibility.", help="Problem query")
    p.add_argument("--dmax", type=int, default=2)
    p.add_argument("--lmax", type=int, default=3)
    p.add_argument("--nmax", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--prune_k", type=int, default=8)
    p.add_argument("--prune_sim", type=float, default=0.92)
    p.add_argument("--complex_thresh", type=float, default=0.5)
    p.add_argument("--cache", action="store_true")
    p.add_argument("--export_dir", default="results", help="Directory to save JSON export")
    args = p.parse_args()

    if args.backend == "ollama":
        llm = OllamaAdapter(base_url=args.ollama_url, model=args.model)
    elif args.backend == "hf":
        llm = HFAdapter(args.model)
    else:
        raise ValueError("Unsupported backend")

    embedder = Embedder()
    agents = Agents(llm, embedder, cache_enabled=args.cache)
    agot = AGoT(agents, dmax=args.dmax, lmax=args.lmax, nmax=args.nmax, concurrency=args.concurrency,
                pruning_k=args.prune_k, prune_sim_threshold=args.prune_sim, complexity_threshold=args.complex_thresh)

    print(f"Starting AGoT run: backend={args.backend} model={args.model}")
    final, graph, metrics = await agot.run(args.query)
    print("\n=== FINAL SYNTHESIS ===\n")
    print(final)
    print("\n=== GRAPH SUMMARY ===\n")
    print(graph.summary(300))
    print("\n=== METRICS ===")
    for k,v in metrics.items():
        print(f"{k}: {v}")

    ensure_dir(args.export_dir)
    out_path = os.path.join(args.export_dir, f"agot_{now_ts()}.json")
    out = {"query": args.query, "final": final, "graph": graph.to_dict(), "metrics": metrics}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nExported run data to {out_path}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
