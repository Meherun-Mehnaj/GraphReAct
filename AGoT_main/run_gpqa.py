import asyncio
import json
import re
import os
from agot import AGoT, OllamaAdapter, HFAdapter, Agents, Embedder
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio
from bert_score import score as bert_score
from typing import List, Dict, Any

# ---------------------------
# Setup Environment
# ---------------------------
if "GEMINI_API_KEY" not in os.environ and "AGOT_BACKEND" not in os.environ:
    print("⚠️  Warning: GEMINI_API_KEY not set and AGOT_BACKEND not specified.")
    print("   Set AGOT_BACKEND=gemini|openai|mock")
    print("   For Gemini: export GEMINI_API_KEY=your_key")
    print("   For OpenAI: export OPENAI_API_KEY=your_key")

# ---------------------------
# Evaluation Utilities
# ---------------------------
def normalize_answer(ans: str) -> str:
    """Extract only A, B, C, D from answer (case-insensitive)."""
    if not isinstance(ans, str):
        return ""
    text = ans.strip().upper()
    match = re.search(r'\b([ABCD])\b', text)
    return match.group(1) if match else ""

def compute_exact_match(preds: List[str], refs: List[str]) -> float:
    matches = [p == r for p, r in zip(preds, refs) if p and r]
    return (sum(matches) / len(matches) * 100) if matches else 0.0

# ---------------------------
# Main Runner
# ---------------------------
async def run_sample(sample: Dict[str, Any], llm_backend: str, shared_agents: Agents) -> Dict[str, Any]:
    # Use shared agents with simplified parameters for faster processing
    agot = AGoT(shared_agents, lmax=2, dmax=1, nmax=2)  # Simplified for speed

    question = sample["question"]
    ref_answer = sample["answer"]

    try:
        prediction_raw, graph, metrics = await agot.run(question)
    except Exception as e:
        error_msg = str(e) if str(e) else f"Unknown error: {type(e).__name__}"
        print(f"❌ Error on ID {sample.get('id', 'N/A')}: {error_msg}")
        prediction_raw = f"ERROR: {error_msg}"
        graph = None

    pred_norm = normalize_answer(prediction_raw)
    ref_norm = normalize_answer(ref_answer)

    return {
        "id": sample.get("id", "unknown"),
        "question": question,
        "reference_answer": ref_answer,
        "prediction_raw": prediction_raw,
        "prediction_normalized": pred_norm,
        "reference_normalized": ref_norm,
        "correct": pred_norm == ref_norm,
        "graph": graph.to_dict() if graph else {}
    }

# ---------------------------
# Dataset Runner
# ---------------------------
async def run_agot_on_dataset(
    dataset,
    backend: str = "gemini",
    max_samples: int = None,
    save_prefix: str = "agot_gpqa_results"
) -> Dict[str, Any]:

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"🚀 Running AGoT ({backend}) on {len(dataset)} GPQA-Diamond samples...")

    # Create shared LLM and agents (avoid recreating embedder for each sample)
    if backend == "ollama":
        llm = OllamaAdapter(model="t1c/deepseek-math-7b-rl:Q8")
        
        # Health check function for Ollama
        async def ollama_health_check():
            try:
                test_response = await llm.generate("Test", temperature=0.1, max_tokens=10)
                return len(test_response.strip()) > 0
            except:
                return False
        
        # Initial health check
        print("🔍 Performing initial Ollama health check...")
        if not await ollama_health_check():
            print("⚠️  Ollama health check failed. Please ensure Ollama is running and the model is loaded.")
            
    elif backend == "hf":
        llm = HFAdapter("deepseek-ai/deepseek-math-7b-instruct")
    else:
        # Mock backend
        from agot import BaseLLM
        class MockLLM(BaseLLM):
            async def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
                return "Answer: A (mock response)"
        llm = MockLLM()
    
    print("📦 Loading embedder (one-time setup)...")
    try:
        embedder = Embedder()
        print("✅ Embedder loaded successfully")
    except Exception as e:
        print(f"⚠️ Embedder failed to load: {e}")
        print("🔄 Using fallback without sentence-transformers...")
        # Create a simple embedder that doesn't use sentence-transformers
        class SimpleEmbedder:
            def __init__(self):
                self.model = None
            async def embed(self, text: str):
                return None
        embedder = SimpleEmbedder()
    
    agents = Agents(llm, embedder, cache_enabled=True)
    
    # Process samples with periodic health checks for Ollama
    if backend == "ollama":
        results = []
        batch_size = 20  # Process in batches of 20 to allow health checks
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            print(f"📊 Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size} ({len(batch)} samples)")
            
            # Health check every batch
            if i > 0:  # Skip initial check since we already did it
                print("🔍 Performing periodic Ollama health check...")
                if not await ollama_health_check():
                    print("⚠️  Ollama health check failed during processing. Continuing with current batch...")
            
            # Process this batch
            batch_tasks = [run_sample(sample, backend, agents) for sample in batch]
            batch_results = await tqdm_asyncio.gather(*batch_tasks, desc=f"Batch {i//batch_size + 1}", unit="sample")
            results.extend(batch_results)
    else:
        # For non-Ollama backends, process all at once
        tasks = [run_sample(sample, backend, agents) for sample in dataset]
        results = await tqdm_asyncio.gather(*tasks, desc="Processing", unit="sample")

    # ---------------------------
    # Metrics
    # ---------------------------
    preds_norm = [r["prediction_normalized"] for r in results]
    refs_norm = [r["reference_normalized"] for r in results]
    preds_raw = [r["prediction_raw"] for r in results]
    refs_raw = [r["reference_answer"] for r in results]

    exact_match = compute_exact_match(preds_norm, refs_norm)

    try:
        P, R, F1 = bert_score(preds_raw, refs_raw, lang="en", verbose=False)
        bert_f1 = float(F1.mean())
    except Exception as e:
        print(f"⚠️ BERTScore failed: {e}")
        bert_f1 = 0.0

    metrics = {
        "backend": backend,
        "exact_match_percent": round(exact_match, 2),
        "bertscore_f1": round(bert_f1, 6),
        "total_samples": len(dataset),
        "failed_samples": len([r for r in results if "ERROR:" in r["prediction_raw"]]),
        "correct_samples": sum(r["correct"] for r in results)
    }

    # ---------------------------
    # Save Results
    # ---------------------------
    os.makedirs("results", exist_ok=True)

    detail_path = f"results/{save_prefix}_details.json"
    metrics_path = f"results/{save_prefix}_metrics.json"

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print("🧠 AGoT + GPQA-Diamond Results")
    print("="*50)
    print(json.dumps(metrics, indent=2))
    print(f"📄 Details saved to: {detail_path}")
    print(f"📊 Metrics saved to: {metrics_path}")
    print("="*50)

    return metrics

# ---------------------------
# Main Entry
# ---------------------------
if __name__ == "__main__":
    async def main():
        print("📊 Loading GPQA-Diamond (fingertap/GPQA-Diamond)...")
        
        try:
            ds_dict = load_dataset("fingertap/GPQA-Diamond")
            if 'test' not in ds_dict:
                print(f"Available splits: {list(ds_dict.keys())}")
                raise ValueError("No 'test' split found!")
            dataset = ds_dict['test']
            print(f"✅ Loaded {len(dataset)} test samples.")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            print("💡 Run: huggingface-cli login")
            print("    and ensure you have access to fingertap/GPQA-Diamond")
            return

        backend = os.getenv("AGOT_BACKEND", "ollama").lower()
        if backend not in ["ollama", "hf", "mock"]:
            print(f"Invalid backend '{backend}'. Using 'ollama'.")
            backend = "ollama"

        await run_agot_on_dataset(
            dataset=dataset,
            backend=backend,
            max_samples=None,  # Run on full dataset (198 samples)
            save_prefix=f"agot_gpqa_diamond_deepseek_{backend}_full_notimeout"
        )

    asyncio.run(main())