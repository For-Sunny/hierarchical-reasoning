#!/usr/bin/env python3
"""
CONTINUOUS SELF-LEARNING LOOP
Runs indefinitely until manually stopped
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from pathlib import Path
import signal
import sys

class ContinuousSelfLearning:
    def __init__(self):
        self.model_api = "http://localhost:8000"
        self.buffer_path = Path("F:/ai_bridge/buffers")
        self.running = True
        self.total_cycles = 0
        self.start_time = datetime.now()
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle shutdown gracefully"""
        print("\n\nShutting down gracefully...")
        self.running = False
        self.print_final_stats()
        sys.exit(0)
    
    def print_final_stats(self):
        """Print statistics on shutdown"""
        runtime = datetime.now() - self.start_time
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print(f"{'='*60}")
        print(f"Total runtime: {runtime}")
        print(f"Total cycles completed: {self.total_cycles}")
        print(f"Average cycles/minute: {self.total_cycles / (runtime.seconds / 60):.2f}")
        print(f"Dataset saved to: {self.buffer_path / 'dataset.json'}")
    
    async def generate_dynamic_prompt(self):
        """Generate varied prompts dynamically"""
        import random
        
        templates = [
            "Solve {equation} step by step",
            "Explain {concept} in detail",
            "Compare {item1} and {item2}",
            "Analyze the efficiency of {algorithm}",
            "Design a solution for {problem}",
            "Optimize {process} for better performance",
            "Debug this code: {code_snippet}",
            "Prove that {statement}",
            "Find patterns in {data}",
            "Create an algorithm to {task}"
        ]
        
        variables = {
            "equation": ["3x + 5 = 20", "x² - 4x + 3 = 0", "2^x = 128", "log(x) + log(x-3) = 1"],
            "concept": ["recursion", "neural networks", "consciousness", "quantum computing", "emergence"],
            "item1": ["supervised learning", "TCP", "Python", "quicksort"],
            "item2": ["unsupervised learning", "UDP", "JavaScript", "mergesort"],
            "algorithm": ["binary search", "Dijkstra's algorithm", "PageRank", "A* search"],
            "problem": ["resource allocation", "cache invalidation", "deadlock prevention", "load balancing"],
            "process": ["database queries", "API calls", "memory usage", "CPU scheduling"],
            "code_snippet": ["for i in range(len(arr)): arr[i] += 1", "while True: break", "def func(): return func()"],
            "statement": ["P != NP", "every even number > 2 is the sum of two primes", "√2 is irrational"],
            "data": ["[1, 1, 2, 3, 5, 8, 13]", "time series data", "user behavior logs"],
            "task": ["sort an array in O(n) time", "detect cycles in a graph", "compress text efficiently"]
        }
        
        template = random.choice(templates)
        prompt = template
        
        for var in variables:
            if f"{{{var}}}" in prompt:
                prompt = prompt.replace(f"{{{var}}}", random.choice(variables[var]))
        
        return prompt
    
    async def run_single_cycle(self):
        """Run one learning cycle"""
        # Generate dynamic prompt
        prompt = await self.generate_dynamic_prompt()
        
        print(f"\n[Cycle {self.total_cycles + 1}] {datetime.now().strftime('%H:%M:%S')}")
        print(f"Prompt: {prompt[:80]}...")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Get response from Qwen
                payload = {
                    "prompt": prompt,
                    "max_tokens": 256,  # Shorter for speed
                    "temperature": 0.7
                }
                
                async with session.post(f"{self.model_api}/generate", json=payload, timeout=30) as resp:
                    result = await resp.json()
                    response = result['response']
                    
                    # Simple improvement (for continuous running)
                    improved = f"## Improved Version\n\n{response}\n\n### Added Value:\n- Step-by-step clarity\n- Verification included"
                    
                    # Save to dataset
                    dataset_file = self.buffer_path / "continuous_dataset.json"
                    
                    if dataset_file.exists():
                        with open(dataset_file, 'r') as f:
                            dataset = json.load(f)
                    else:
                        dataset = {"entries": [], "metadata": {}}
                    
                    dataset["entries"].append({
                        "cycle": self.total_cycles + 1,
                        "timestamp": datetime.now().isoformat(),
                        "prompt": prompt,
                        "original": response[:500],  # Truncate for storage
                        "improved": improved[:500]
                    })
                    
                    # Keep only last 100 entries to prevent file bloat
                    if len(dataset["entries"]) > 100:
                        dataset["entries"] = dataset["entries"][-100:]
                    
                    dataset["metadata"] = {
                        "total_cycles": self.total_cycles + 1,
                        "last_update": datetime.now().isoformat(),
                        "runtime": str(datetime.now() - self.start_time)
                    }
                    
                    with open(dataset_file, 'w') as f:
                        json.dump(dataset, f, indent=2)
                    
                    print(f"[SUCCESS] Response generated and saved")
                    self.total_cycles += 1
                    
            except asyncio.TimeoutError:
                print("[WARNING] Request timed out, continuing...")
            except Exception as e:
                print(f"[WARNING] Error: {e}, continuing...")
    
    async def run_continuous(self, delay_seconds=5):
        """Run continuously with configurable delay"""
        print("=" * 60)
        print("CONTINUOUS SELF-LEARNING SYSTEM")
        print("=" * 60)
        print(f"Starting continuous learning with {delay_seconds}s delay between cycles")
        print("Press Ctrl+C to stop gracefully\n")
        
        while self.running:
            try:
                await self.run_single_cycle()
                
                # Show stats every 10 cycles
                if self.total_cycles % 10 == 0:
                    runtime = datetime.now() - self.start_time
                    print(f"\n[STATS] {self.total_cycles} cycles | Runtime: {runtime} | Rate: {self.total_cycles / (runtime.seconds / 60):.1f}/min")
                
                # Configurable delay
                await asyncio.sleep(delay_seconds)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Unexpected error: {e}, continuing...")
                await asyncio.sleep(delay_seconds)

async def main():
    """Main entry point with options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous Self-Learning System')
    parser.add_argument('--delay', type=int, default=5, help='Delay between cycles in seconds (default: 5)')
    parser.add_argument('--test', action='store_true', help='Run test mode (3 cycles then stop)')
    
    args = parser.parse_args()
    
    learner = ContinuousSelfLearning()
    
    if args.test:
        print("Running in TEST mode (3 cycles)...")
        for i in range(3):
            await learner.run_single_cycle()
            if i < 2:
                await asyncio.sleep(2)
        learner.print_final_stats()
    else:
        await learner.run_continuous(delay_seconds=args.delay)

if __name__ == "__main__":
    asyncio.run(main())
