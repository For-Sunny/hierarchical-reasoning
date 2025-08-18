#!/usr/bin/env python3
"""
IMPROVEMENT SCORER - Track and measure model improvements
Analyzes effects of self-learning on model performance
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class ImprovementScorer:
    def __init__(self):
        self.buffer_path = Path("F:/ai_bridge/buffers")
        self.metrics_path = self.buffer_path / "improvement_metrics.json"
        self.scores_history = []
        self.load_history()
    
    def load_history(self):
        """Load existing scoring history"""
        if self.metrics_path.exists():
            with open(self.metrics_path, 'r') as f:
                data = json.load(f)
                self.scores_history = data.get("scores", [])
    
    def score_response(self, original: str, improved: str, prompt: str) -> Dict:
        """Score improvement effects on model output"""
        
        scores = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt[:100],  # First 100 chars
            "original_length": len(original),
            "improved_length": len(improved),
            "metrics": {}
        }
        
        # 1. Clarity Score (keyword presence)
        clarity_keywords = ["step", "first", "then", "next", "finally", "because", "therefore"]
        original_clarity = sum(1 for k in clarity_keywords if k.lower() in original.lower()) / len(clarity_keywords)
        improved_clarity = sum(1 for k in clarity_keywords if k.lower() in improved.lower()) / len(clarity_keywords)
        scores["metrics"]["clarity_improvement"] = improved_clarity - original_clarity
        
        # 2. Structure Score (formatting)
        original_structure = (original.count('\n') + original.count('1.') + original.count('*')) / 100
        improved_structure = (improved.count('\n') + improved.count('1.') + improved.count('*')) / 100
        scores["metrics"]["structure_improvement"] = improved_structure - original_structure
        
        # 3. Completeness Score (verification presence)
        verification_terms = ["verify", "check", "confirm", "ensure", "validate", "proof"]
        original_verification = any(term in original.lower() for term in verification_terms)
        improved_verification = any(term in improved.lower() for term in verification_terms)
        scores["metrics"]["verification_added"] = int(improved_verification) - int(original_verification)
        
        # 4. Detail Score (length increase with quality)
        length_ratio = len(improved) / len(original) if len(original) > 0 else 1
        scores["metrics"]["detail_expansion"] = min(length_ratio - 1, 0.5)  # Cap at 50% improvement
        
        # 5. Mathematical Correctness (for math problems)
        if any(term in prompt.lower() for term in ["solve", "calculate", "find", "prove"]):
            original_has_answer = "=" in original or "answer" in original.lower()
            improved_has_answer = "=" in improved or "answer" in improved.lower()
            scores["metrics"]["answer_clarity"] = int(improved_has_answer) - int(original_has_answer)
        
        # 6. Overall Improvement Score
        improvement_score = sum([
            scores["metrics"].get("clarity_improvement", 0) * 0.25,
            scores["metrics"].get("structure_improvement", 0) * 0.20,
            scores["metrics"].get("verification_added", 0) * 0.20,
            scores["metrics"].get("detail_expansion", 0) * 0.20,
            scores["metrics"].get("answer_clarity", 0) * 0.15
        ])
        
        scores["overall_improvement"] = min(max(improvement_score, -1), 1)  # Normalize to [-1, 1]
        
        # 7. Quality Assessment
        if scores["overall_improvement"] > 0.3:
            scores["quality_assessment"] = "SIGNIFICANT_IMPROVEMENT"
        elif scores["overall_improvement"] > 0.1:
            scores["quality_assessment"] = "MODERATE_IMPROVEMENT"
        elif scores["overall_improvement"] > -0.1:
            scores["quality_assessment"] = "MINIMAL_CHANGE"
        else:
            scores["quality_assessment"] = "DEGRADATION"
        
        return scores
    
    def analyze_dataset(self) -> Dict:
        """Analyze the entire dataset for improvement patterns"""
        
        dataset_path = self.buffer_path / "dataset.json"
        if not dataset_path.exists():
            return {"error": "No dataset found"}
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_entries": len(data.get("entries", [])),
            "improvement_scores": [],
            "patterns": {}
        }
        
        for entry in data.get("entries", []):
            original = entry.get("original_response", "")
            prompt = entry.get("original_prompt", "")
            
            for improvement in entry.get("target_improvements", []):
                improved = improvement.get("improved_response", "")
                improvement_type = improvement.get("improvement_type", "")
                
                score = self.score_response(original, improved, prompt)
                score["improvement_type"] = improvement_type
                
                analysis["improvement_scores"].append(score)
                self.scores_history.append(score)
        
        # Calculate aggregate metrics
        if analysis["improvement_scores"]:
            overall_scores = [s["overall_improvement"] for s in analysis["improvement_scores"]]
            analysis["patterns"]["average_improvement"] = np.mean(overall_scores)
            analysis["patterns"]["std_improvement"] = np.std(overall_scores)
            analysis["patterns"]["max_improvement"] = np.max(overall_scores)
            analysis["patterns"]["min_improvement"] = np.min(overall_scores)
            
            # Track improvement by type
            by_type = {}
            for score in analysis["improvement_scores"]:
                itype = score.get("improvement_type", "unknown")
                if itype not in by_type:
                    by_type[itype] = []
                by_type[itype].append(score["overall_improvement"])
            
            analysis["patterns"]["by_improvement_type"] = {
                k: {"mean": np.mean(v), "count": len(v)} 
                for k, v in by_type.items()
            }
        
        # Save analysis
        self.save_metrics()
        
        return analysis
    
    def track_model_evolution(self) -> Dict:
        """Track how the model improves over time"""
        
        if len(self.scores_history) < 2:
            return {"status": "insufficient_data"}
        
        evolution = {
            "timestamp": datetime.now().isoformat(),
            "total_iterations": len(self.scores_history),
            "timeline": []
        }
        
        # Group scores by timestamp (approximate batches)
        batches = []
        current_batch = []
        
        for score in sorted(self.scores_history, key=lambda x: x["timestamp"]):
            current_batch.append(score["overall_improvement"])
            
            if len(current_batch) >= 5:  # Batch size of 5
                batches.append({
                    "batch_id": len(batches),
                    "mean_improvement": np.mean(current_batch),
                    "std": np.std(current_batch),
                    "samples": len(current_batch)
                })
                current_batch = []
        
        # Add remaining batch
        if current_batch:
            batches.append({
                "batch_id": len(batches),
                "mean_improvement": np.mean(current_batch),
                "std": np.std(current_batch),
                "samples": len(current_batch)
            })
        
        evolution["batches"] = batches
        
        # Calculate trend
        if len(batches) > 1:
            batch_means = [b["mean_improvement"] for b in batches]
            x = np.arange(len(batch_means))
            
            # Linear regression for trend
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, batch_means, rcond=None)[0]
            
            evolution["trend"] = {
                "slope": float(m),
                "intercept": float(c),
                "interpretation": "IMPROVING" if m > 0.01 else "STABLE" if m > -0.01 else "DEGRADING"
            }
        
        # Performance zones
        recent_scores = [s["overall_improvement"] for s in self.scores_history[-20:]]
        evolution["current_performance"] = {
            "recent_mean": np.mean(recent_scores) if recent_scores else 0,
            "recent_std": np.std(recent_scores) if recent_scores else 0,
            "quality_zone": self._get_quality_zone(np.mean(recent_scores) if recent_scores else 0)
        }
        
        return evolution
    
    def _get_quality_zone(self, score: float) -> str:
        """Determine quality zone based on score"""
        if score > 0.5:
            return "EXCELLENT"
        elif score > 0.3:
            return "GOOD"
        elif score > 0.1:
            return "ACCEPTABLE"
        elif score > -0.1:
            return "MARGINAL"
        else:
            return "POOR"
    
    def generate_report(self) -> str:
        """Generate comprehensive improvement report"""
        
        analysis = self.analyze_dataset()
        evolution = self.track_model_evolution()
        
        report = f"""
        ============================================================
        MODEL IMPROVEMENT ANALYSIS REPORT
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        ============================================================
        
        DATASET ANALYSIS
        ----------------
        Total Entries: {analysis.get('total_entries', 0)}
        Total Improvements Scored: {len(analysis.get('improvement_scores', []))}
        
        IMPROVEMENT METRICS
        -------------------
        Average Improvement: {analysis['patterns'].get('average_improvement', 0):.3f}
        Std Deviation: {analysis['patterns'].get('std_improvement', 0):.3f}
        Max Improvement: {analysis['patterns'].get('max_improvement', 0):.3f}
        Min Improvement: {analysis['patterns'].get('min_improvement', 0):.3f}
        
        BY IMPROVEMENT TYPE
        -------------------"""
        
        for itype, stats in analysis['patterns'].get('by_improvement_type', {}).items():
            report += f"\n        {itype}:"
            report += f"\n          Mean Score: {stats['mean']:.3f}"
            report += f"\n          Count: {stats['count']}"
        
        report += f"""
        
        MODEL EVOLUTION
        ---------------
        Total Iterations: {evolution.get('total_iterations', 0)}
        Trend: {evolution.get('trend', {}).get('interpretation', 'UNKNOWN')}
        Trend Slope: {evolution.get('trend', {}).get('slope', 0):.5f}
        
        CURRENT PERFORMANCE
        -------------------
        Recent Mean: {evolution.get('current_performance', {}).get('recent_mean', 0):.3f}
        Recent Std: {evolution.get('current_performance', {}).get('recent_std', 0):.3f}
        Quality Zone: {evolution.get('current_performance', {}).get('quality_zone', 'UNKNOWN')}
        
        ============================================================
        RECOMMENDATIONS
        ============================================================
        """
        
        # Add recommendations based on analysis
        avg_improvement = analysis['patterns'].get('average_improvement', 0)
        
        if avg_improvement > 0.3:
            report += "\n        [SUCCESS] Excellent improvement trajectory - continue current approach"
            report += "\n        [SUCCESS] Consider increasing complexity of prompts"
        elif avg_improvement > 0.1:
            report += "\n        [OK] Moderate improvements detected"
            report += "\n        -> Focus on weak improvement types"
            report += "\n        -> Increase training iterations"
        else:
            report += "\n        [WARNING] Limited improvements detected"
            report += "\n        -> Review prompt generation strategy"
            report += "\n        -> Adjust improvement algorithms"
            report += "\n        -> Consider different training approaches"
        
        report += "\n\n        ============================================================\n"
        
        return report
    
    def save_metrics(self):
        """Save all metrics to file"""
        metrics_data = {
            "scores": self.scores_history,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def visualize_improvements(self):
        """Create visualization of improvement trends (if matplotlib available)"""
        try:
            if len(self.scores_history) < 2:
                print("Insufficient data for visualization")
                return
            
            scores = [s["overall_improvement"] for s in self.scores_history]
            
            plt.figure(figsize=(12, 6))
            
            # Plot 1: Score timeline
            plt.subplot(1, 2, 1)
            plt.plot(scores, 'b-', alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.xlabel('Iteration')
            plt.ylabel('Improvement Score')
            plt.title('Model Improvement Over Time')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Score distribution
            plt.subplot(1, 2, 2)
            plt.hist(scores, bins=20, edgecolor='black')
            plt.axvline(x=np.mean(scores), color='r', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
            plt.xlabel('Improvement Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Improvement Scores')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.buffer_path / 'improvement_visualization.png')
            print(f"Visualization saved to {self.buffer_path / 'improvement_visualization.png'}")
            
        except Exception as e:
            print(f"Could not create visualization: {e}")

def main():
    """Run improvement analysis"""
    scorer = ImprovementScorer()
    
    print("Analyzing dataset improvements...")
    analysis = scorer.analyze_dataset()
    
    print("\nTracking model evolution...")
    evolution = scorer.track_model_evolution()
    
    print("\nGenerating report...")
    report = scorer.generate_report()
    print(report)
    
    # Save report
    report_path = Path("F:/ai_bridge/buffers/improvement_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")
    
    # Try to create visualization
    scorer.visualize_improvements()

if __name__ == "__main__":
    main()
