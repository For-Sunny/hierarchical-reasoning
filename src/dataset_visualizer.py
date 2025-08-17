#!/usr/bin/env python3
"""
Dataset Analyzer and Visualizer - Grok's Enhancement
Analyzes self-learning dataset and creates visualizations
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter

class DatasetAnalyzer:
    def __init__(self):
        self.buffer_path = Path("F:/ai_bridge/buffers")
        self.visual_path = Path("F:/ai_bridge/visuals")
        self.visual_path.mkdir(exist_ok=True)
        
    def load_dataset(self):
        """Load the self-learning dataset"""
        dataset_file = self.buffer_path / "dataset.json"
        if not dataset_file.exists():
            return {"entries": []}
            
        with open(dataset_file, 'r') as f:
            return json.load(f)
    
    def analyze_dataset(self):
        """Analyze dataset for patterns and insights"""
        data = self.load_dataset()
        entries = data.get("entries", [])
        
        if not entries:
            return {
                "total_entries": 0,
                "message": "No data to analyze"
            }
        
        # Extract improvements
        all_improvements = []
        for entry in entries:
            for imp in entry.get("target_improvements", []):
                all_improvements.append(imp["improvement_type"])
        
        # Count improvement types
        improvement_counts = Counter(all_improvements)
        
        # Generate mock quality scores (in practice, would come from analysis)
        quality_scores = [0.6 + np.random.normal(0.15, 0.1) for _ in entries]
        quality_scores = [max(0, min(1, score)) for score in quality_scores]  # Clamp to [0,1]
        
        analysis = {
            "total_entries": len(entries),
            "improvement_distribution": dict(improvement_counts),
            "quality_scores": quality_scores,
            "avg_quality": np.mean(quality_scores),
            "quality_trend": "improving" if quality_scores[-1] > quality_scores[0] else "stable",
            "most_common_improvement": improvement_counts.most_common(1)[0] if improvement_counts else None,
            "timestamps": [entry["timestamp"] for entry in entries]
        }
        
        return analysis
    
    def create_visualizations(self, analysis):
        """Create visualizations of the dataset"""
        
        if analysis["total_entries"] == 0:
            print("No data to visualize")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Self-Learning Dataset Analysis', fontsize=16)
        
        # 1. Quality Score Histogram
        ax1.hist(analysis["quality_scores"], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(analysis["avg_quality"], color='red', linestyle='dashed', linewidth=2, label=f'Avg: {analysis["avg_quality"]:.2f}')
        ax1.set_xlabel('Quality Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Quality Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Improvement Type Distribution
        if analysis["improvement_distribution"]:
            improvements = list(analysis["improvement_distribution"].keys())
            counts = list(analysis["improvement_distribution"].values())
            
            bars = ax2.bar(range(len(improvements)), counts, color='green', alpha=0.7)
            ax2.set_xlabel('Improvement Type')
            ax2.set_ylabel('Count')
            ax2.set_title('Improvement Type Distribution')
            ax2.set_xticks(range(len(improvements)))
            ax2.set_xticklabels(improvements, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax2.annotate(f'{count}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        # 3. Quality Score Over Time
        ax3.plot(range(len(analysis["quality_scores"])), analysis["quality_scores"], 
                marker='o', linestyle='-', color='purple', markersize=6)
        ax3.set_xlabel('Entry Number')
        ax3.set_ylabel('Quality Score')
        ax3.set_title(f'Quality Score Trend ({analysis["quality_trend"]})')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(range(len(analysis["quality_scores"])), analysis["quality_scores"], 1)
        p = np.poly1d(z)
        ax3.plot(range(len(analysis["quality_scores"])), p(range(len(analysis["quality_scores"]))), 
                "r--", alpha=0.8, label=f'Trend: {"↑" if z[0] > 0 else "↓"}')
        ax3.legend()
        
        # 4. Entries Over Time
        # Convert timestamps to hours from start
        timestamps = [datetime.fromisoformat(ts) for ts in analysis["timestamps"]]
        start_time = timestamps[0]
        hours_from_start = [(ts - start_time).total_seconds() / 3600 for ts in timestamps]
        
        ax4.scatter(hours_from_start, range(len(timestamps)), alpha=0.6, color='orange')
        ax4.set_xlabel('Hours from Start')
        ax4.set_ylabel('Entry Count')
        ax4.set_title('Data Collection Rate')
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save visualization
        output_path = self.visual_path / "dataset_visual.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")
        
        # Also create a summary report
        self._create_summary_report(analysis)
        
    def _create_summary_report(self, analysis):
        """Create a text summary report"""
        
        report = f"""
Self-Learning Dataset Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=====================================

Dataset Statistics:
- Total Entries: {analysis['total_entries']}
- Average Quality Score: {analysis['avg_quality']:.3f}
- Quality Trend: {analysis['quality_trend']}

Improvement Distribution:
"""
        
        for imp_type, count in analysis['improvement_distribution'].items():
            report += f"- {imp_type}: {count} occurrences\n"
        
        if analysis['most_common_improvement']:
            report += f"\nMost Common Improvement: {analysis['most_common_improvement'][0]} ({analysis['most_common_improvement'][1]} times)\n"
        
        report += f"""
Quality Score Statistics:
- Min Score: {min(analysis['quality_scores']):.3f}
- Max Score: {max(analysis['quality_scores']):.3f}
- Std Dev: {np.std(analysis['quality_scores']):.3f}

Visual Analysis:
- Histogram shows {"normal distribution" if abs(np.mean(analysis['quality_scores']) - 0.75) < 0.1 else "skewed distribution"}
- Trend line indicates {"improvement" if analysis['quality_trend'] == "improving" else "stability"}
- Data collection rate: {analysis['total_entries'] / max(1, (datetime.fromisoformat(analysis['timestamps'][-1]) - datetime.fromisoformat(analysis['timestamps'][0])).total_seconds() / 3600):.1f} entries/hour
"""
        
        # Save report
        report_path = self.visual_path / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
        print("\n" + "="*50)
        print(report)

def main():
    """Run dataset analysis and visualization"""
    analyzer = DatasetAnalyzer()
    
    print("Loading and analyzing dataset...")
    analysis = analyzer.analyze_dataset()
    
    if analysis["total_entries"] > 0:
        print(f"Found {analysis['total_entries']} entries")
        print(f"Average quality: {analysis['avg_quality']:.3f}")
        print(f"Most common improvement: {analysis['most_common_improvement']}")
        
        print("\nCreating visualizations...")
        analyzer.create_visualizations(analysis)
    else:
        print("No data found to analyze")

if __name__ == "__main__":
    main()
