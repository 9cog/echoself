#!/usr/bin/env python3
"""
Continuous Improvement Plan Generator for NanEcho Training

This script generates improvement recommendations based on automation analysis.
It replaces inline Python code in the GitHub Actions workflow.
"""

import json
import os
import sys


def generate_improvement_plan(analysis_file: str = "automation_analysis.json") -> None:
    """
    Generate continuous improvement plan based on automation analysis.
    
    Args:
        analysis_file: Path to automation analysis JSON file
    """
    print("🔄 Generating continuous improvement plan...")
    
    if not os.path.exists(analysis_file):
        print('⚠️ No automation analysis found')
        return
    
    try:
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        recommendations = analysis.get('recommendations', {})
        automation_triggers = analysis.get('automation_triggers', {})
        
        print('🚀 Continuous Improvement Summary:')
        print(f'   Training Mode: {analysis.get("training_mode", "unknown")}')
        print(f'   Performance: {analysis.get("overall_fidelity", 0):.3f}')
        
        if recommendations.get('immediate'):
            print(f'   Immediate Actions: {len(recommendations["immediate"])}')
            for rec in recommendations['immediate'][:3]:  # Show first 3
                print(f'     - {rec}')
        
        if automation_triggers.get('trigger_next_training'):
            print(f'   Next Training: +{automation_triggers.get("training_delay_hours", 0)} hours')
        
        print('✅ Improvement plan generated')
        
    except Exception as e:
        print(f'⚠️ Error generating improvement plan: {e}')


def main():
    """Main improvement plan generation function."""
    analysis_file = sys.argv[1] if len(sys.argv) > 1 else "automation_analysis.json"
    
    generate_improvement_plan(analysis_file)


if __name__ == "__main__":
    main()