#!/usr/bin/env python3
"""
Quality Gate Analysis Script for NanEcho Training

This script analyzes automation results and determines next steps for training.
It replaces inline Python code in the GitHub Actions workflow.
"""

import json
import os
import sys


def analyze_quality_gates(analysis_file: str = "automation_analysis.json") -> dict:
    """
    Analyze quality gates and determine next steps.
    
    Args:
        analysis_file: Path to automation analysis JSON file
        
    Returns:
        dict: Analysis results with status and recommendations
    """
    if not os.path.exists(analysis_file):
        print('❌ No automation analysis found')
        return {
            'success': False,
            'error': 'No automation analysis found',
            'quality_gate_passed': False,
            'deployment_ready': False,
            'overall_fidelity': 0.0
        }
    
    try:
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
    except Exception as e:
        print(f'❌ Error loading automation analysis: {e}')
        return {
            'success': False,
            'error': f'Error loading analysis: {e}',
            'quality_gate_passed': False,
            'deployment_ready': False,
            'overall_fidelity': 0.0
        }
    
    # Extract key metrics
    overall_fidelity = analysis.get('overall_fidelity', 0)
    quality_status = analysis.get('quality_gate_status', {}).get('status', 'unknown')
    deployment_ready = analysis.get('quality_gate_status', {}).get('deployment_ready', False)
    next_actions = analysis.get('next_actions', {})
    
    print(f'🎯 Quality Gate Results:')
    print(f'   Overall Fidelity: {overall_fidelity:.3f}')
    print(f'   Quality Gate Status: {quality_status.upper()}')
    print(f'   Deployment Ready: {"✅" if deployment_ready else "❌"}')
    
    # Display next actions
    print(f'🚀 Next Actions:')
    for action, should_do in next_actions.items():
        if should_do:
            print(f'   ✓ {action.replace("_", " ").title()}')
    
    # Create status info for workflow decisions
    status_info = {
        'success': True,
        'quality_gate_passed': quality_status == 'passed',
        'deployment_ready': deployment_ready,
        'overall_fidelity': overall_fidelity,
        'continue_training': next_actions.get('continue_training', False),
        'schedule_next_cycle': next_actions.get('schedule_next_cycle', False)
    }
    
    # Save workflow status
    with open('workflow_status.json', 'w') as f:
        json.dump(status_info, f, indent=2)
    
    print('📊 Workflow status saved for next steps')
    
    return status_info


def main():
    """Main analysis function."""
    analysis_file = sys.argv[1] if len(sys.argv) > 1 else "automation_analysis.json"
    
    print("🎯 Applying quality gates and determining next steps...")
    
    result = analyze_quality_gates(analysis_file)
    
    if not result['success']:
        sys.exit(1)
    
    if result['quality_gate_passed']:
        print("✅ Quality gates passed!")
    else:
        print("⚠️ Quality gates need attention")
    
    return 0


if __name__ == "__main__":
    main()