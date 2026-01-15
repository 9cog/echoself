#!/usr/bin/env python3
"""
Training Cycle Trigger Analysis for NanEcho Training

This script analyzes whether next training cycle should be triggered.
It replaces inline Python code in the GitHub Actions workflow.
"""

import json
import os
import sys


def analyze_training_triggers(analysis_file: str = "automation_analysis.json", 
                            event_name: str = "unknown", 
                            relentless_training: str = "False") -> None:
    """
    Analyze whether next training cycle should be triggered.
    
    Args:
        analysis_file: Path to automation analysis JSON file
        event_name: GitHub event name
        relentless_training: Whether relentless training is enabled
    """
    print("🔄 Checking if next training cycle should be triggered...")
    
    if os.path.exists(analysis_file):
        try:
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            automation_triggers = analysis.get('automation_triggers', {})
            should_trigger = automation_triggers.get('trigger_next_training', False)
            delay_hours = automation_triggers.get('training_delay_hours', 4)
            current_fidelity = analysis.get('overall_fidelity', 0)
            
            if should_trigger:
                print(f'📊 Automation Analysis: Performance {current_fidelity:.3f}')
                print(f'🔄 Next training cycle scheduled for +{delay_hours} hours')
                
                # Create trigger configuration for future automation
                trigger_info = {
                    'trigger_next_cycle': True,
                    'delay_hours': delay_hours,
                    'reason': f'Automation analysis recommends continuation',
                    'current_performance': current_fidelity,
                    'hyperparameter_adjustments': automation_triggers.get('hyperparameter_adjustments', {}),
                    'training_mode': analysis.get('training_mode', 'standard')
                }
                
                with open('next_cycle_trigger.json', 'w') as f:
                    json.dump(trigger_info, f, indent=2)
                
                print('✅ Next cycle automation configured')
                
                # In a production environment, this would trigger the actual workflow
                # For now, we document the automation decision
                print('🤖 NANECHO AUTOMATION: Next training cycle would be triggered')
            else:
                print('✅ Performance satisfactory or no automation trigger needed')
                print(f'   Current fidelity: {current_fidelity:.3f}')
                
        except Exception as e:
            print(f'⚠️ Error analyzing triggers: {e}')
            print('⚠️ Using fallback logic')
            use_fallback_logic(event_name, relentless_training)
    else:
        print('⚠️ No automation analysis found, using default schedule logic')
        use_fallback_logic(event_name, relentless_training)


def use_fallback_logic(event_name: str, relentless_training: str) -> None:
    """
    Use fallback logic when automation analysis is not available.
    
    Args:
        event_name: GitHub event name
        relentless_training: Whether relentless training is enabled
    """
    if event_name == 'schedule':
        print('📅 Scheduled run: Would continue relentless training')
    elif relentless_training == 'True':
        print('🔄 Relentless mode: Would schedule next cycle')
    else:
        print('💤 No trigger conditions met')


def main():
    """Main trigger analysis function."""
    if len(sys.argv) < 3:
        print("Usage: python analyze_training_triggers.py <analysis_file> <event_name> [relentless_training]")
        print("Example: python analyze_training_triggers.py automation_analysis.json schedule True")
        sys.exit(1)
    
    analysis_file = sys.argv[1]
    event_name = sys.argv[2]
    relentless_training = sys.argv[3] if len(sys.argv) > 3 else "False"
    
    analyze_training_triggers(analysis_file, event_name, relentless_training)


if __name__ == "__main__":
    main()