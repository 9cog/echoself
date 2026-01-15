#!/usr/bin/env python3
"""
Test script for EchoSpace implementation
Validates the core Agent-Arena-Relation architecture
"""

import json
import subprocess
import sys

def test_typescript_compilation():
    """Test that TypeScript compiles without errors"""
    print("🔍 Testing TypeScript compilation...")
    try:
        result = subprocess.run(['npm', 'run', 'typecheck'], 
                              capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            print("✅ TypeScript compilation successful")
            return True
        else:
            print(f"❌ TypeScript compilation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running TypeScript check: {e}")
        return False

def test_build_process():
    """Test that the build process completes successfully"""
    print("🔍 Testing build process...")
    try:
        result = subprocess.run(['npm', 'run', 'build'], 
                              capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            print("✅ Build process successful")
            return True
        else:
            print(f"❌ Build process failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error during build: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("🔍 Testing file structure...")
    required_files = [
        'src/types/EchoSpace.ts',
        'src/services/echoSpaceService.ts', 
        'src/services/echoSpaceWorkflows.ts',
        'src/components/EchoSpaceControlPanel.tsx'
    ]
    
    all_exist = True
    for file_path in required_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if len(content) > 1000:  # Basic content check
                    print(f"✅ {file_path} exists and has content")
                else:
                    print(f"⚠️ {file_path} exists but may be incomplete")
                    all_exist = False
        except FileNotFoundError:
            print(f"❌ {file_path} not found")
            all_exist = False
        except Exception as e:
            print(f"❌ Error checking {file_path}: {e}")
            all_exist = False
    
    return all_exist

def test_echospace_types():
    """Test that EchoSpace types are properly defined"""
    print("🔍 Testing EchoSpace type definitions...")
    
    type_file = 'src/types/EchoSpace.ts'
    required_interfaces = [
        'AgentNamespace',
        'ArenaNamespace', 
        'AgentArenaRelation',
        'MardukNamespaces',
        'EchoCogNamespaces',
        'SimulationRecord',
        'ConsensusState'
    ]
    
    try:
        with open(type_file, 'r') as f:
            content = f.read()
            
        all_found = True
        for interface in required_interfaces:
            if f'interface {interface}' in content:
                print(f"✅ {interface} interface defined")
            else:
                print(f"❌ {interface} interface missing")
                all_found = False
                
        return all_found
    except Exception as e:
        print(f"❌ Error checking types: {e}")
        return False

def test_echospace_service():
    """Test EchoSpace service structure"""
    print("🔍 Testing EchoSpace service...")
    
    service_file = 'src/services/echoSpaceService.ts'
    required_methods = [
        'spawnVirtualMarduk',
        'establishRelation',
        'getAgentRelations',
        'getArenaAgents',
        'getSystemState'
    ]
    
    try:
        with open(service_file, 'r') as f:
            content = f.read()
            
        all_found = True
        for method in required_methods:
            if method in content:
                print(f"✅ {method} method found")
            else:
                print(f"❌ {method} method missing")
                all_found = False
                
        # Check for core namespace initialization
        if 'createMardukNamespaces' in content and 'createEchoCogNamespaces' in content:
            print("✅ Namespace initialization methods found")
        else:
            print("❌ Namespace initialization missing")
            all_found = False
            
        return all_found
    except Exception as e:
        print(f"❌ Error checking service: {e}")
        return False

def test_echospace_workflows():
    """Test EchoSpace workflows"""
    print("🔍 Testing EchoSpace workflows...")
    
    workflow_file = 'src/services/echoSpaceWorkflows.ts'
    required_workflows = [
        'MemoryWorkflow',
        'SandboxWorkflow',
        'VirtualMardukWorkflow', 
        'ConsensusWorkflow',
        'ActualMardukWorkflow',
        'EchoSpaceWorkflowOrchestrator'
    ]
    
    try:
        with open(workflow_file, 'r') as f:
            content = f.read()
            
        all_found = True
        for workflow in required_workflows:
            if f'class {workflow}' in content:
                print(f"✅ {workflow} class found")
            else:
                print(f"❌ {workflow} class missing")
                all_found = False
        
        # Check for main pipeline method
        if 'executeMardukPipeline' in content:
            print("✅ Main pipeline execution method found")
        else:
            print("❌ Main pipeline execution method missing")
            all_found = False
            
        return all_found
    except Exception as e:
        print(f"❌ Error checking workflows: {e}")
        return False

def test_echospace_ui():
    """Test EchoSpace UI component"""
    print("🔍 Testing EchoSpace UI component...")
    
    ui_file = 'src/components/EchoSpaceControlPanel.tsx'
    
    try:
        with open(ui_file, 'r') as f:
            content = f.read()
            
        ui_elements = [
            'EchoSpaceControlPanel',
            'executeMardukPipeline',
            'spawnVirtualMarduk',
            'systemState'
        ]
        
        all_found = True
        for element in ui_elements:
            if element in content:
                print(f"✅ {element} found in UI")
            else:
                print(f"❌ {element} missing from UI")
                all_found = False
                
        return all_found
    except Exception as e:
        print(f"❌ Error checking UI: {e}")
        return False

def test_mapview_integration():
    """Test MapView integration"""
    print("🔍 Testing MapView integration...")
    
    mapview_file = 'src/components/MapView.tsx'
    
    try:
        with open(mapview_file, 'r') as f:
            content = f.read()
            
        integration_elements = [
            'EchoSpaceControlPanel',
            'showEchoSpace',
            'toggleEchoSpace',
            'echospace'
        ]
        
        all_found = True
        for element in integration_elements:
            if element in content:
                print(f"✅ {element} integrated in MapView")
            else:
                print(f"❌ {element} missing from MapView integration")
                all_found = False
                
        return all_found
    except Exception as e:
        print(f"❌ Error checking MapView integration: {e}")
        return False

def test_echohome_map_integration():
    """Test EchoHomeMap integration"""
    print("🔍 Testing EchoHomeMap integration...")
    
    map_file = 'src/components/EchoHomeMap.tsx'
    
    try:
        with open(map_file, 'r') as f:
            content = f.read()
            
        if '"echospace"' in content and 'EchoSpace' in content:
            print("✅ EchoSpace room added to EchoHomeMap")
            return True
        else:
            print("❌ EchoSpace room missing from EchoHomeMap")
            return False
            
    except Exception as e:
        print(f"❌ Error checking EchoHomeMap: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 Running EchoSpace Implementation Tests\n")
    
    tests = [
        test_file_structure,
        test_echospace_types,
        test_echospace_service,
        test_echospace_workflows,
        test_echospace_ui,
        test_mapview_integration,
        test_echohome_map_integration,
        test_typescript_compilation,
        test_build_process
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {e}\n")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! EchoSpace implementation is complete.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)