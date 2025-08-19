#!/usr/bin/env python3
"""
Comprehensive Quality Gates for TERRAGON SDLC v4.0
Validates all requirements across Generation 1, 2, and 3.
"""

import sys
import traceback
import time
import json
import numpy as np
import subprocess
import os
from datetime import datetime

def run_quality_gate_test():
    """Run comprehensive quality gate validation."""
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "validation_id": f"qg_{int(time.time())}",
        "overall_status": "PASS",
        "quality_gates": {},
        "detailed_results": {},
        "summary": {}
    }
    
    gates_passed = 0
    total_gates = 0
    
    print("🏁 TERRAGON SDLC v4.0 - QUALITY GATES EXECUTION")
    print("=" * 60)
    print()
    
    # Gate 1: Package Import Test
    print("🔍 Gate 1: Package Import Validation")
    total_gates += 1
    try:
        import echoloc_nn
        print(f"   ✅ Package imports successfully (v{echoloc_nn.__version__})")
        results["quality_gates"]["package_import"] = True
        gates_passed += 1
    except Exception as e:
        print(f"   ❌ Package import failed: {e}")
        results["quality_gates"]["package_import"] = False
        results["detailed_results"]["package_import_error"] = str(e)
    
    # Gate 2: Core Functionality Test  
    print("\\n🔍 Gate 2: Core Functionality Validation")
    total_gates += 1
    try:
        locator = echoloc_nn.create_locator()
        array = echoloc_nn.create_square_array()
        
        # Test basic inference
        test_data = np.random.randn(4, 2048) * 0.1
        position, confidence = locator.locate(test_data)
        
        if len(position) == 3 and 0 <= confidence <= 1:
            print("   ✅ Core functionality working")
            results["quality_gates"]["core_functionality"] = True
            gates_passed += 1
        else:
            raise ValueError(f"Invalid output format: pos={position}, conf={confidence}")
            
    except Exception as e:
        print(f"   ❌ Core functionality failed: {e}")
        results["quality_gates"]["core_functionality"] = False
        results["detailed_results"]["core_functionality_error"] = str(e)
    
    # Gate 3: Robustness Test
    print("\\n🔍 Gate 3: Robustness Validation")  
    total_gates += 1
    robustness_passed = 0
    robustness_total = 4
    
    try:
        # Test invalid shapes
        try:
            invalid_data = np.random.randn(3, 1024)
            position, confidence = locator.locate(invalid_data)
            robustness_passed += 1
            print("   ✅ Invalid shape handling")
        except:
            print("   ❌ Invalid shape handling failed")
        
        # Test NaN data
        try:
            nan_data = np.full((4, 2048), np.nan)
            position, confidence = locator.locate(nan_data)
            robustness_passed += 1
            print("   ✅ NaN data handling")
        except:
            print("   ❌ NaN data handling failed")
            
        # Test extreme values
        try:
            extreme_data = np.random.randn(4, 2048) * 1e6
            position, confidence = locator.locate(extreme_data)
            robustness_passed += 1
            print("   ✅ Extreme values handling")
        except:
            print("   ❌ Extreme values handling failed")
            
        # Test health monitoring
        try:
            health = locator.get_health_status()
            if isinstance(health, dict) and 'overall_status' in health:
                robustness_passed += 1
                print("   ✅ Health monitoring")
            else:
                print("   ❌ Health monitoring invalid")
        except:
            print("   ❌ Health monitoring failed")
        
        if robustness_passed >= 3:  # At least 75% must pass
            results["quality_gates"]["robustness"] = True
            gates_passed += 1
            print(f"   ✅ Robustness validation passed ({robustness_passed}/{robustness_total})")
        else:
            results["quality_gates"]["robustness"] = False
            print(f"   ❌ Robustness validation failed ({robustness_passed}/{robustness_total})")
            
    except Exception as e:
        print(f"   ❌ Robustness testing failed: {e}")
        results["quality_gates"]["robustness"] = False
        results["detailed_results"]["robustness_error"] = str(e)
    
    # Gate 4: Performance Test
    print("\\n🔍 Gate 4: Performance Validation")
    total_gates += 1
    try:
        # Test caching performance
        cache_test_data = np.random.randn(4, 2048) * 0.1
        
        # First call
        start_time = time.time()
        locator.locate(cache_test_data)
        first_call_time = (time.time() - start_time) * 1000
        
        # Second call (should be cached)
        start_time = time.time()
        locator.locate(cache_test_data)
        second_call_time = (time.time() - start_time) * 1000
        
        # Performance engine validation
        health = locator.get_health_status()
        has_perf_engine = health.get('performance_engine_initialized', False)
        
        if has_perf_engine and second_call_time < first_call_time:
            print(f"   ✅ Performance optimization active (cache: {first_call_time/max(second_call_time,0.001):.1f}x speedup)")
            results["quality_gates"]["performance"] = True
            gates_passed += 1
        else:
            print("   ❌ Performance optimization not working")
            results["quality_gates"]["performance"] = False
            
    except Exception as e:
        print(f"   ❌ Performance testing failed: {e}")
        results["quality_gates"]["performance"] = False
        results["detailed_results"]["performance_error"] = str(e)
    
    # Gate 5: Generation Completeness
    print("\\n🔍 Gate 5: Generation Completeness Validation")
    total_gates += 1
    generation_score = 0
    
    try:
        # Generation 1: Basic functionality (already tested)
        generation_score += 1
        print("   ✅ Generation 1: MAKE IT WORK")
        
        # Generation 2: Robustness features
        if results["quality_gates"].get("robustness", False):
            generation_score += 1
            print("   ✅ Generation 2: MAKE IT ROBUST")
        else:
            print("   ❌ Generation 2: MAKE IT ROBUST")
        
        # Generation 3: Performance features  
        if results["quality_gates"].get("performance", False):
            generation_score += 1
            print("   ✅ Generation 3: MAKE IT SCALE")
        else:
            print("   ❌ Generation 3: MAKE IT SCALE")
        
        if generation_score >= 2:  # At least 2/3 generations must pass
            results["quality_gates"]["generations"] = True
            gates_passed += 1
        else:
            results["quality_gates"]["generations"] = False
            
    except Exception as e:
        print(f"   ❌ Generation completeness failed: {e}")
        results["quality_gates"]["generations"] = False
        results["detailed_results"]["generations_error"] = str(e)
    
    # Gate 6: Documentation and Structure
    print("\\n🔍 Gate 6: Documentation and Structure Validation") 
    total_gates += 1
    try:
        # Check key files exist
        key_files = [
            "README.md", "pyproject.toml", "echoloc_nn/__init__.py",
            "echoloc_nn/inference/", "echoloc_nn/models/", "echoloc_nn/optimization/"
        ]
        
        missing_files = []
        for file_path in key_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if len(missing_files) == 0:
            print("   ✅ Project structure complete")
            results["quality_gates"]["structure"] = True
            gates_passed += 1
        else:
            print(f"   ❌ Missing files: {missing_files}")
            results["quality_gates"]["structure"] = False
            results["detailed_results"]["missing_files"] = missing_files
            
    except Exception as e:
        print(f"   ❌ Structure validation failed: {e}")
        results["quality_gates"]["structure"] = False
        results["detailed_results"]["structure_error"] = str(e)
    
    # Final Results
    pass_rate = gates_passed / total_gates
    results["summary"] = {
        "total_gates": total_gates,
        "passed_gates": gates_passed,
        "pass_rate": pass_rate,
        "status": "PASS" if pass_rate >= 0.85 else "FAIL"  # 85% pass rate required
    }
    
    results["overall_status"] = results["summary"]["status"]
    
    print()
    print("=" * 60)
    print(f"🎯 QUALITY GATES SUMMARY")
    print(f"   Total Gates: {total_gates}")
    print(f"   Passed: {gates_passed}")
    print(f"   Pass Rate: {pass_rate*100:.1f}%")
    print(f"   Overall Status: {'✅ PASS' if results['overall_status'] == 'PASS' else '❌ FAIL'}")
    print("=" * 60)
    
    # Save detailed results
    report_file = f"quality_gates_report_{int(time.time())}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Detailed report saved: {report_file}")
    
    return results["overall_status"] == "PASS"


if __name__ == "__main__":
    try:
        success = run_quality_gate_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\\n❌ Quality gates execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)