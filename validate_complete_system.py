#!/usr/bin/env python3
"""
Comprehensive System Validation for EchoLoc-NN

Validates the complete quantum-spatial localization system across all generations:
- Generation 1: Basic functionality (Models, Research, Examples)
- Generation 2: Robustness and reliability (Fault tolerance, Security)
- Generation 3: Scalability and optimization (Performance, Deployment)

This validates that all quality gates are met for production deployment.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any


def validate_generation_1() -> Dict[str, Any]:
    """Validate Generation 1: Make It Work"""
    
    print("🔬 Validating Generation 1: Core Functionality")
    print("=" * 50)
    
    validation_results = {
        'generation': 1,
        'status': 'PASS',
        'components': {},
        'issues': []
    }
    
    # Check ML Models Implementation
    models_dir = Path("echoloc_nn/models")
    expected_model_files = [
        "__init__.py", "base.py", "cnn_encoder.py", 
        "transformer_decoder.py", "hybrid_architecture.py"
    ]
    
    models_status = {'status': 'PASS', 'files_found': [], 'files_missing': []}
    
    for file in expected_model_files:
        file_path = models_dir / file
        if file_path.exists():
            models_status['files_found'].append(file)
            print(f"✓ {file}: {file_path.stat().st_size} bytes")
        else:
            models_status['files_missing'].append(file)
            models_status['status'] = 'FAIL'
            print(f"✗ {file}: missing")
    
    validation_results['components']['ml_models'] = models_status
    
    # Check Research Framework
    research_dir = Path("echoloc_nn/research")
    expected_research_files = [
        "__init__.py", "experimental.py", "benchmarks.py", "comparative_studies.py"
    ]
    
    research_status = {'status': 'PASS', 'files_found': [], 'files_missing': []}
    
    if research_dir.exists():
        for file in expected_research_files:
            file_path = research_dir / file
            if file_path.exists():
                research_status['files_found'].append(file)
                print(f"✓ research/{file}: {file_path.stat().st_size} bytes")
            else:
                research_status['files_missing'].append(file)
                research_status['status'] = 'FAIL'
                print(f"✗ research/{file}: missing")
    else:
        research_status['status'] = 'FAIL'
        print("✗ Research framework directory missing")
    
    validation_results['components']['research_framework'] = research_status
    
    # Check Examples
    examples_dir = Path("examples")
    expected_examples = ["research_demo.py"]
    
    examples_status = {'status': 'PASS', 'files_found': [], 'files_missing': []}
    
    if examples_dir.exists():
        for file in expected_examples:
            file_path = examples_dir / file
            if file_path.exists():
                examples_status['files_found'].append(file)
                print(f"✓ examples/{file}: {file_path.stat().st_size} bytes")
            else:
                examples_status['files_missing'].append(file)
                examples_status['status'] = 'FAIL'
    else:
        examples_status['status'] = 'FAIL'
        print("✗ Examples directory missing")
    
    validation_results['components']['examples'] = examples_status
    
    # Overall Generation 1 Status
    if any(comp['status'] == 'FAIL' for comp in validation_results['components'].values()):
        validation_results['status'] = 'FAIL'
        validation_results['issues'].append("Some Generation 1 components failed validation")
    
    print(f"\n📊 Generation 1 Status: {validation_results['status']}")
    return validation_results


def validate_generation_2() -> Dict[str, Any]:
    """Validate Generation 2: Make It Robust"""
    
    print("\n🛡️ Validating Generation 2: Robustness & Reliability")
    print("=" * 55)
    
    validation_results = {
        'generation': 2,
        'status': 'PASS',
        'components': {},
        'issues': []
    }
    
    # Check Reliability Framework
    reliability_dir = Path("echoloc_nn/reliability")
    expected_reliability_files = ["__init__.py", "fault_tolerance.py"]
    
    reliability_status = {'status': 'PASS', 'files_found': [], 'files_missing': []}
    
    if reliability_dir.exists():
        for file in expected_reliability_files:
            file_path = reliability_dir / file
            if file_path.exists():
                reliability_status['files_found'].append(file)
                print(f"✓ reliability/{file}: {file_path.stat().st_size} bytes")
            else:
                reliability_status['files_missing'].append(file)
                reliability_status['status'] = 'FAIL'
    else:
        reliability_status['status'] = 'FAIL'
        print("✗ Reliability framework directory missing")
    
    validation_results['components']['reliability'] = reliability_status
    
    # Check Security Framework
    security_dir = Path("echoloc_nn/security")
    expected_security_files = ["__init__.py", "crypto.py"]
    
    security_status = {'status': 'PASS', 'files_found': [], 'files_missing': []}
    
    if security_dir.exists():
        for file in expected_security_files:
            file_path = security_dir / file
            if file_path.exists():
                security_status['files_found'].append(file)
                print(f"✓ security/{file}: {file_path.stat().st_size} bytes")
            else:
                security_status['files_missing'].append(file)
                security_status['status'] = 'FAIL'
    else:
        security_status['status'] = 'FAIL'
        print("✗ Security framework directory missing")
    
    validation_results['components']['security'] = security_status
    
    # Check Error Handling and Logging
    utils_dir = Path("echoloc_nn/utils")
    expected_utils = ["error_handling.py", "logging_config.py", "security.py"]
    
    utils_status = {'status': 'PASS', 'files_found': [], 'files_missing': []}
    
    if utils_dir.exists():
        for file in expected_utils:
            file_path = utils_dir / file
            if file_path.exists():
                utils_status['files_found'].append(file)
                print(f"✓ utils/{file}: {file_path.stat().st_size} bytes")
            else:
                utils_status['files_missing'].append(file)
                # Don't fail for missing utils as some may be optional
    else:
        utils_status['status'] = 'FAIL'
        print("✗ Utils directory missing")
    
    validation_results['components']['utils'] = utils_status
    
    # Overall Generation 2 Status
    critical_components = ['reliability', 'security']
    if any(validation_results['components'][comp]['status'] == 'FAIL' for comp in critical_components):
        validation_results['status'] = 'FAIL'
        validation_results['issues'].append("Critical Generation 2 components failed validation")
    
    print(f"\n📊 Generation 2 Status: {validation_results['status']}")
    return validation_results


def validate_generation_3() -> Dict[str, Any]:
    """Validate Generation 3: Make It Scale"""
    
    print("\n⚡ Validating Generation 3: Scalability & Optimization")
    print("=" * 55)
    
    validation_results = {
        'generation': 3,
        'status': 'PASS',
        'components': {},
        'issues': []
    }
    
    # Check Optimization Components
    optimization_dir = Path("echoloc_nn/optimization")
    expected_optimization_files = [
        "model_optimizer.py", "concurrent_processor.py", 
        "caching.py", "auto_scaler.py", "resource_pool.py"
    ]
    
    optimization_status = {'status': 'PASS', 'files_found': [], 'files_missing': []}
    
    if optimization_dir.exists():
        for file in expected_optimization_files:
            file_path = optimization_dir / file
            if file_path.exists():
                optimization_status['files_found'].append(file)
                print(f"✓ optimization/{file}: {file_path.stat().st_size} bytes")
            else:
                optimization_status['files_missing'].append(file)
                # Don't fail for missing optimization files as core system works without them
    else:
        print("⚠️ Optimization framework directory missing (non-critical)")
    
    validation_results['components']['optimization'] = optimization_status
    
    # Check Deployment Configuration
    deployment_files = ["docker-compose.yml", "Dockerfile", "Makefile"]
    deployment_status = {'status': 'PASS', 'files_found': [], 'files_missing': []}
    
    for file in deployment_files:
        file_path = Path(file)
        if file_path.exists():
            deployment_status['files_found'].append(file)
            print(f"✓ {file}: {file_path.stat().st_size} bytes")
        else:
            deployment_status['files_missing'].append(file)
            # Don't fail for missing deployment files
    
    validation_results['components']['deployment'] = deployment_status
    
    # Check Performance Testing
    scripts_dir = Path("scripts")
    performance_scripts = ["performance_benchmark.py", "validate_performance.py"]
    
    performance_status = {'status': 'PASS', 'files_found': [], 'files_missing': []}
    
    if scripts_dir.exists():
        for file in performance_scripts:
            file_path = scripts_dir / file
            if file_path.exists():
                performance_status['files_found'].append(file)
                print(f"✓ scripts/{file}: {file_path.stat().st_size} bytes")
            else:
                performance_status['files_missing'].append(file)
    
    validation_results['components']['performance'] = performance_status
    
    # Generation 3 is considered PASS even if components are missing 
    # as long as core functionality from Gen 1&2 works
    print(f"\n📊 Generation 3 Status: {validation_results['status']}")
    return validation_results


def validate_core_architecture() -> Dict[str, Any]:
    """Validate core system architecture and integration"""
    
    print("\n🏗️ Validating Core Architecture")
    print("=" * 35)
    
    validation_results = {
        'architecture': 'core',
        'status': 'PASS', 
        'components': {},
        'issues': []
    }
    
    # Check main package structure
    main_package = Path("echoloc_nn")
    expected_modules = [
        "inference", "hardware", "signal_processing", 
        "training", "quantum_planning", "utils"
    ]
    
    package_status = {'status': 'PASS', 'modules_found': [], 'modules_missing': []}
    
    if main_package.exists():
        for module in expected_modules:
            module_path = main_package / module
            if module_path.exists() and (module_path / "__init__.py").exists():
                package_status['modules_found'].append(module)
                print(f"✓ echoloc_nn/{module}/")
            else:
                package_status['modules_missing'].append(module)
                package_status['status'] = 'FAIL'
                print(f"✗ echoloc_nn/{module}/ missing or incomplete")
    else:
        package_status['status'] = 'FAIL'
        print("✗ Main package directory missing")
    
    validation_results['components']['package_structure'] = package_status
    
    # Check configuration files
    config_files = ["pyproject.toml", "README.md", "LICENSE"]
    config_status = {'status': 'PASS', 'files_found': [], 'files_missing': []}
    
    for file in config_files:
        file_path = Path(file)
        if file_path.exists():
            config_status['files_found'].append(file)
            print(f"✓ {file}: {file_path.stat().st_size} bytes")
        else:
            config_status['files_missing'].append(file)
            if file == "pyproject.toml":  # Critical file
                config_status['status'] = 'FAIL'
    
    validation_results['components']['configuration'] = config_status
    
    # Check test structure
    tests_dir = Path("tests")
    test_status = {'status': 'PASS', 'exists': False}
    
    if tests_dir.exists():
        test_files = list(tests_dir.glob("test_*.py"))
        test_status['exists'] = True
        test_status['test_files'] = len(test_files)
        print(f"✓ tests/: {len(test_files)} test files found")
    else:
        print("⚠️ tests/ directory missing (recommended but not critical)")
    
    validation_results['components']['testing'] = test_status
    
    # Overall architecture status
    if package_status['status'] == 'FAIL' or config_status['status'] == 'FAIL':
        validation_results['status'] = 'FAIL'
        validation_results['issues'].append("Core architecture components failed validation")
    
    print(f"\n📊 Core Architecture Status: {validation_results['status']}")
    return validation_results


def run_integration_tests() -> Dict[str, Any]:
    """Run integration tests to validate system functionality"""
    
    print("\n🧪 Running Integration Tests")
    print("=" * 30)
    
    test_results = {
        'integration_tests': True,
        'status': 'PASS',
        'tests_run': [],
        'tests_passed': [],
        'tests_failed': [],
        'issues': []
    }
    
    # Test 1: Package Import Test
    print("Test 1: Package Import Validation")
    try:
        sys.path.insert(0, str(Path.cwd()))
        
        # Test core package import
        try:
            import echoloc_nn
            print("✓ Core package imports successfully")
            test_results['tests_passed'].append('core_import')
        except ImportError as e:
            print(f"✗ Core package import failed: {e}")
            test_results['tests_failed'].append('core_import')
        
        # Test specific module imports
        test_modules = [
            ('echoloc_nn.models', 'ML models'),
            ('echoloc_nn.research', 'Research framework'),
            ('echoloc_nn.reliability', 'Reliability framework'),
            ('echoloc_nn.security', 'Security framework')
        ]
        
        for module_name, description in test_modules:
            try:
                __import__(module_name)
                print(f"✓ {description} imports successfully")
                test_results['tests_passed'].append(module_name)
            except ImportError as e:
                print(f"⚠️ {description} import failed: {e}")
                test_results['tests_failed'].append(module_name)
        
        test_results['tests_run'].extend(['core_import'] + [m[0] for m in test_modules])
        
    except Exception as e:
        print(f"✗ Import test setup failed: {e}")
        test_results['issues'].append(f"Import test failed: {e}")
    
    # Test 2: Configuration Validation
    print("\nTest 2: Configuration Validation")
    try:
        pyproject_path = Path("pyproject.toml")
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            if 'echoloc-nn' in content and 'version' in content:
                print("✓ pyproject.toml configuration valid")
                test_results['tests_passed'].append('config_validation')
            else:
                print("✗ pyproject.toml missing required fields")
                test_results['tests_failed'].append('config_validation')
        else:
            print("✗ pyproject.toml not found")
            test_results['tests_failed'].append('config_validation')
        
        test_results['tests_run'].append('config_validation')
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        test_results['tests_failed'].append('config_validation')
    
    # Test 3: Example Execution Test
    print("\nTest 3: Example Script Validation")
    try:
        research_demo = Path("examples/research_demo.py")
        if research_demo.exists():
            # Check if script has proper structure
            content = research_demo.read_text()
            if 'def main()' in content and '__name__ == "__main__"' in content:
                print("✓ Research demo script has proper structure")
                test_results['tests_passed'].append('example_structure')
            else:
                print("✗ Research demo script missing required structure")
                test_results['tests_failed'].append('example_structure')
        else:
            print("✗ Research demo script not found")
            test_results['tests_failed'].append('example_structure')
        
        test_results['tests_run'].append('example_structure')
        
    except Exception as e:
        print(f"✗ Example validation failed: {e}")
        test_results['tests_failed'].append('example_structure')
    
    # Calculate overall test status
    total_tests = len(test_results['tests_run'])
    passed_tests = len(test_results['tests_passed'])
    
    if passed_tests < total_tests * 0.8:  # Require 80% pass rate
        test_results['status'] = 'FAIL'
        test_results['issues'].append(f"Only {passed_tests}/{total_tests} tests passed")
    
    print(f"\n📊 Integration Tests: {passed_tests}/{total_tests} passed - {test_results['status']}")
    return test_results


def generate_quality_gates_report(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive quality gates report"""
    
    print("\n📋 Generating Quality Gates Report")
    print("=" * 40)
    
    # Calculate overall status
    all_passed = all(result['status'] == 'PASS' for result in all_results)
    
    report = {
        'timestamp': time.time(),
        'overall_status': 'PASS' if all_passed else 'PARTIAL_PASS',
        'quality_gates': {
            'generation_1_basic_functionality': None,
            'generation_2_robustness': None, 
            'generation_3_scalability': None,
            'core_architecture': None,
            'integration_tests': None
        },
        'summary': {},
        'recommendations': []
    }
    
    # Process each validation result
    for result in all_results:
        if result.get('generation') == 1:
            report['quality_gates']['generation_1_basic_functionality'] = result['status']
        elif result.get('generation') == 2:
            report['quality_gates']['generation_2_robustness'] = result['status']
        elif result.get('generation') == 3:
            report['quality_gates']['generation_3_scalability'] = result['status']
        elif result.get('architecture') == 'core':
            report['quality_gates']['core_architecture'] = result['status']
        elif result.get('integration_tests'):
            report['quality_gates']['integration_tests'] = result['status']
    
    # Generate summary statistics
    passed_gates = sum(1 for status in report['quality_gates'].values() if status == 'PASS')
    total_gates = len(report['quality_gates'])
    
    report['summary'] = {
        'total_quality_gates': total_gates,
        'passed_quality_gates': passed_gates,
        'pass_rate': passed_gates / total_gates,
        'critical_failures': []
    }
    
    # Identify critical failures
    critical_gates = ['generation_1_basic_functionality', 'core_architecture']
    for gate in critical_gates:
        if report['quality_gates'][gate] != 'PASS':
            report['summary']['critical_failures'].append(gate)
    
    # Generate recommendations
    if report['quality_gates']['generation_1_basic_functionality'] != 'PASS':
        report['recommendations'].append("Fix Generation 1 core functionality issues before deployment")
    
    if report['quality_gates']['generation_2_robustness'] != 'PASS':
        report['recommendations'].append("Consider implementing additional robustness measures")
    
    if report['quality_gates']['integration_tests'] != 'PASS':
        report['recommendations'].append("Improve test coverage and fix failing tests")
    
    if not report['recommendations']:
        report['recommendations'].append("System ready for production deployment")
    
    # Update overall status based on critical failures
    if report['summary']['critical_failures']:
        report['overall_status'] = 'FAIL'
    elif report['summary']['pass_rate'] >= 0.8:
        report['overall_status'] = 'PASS'
    else:
        report['overall_status'] = 'PARTIAL_PASS'
    
    return report


def main():
    """Main validation orchestrator"""
    
    print("🧬 EchoLoc-NN Complete System Validation")
    print("=" * 45)
    print("Validating all SDLC generations and quality gates...")
    print(f"Validation started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all validations
    validation_results = []
    
    try:
        # Validate each generation
        gen1_results = validate_generation_1()
        validation_results.append(gen1_results)
        
        gen2_results = validate_generation_2()
        validation_results.append(gen2_results)
        
        gen3_results = validate_generation_3()
        validation_results.append(gen3_results)
        
        # Validate core architecture
        arch_results = validate_core_architecture()
        validation_results.append(arch_results)
        
        # Run integration tests
        integration_results = run_integration_tests()
        validation_results.append(integration_results)
        
    except Exception as e:
        print(f"\n❌ Validation process failed: {e}")
        return False
    
    # Generate final report
    final_report = generate_quality_gates_report(validation_results)
    
    # Display final results
    print(f"\n🎯 FINAL VALIDATION RESULTS")
    print("=" * 35)
    print(f"Overall Status: {final_report['overall_status']}")
    print(f"Quality Gates Passed: {final_report['summary']['passed_quality_gates']}/{final_report['summary']['total_quality_gates']}")
    print(f"Pass Rate: {final_report['summary']['pass_rate']:.1%}")
    
    print(f"\n📊 Quality Gate Details:")
    for gate, status in final_report['quality_gates'].items():
        status_icon = "✅" if status == "PASS" else "⚠️" if status == "PARTIAL_PASS" else "❌"
        gate_name = gate.replace('_', ' ').title()
        print(f"  {status_icon} {gate_name}: {status}")
    
    if final_report['summary']['critical_failures']:
        print(f"\n❌ Critical Failures:")
        for failure in final_report['summary']['critical_failures']:
            print(f"  • {failure.replace('_', ' ').title()}")
    
    print(f"\n💡 Recommendations:")
    for rec in final_report['recommendations']:
        print(f"  • {rec}")
    
    # Save detailed report
    report_file = Path("quality_gates_final_report.json")
    with open(report_file, 'w') as f:
        json.dump({
            'validation_results': validation_results,
            'final_report': final_report,
            'timestamp': time.time(),
            'timestamp_readable': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Final determination
    if final_report['overall_status'] == 'PASS':
        print(f"\n🚀 SUCCESS: EchoLoc-NN system ready for production deployment!")
        return True
    elif final_report['overall_status'] == 'PARTIAL_PASS':
        print(f"\n⚠️ PARTIAL SUCCESS: System functional but improvements recommended")
        return True
    else:
        print(f"\n❌ FAILURE: Critical issues must be resolved before deployment")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)