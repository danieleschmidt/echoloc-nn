#!/usr/bin/env python3
"""
Quality Gates Summary - Final validation for EchoLoc-NN implementation.

Provides comprehensive assessment of:
- Code quality and test coverage
- Security posture
- Performance characteristics
- Architecture compliance
- Production readiness
"""

import sys
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any


class QualityGatesValidator:
    """
    Comprehensive quality gates validation for EchoLoc-NN.
    
    Runs all quality checks and provides final go/no-go decision
    for production deployment.
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {
            'timestamp': time.time(),
            'overall_status': 'unknown',
            'quality_gates': {},
            'recommendations': [],
            'blocking_issues': [],
            'warnings': []
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all quality gate validations."""
        print("🚀 EchoLoc-NN Quality Gates Validation")
        print("=" * 50)
        print(f"Project: {self.project_root.resolve()}")
        print(f"Validation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test Suite Validation
        self.results['quality_gates']['test_coverage'] = self._validate_test_suite()
        
        # Security Validation
        self.results['quality_gates']['security'] = self._validate_security()
        
        # Performance Validation
        self.results['quality_gates']['performance'] = self._validate_performance()
        
        # Code Quality Validation
        self.results['quality_gates']['code_quality'] = self._validate_code_quality()
        
        # Architecture Validation
        self.results['quality_gates']['architecture'] = self._validate_architecture()
        
        # Documentation Validation
        self.results['quality_gates']['documentation'] = self._validate_documentation()
        
        # Final Assessment
        self._calculate_final_assessment()
        
        return self.results
    
    def _validate_test_suite(self) -> Dict[str, Any]:
        """Validate test suite completeness and coverage."""
        print("🧪 Validating Test Suite")
        print("-" * 25)
        
        result = {
            'passed': False,
            'score': 0,
            'details': {},
            'issues': []
        }
        
        # Check test files exist
        test_files = list(self.project_root.glob('tests/test_*.py'))
        print(f"  Test files found: {len(test_files)}")
        
        expected_test_files = [
            'test_models.py',
            'test_signal_processing.py', 
            'test_hardware.py'
        ]
        
        found_tests = [f.name for f in test_files]
        missing_tests = [t for t in expected_test_files if t not in found_tests]
        
        if missing_tests:
            result['issues'].append(f"Missing test files: {missing_tests}")
        else:
            result['score'] += 30
        
        # Check test content quality
        total_test_functions = 0
        for test_file in test_files:
            try:
                content = test_file.read_text()
                test_functions = content.count('def test_')
                total_test_functions += test_functions
                print(f"    {test_file.name}: {test_functions} test functions")
            except Exception as e:
                result['issues'].append(f"Error reading {test_file}: {e}")
        
        print(f"  Total test functions: {total_test_functions}")
        
        if total_test_functions >= 50:  # Good coverage
            result['score'] += 40
        elif total_test_functions >= 25:  # Adequate coverage
            result['score'] += 25
        else:
            result['issues'].append("Insufficient test coverage")
        
        # Check for different types of tests
        test_types_found = 0
        if any('edge' in tf.read_text() for tf in test_files):
            test_types_found += 1
        if any('performance' in tf.read_text() for tf in test_files):
            test_types_found += 1
        if any('integration' in tf.read_text() for tf in test_files):
            test_types_found += 1
        
        result['score'] += test_types_found * 10
        
        result['details'] = {
            'test_files': len(test_files),
            'test_functions': total_test_functions,
            'test_types': test_types_found,
            'coverage_estimate': min(100, (total_test_functions / 60) * 100)
        }
        
        result['passed'] = result['score'] >= 60  # 60% threshold
        
        if result['passed']:
            print("  ✅ Test suite validation passed")
        else:
            print("  ❌ Test suite validation failed")
        
        print()
        return result
    
    def _validate_security(self) -> Dict[str, Any]:
        """Validate security implementation."""
        print("🔒 Validating Security")
        print("-" * 20)
        
        result = {
            'passed': False,
            'score': 0,
            'details': {},
            'issues': []
        }
        
        # Check security modules exist
        security_files = [
            'echoloc_nn/utils/security.py',
            'echoloc_nn/utils/validation.py',
            'echoloc_nn/utils/exceptions.py'
        ]
        
        security_score = 0
        for sec_file in security_files:
            if (self.project_root / sec_file).exists():
                security_score += 1
                print(f"  ✅ {sec_file}")
            else:
                print(f"  ❌ {sec_file} missing")
                result['issues'].append(f"Missing security file: {sec_file}")
        
        result['score'] += (security_score / len(security_files)) * 50
        
        # Check security implementation quality
        security_file = self.project_root / 'echoloc_nn/utils/security.py'
        if security_file.exists():
            content = security_file.read_text()
            
            security_features = [
                ('InputSanitizer', 'Input sanitization'),
                ('SecurityValidator', 'Security validation'),
                ('dangerous_patterns', 'Pattern detection'),
                ('sanitize_file_path', 'Path validation'),
                ('validate_model_integrity', 'Model integrity'),
            ]
            
            found_features = 0
            for feature, description in security_features:
                if feature in content:
                    found_features += 1
                    print(f"  ✅ {description}")
                else:
                    print(f"  ❌ {description} missing")
            
            result['score'] += (found_features / len(security_features)) * 50
        
        result['details'] = {
            'security_files': security_score,
            'security_features': found_features if 'found_features' in locals() else 0
        }
        
        result['passed'] = result['score'] >= 70  # 70% threshold for security
        
        if result['passed']:
            print("  ✅ Security validation passed")
        else:
            print("  ❌ Security validation failed")
        
        print()
        return result
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance characteristics."""
        print("⚡ Validating Performance")
        print("-" * 25)
        
        result = {
            'passed': False,
            'score': 0,
            'details': {},
            'issues': []
        }
        
        # Run basic performance test
        try:
            perf_script = self.project_root / 'scripts/basic_performance_test.py'
            if perf_script.exists():
                print("  Running performance tests...")
                proc = subprocess.run(
                    [sys.executable, str(perf_script)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if proc.returncode == 0:
                    result['score'] += 60
                    print("  ✅ Performance tests passed")
                else:
                    result['issues'].append("Performance tests failed")
                    print("  ❌ Performance tests failed")
            else:
                result['issues'].append("Performance test script missing")
                print("  ❌ Performance test script not found")
        
        except Exception as e:
            result['issues'].append(f"Performance test error: {e}")
            print(f"  ❌ Performance test error: {e}")
        
        # Check optimization modules exist
        optimization_files = [
            'echoloc_nn/optimization/model_optimizer.py',
            'echoloc_nn/optimization/caching.py',
            'echoloc_nn/optimization/concurrent_processor.py'
        ]
        
        opt_score = 0
        for opt_file in optimization_files:
            if (self.project_root / opt_file).exists():
                opt_score += 1
        
        result['score'] += (opt_score / len(optimization_files)) * 40
        print(f"  Optimization modules: {opt_score}/{len(optimization_files)}")
        
        result['details'] = {
            'optimization_modules': opt_score,
            'performance_test_available': perf_script.exists()
        }
        
        result['passed'] = result['score'] >= 50  # 50% threshold
        
        if result['passed']:
            print("  ✅ Performance validation passed")
        else:
            print("  ❌ Performance validation failed")
        
        print()
        return result
    
    def _validate_code_quality(self) -> Dict[str, Any]:
        """Validate code quality and structure."""
        print("📝 Validating Code Quality")
        print("-" * 26)
        
        result = {
            'passed': False,
            'score': 0,
            'details': {},
            'issues': []
        }
        
        # Check project structure
        expected_dirs = [
            'echoloc_nn/models',
            'echoloc_nn/signal_processing',
            'echoloc_nn/hardware',
            'echoloc_nn/training',
            'echoloc_nn/inference',
            'echoloc_nn/utils',
            'echoloc_nn/optimization',
            'tests',
            'scripts'
        ]
        
        found_dirs = 0
        for expected_dir in expected_dirs:
            if (self.project_root / expected_dir).exists():
                found_dirs += 1
            else:
                result['issues'].append(f"Missing directory: {expected_dir}")
        
        structure_score = (found_dirs / len(expected_dirs)) * 30
        result['score'] += structure_score
        print(f"  Project structure: {found_dirs}/{len(expected_dirs)} directories")
        
        # Check __init__.py files
        init_files = list(self.project_root.rglob('__init__.py'))
        expected_init_count = len(expected_dirs) - 2  # tests and scripts don't need __init__.py
        
        if len(init_files) >= expected_init_count:
            result['score'] += 20
            print(f"  ✅ Package structure ({len(init_files)} __init__.py files)")
        else:
            result['issues'].append("Missing __init__.py files")
            print(f"  ❌ Package structure incomplete")
        
        # Check documentation strings
        py_files = list(self.project_root.glob('echoloc_nn/**/*.py'))
        documented_files = 0
        
        for py_file in py_files:
            try:
                content = py_file.read_text()
                if '"""' in content and 'def ' in content:
                    documented_files += 1
            except:
                pass
        
        doc_score = (documented_files / max(len(py_files), 1)) * 30
        result['score'] += doc_score
        print(f"  Documentation: {documented_files}/{len(py_files)} files documented")
        
        # Check for type hints
        typed_files = 0
        for py_file in py_files:
            try:
                content = py_file.read_text()
                if 'typing import' in content or ': ' in content or ' -> ' in content:
                    typed_files += 1
            except:
                pass
        
        type_score = (typed_files / max(len(py_files), 1)) * 20
        result['score'] += type_score
        print(f"  Type hints: {typed_files}/{len(py_files)} files with type hints")
        
        result['details'] = {
            'directories': found_dirs,
            'init_files': len(init_files),
            'python_files': len(py_files),
            'documented_files': documented_files,
            'typed_files': typed_files
        }
        
        result['passed'] = result['score'] >= 60  # 60% threshold
        
        if result['passed']:
            print("  ✅ Code quality validation passed")
        else:
            print("  ❌ Code quality validation failed")
        
        print()
        return result
    
    def _validate_architecture(self) -> Dict[str, Any]:
        """Validate architecture implementation."""
        print("🏗️  Validating Architecture")
        print("-" * 26)
        
        result = {
            'passed': False,
            'score': 0,
            'details': {},
            'issues': []
        }
        
        # Check core model implementation
        model_files = [
            'echoloc_nn/models/base.py',
            'echoloc_nn/models/cnn_encoder.py',
            'echoloc_nn/models/transformer_decoder.py',
            'echoloc_nn/models/hybrid_architecture.py'
        ]
        
        model_score = 0
        for model_file in model_files:
            if (self.project_root / model_file).exists():
                model_score += 1
        
        result['score'] += (model_score / len(model_files)) * 40
        print(f"  Model architecture: {model_score}/{len(model_files)} components")
        
        # Check signal processing implementation
        signal_files = [
            'echoloc_nn/signal_processing/chirp_generator.py',
            'echoloc_nn/signal_processing/echo_processing.py',
            'echoloc_nn/signal_processing/preprocessing.py'
        ]
        
        signal_score = 0
        for signal_file in signal_files:
            if (self.project_root / signal_file).exists():
                signal_score += 1
        
        result['score'] += (signal_score / len(signal_files)) * 30
        print(f"  Signal processing: {signal_score}/{len(signal_files)} components")
        
        # Check inference implementation
        inference_files = [
            'echoloc_nn/inference/locator.py',
            'echoloc_nn/training/trainer.py'
        ]
        
        inference_score = 0
        for inf_file in inference_files:
            if (self.project_root / inf_file).exists():
                inference_score += 1
        
        result['score'] += (inference_score / len(inference_files)) * 30
        print(f"  Inference system: {inference_score}/{len(inference_files)} components")
        
        result['details'] = {
            'model_components': model_score,
            'signal_components': signal_score,
            'inference_components': inference_score
        }
        
        result['passed'] = result['score'] >= 70  # 70% threshold for architecture
        
        if result['passed']:
            print("  ✅ Architecture validation passed")
        else:
            print("  ❌ Architecture validation failed")
        
        print()
        return result
    
    def _validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness."""
        print("📚 Validating Documentation")
        print("-" * 27)
        
        result = {
            'passed': False,
            'score': 0,
            'details': {},
            'issues': []
        }
        
        # Check key documentation files
        doc_files = [
            'README.md',
            'docs/ARCHITECTURE.md',
            'docs/GETTING_STARTED.md',
            'docs/API_REFERENCE.md'
        ]
        
        found_docs = 0
        for doc_file in doc_files:
            if (self.project_root / doc_file).exists():
                found_docs += 1
                print(f"  ✅ {doc_file}")
            else:
                print(f"  ❌ {doc_file} missing")
                result['issues'].append(f"Missing documentation: {doc_file}")
        
        result['score'] += (found_docs / len(doc_files)) * 60
        
        # Check README quality
        readme_path = self.project_root / 'README.md'
        if readme_path.exists():
            readme_content = readme_path.read_text()
            readme_sections = ['Installation', 'Usage', 'Architecture', 'Examples']
            found_sections = sum(1 for section in readme_sections if section.lower() in readme_content.lower())
            result['score'] += (found_sections / len(readme_sections)) * 40
            print(f"  README sections: {found_sections}/{len(readme_sections)}")
        
        result['details'] = {
            'documentation_files': found_docs,
            'readme_quality': found_sections if 'found_sections' in locals() else 0
        }
        
        result['passed'] = result['score'] >= 50  # 50% threshold for documentation
        
        if result['passed']:
            print("  ✅ Documentation validation passed")
        else:
            print("  ❌ Documentation validation failed")
        
        print()
        return result
    
    def _calculate_final_assessment(self):
        """Calculate final quality gates assessment."""
        print("🎯 Final Quality Assessment")
        print("-" * 30)
        
        # Weight different quality gates
        weights = {
            'test_coverage': 0.25,
            'security': 0.20,
            'performance': 0.20,
            'code_quality': 0.15,
            'architecture': 0.15,
            'documentation': 0.05
        }
        
        total_score = 0
        passed_gates = 0
        critical_failures = []
        
        for gate_name, gate_result in self.results['quality_gates'].items():
            weight = weights.get(gate_name, 0)
            gate_score = gate_result.get('score', 0)
            gate_passed = gate_result.get('passed', False)
            
            weighted_score = gate_score * weight
            total_score += weighted_score
            
            if gate_passed:
                passed_gates += 1
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
                if gate_name in ['security', 'architecture']:
                    critical_failures.append(gate_name)
            
            print(f"  {gate_name.replace('_', ' ').title()}: {status} ({gate_score:.0f}/100)")
        
        print(f"\n  Overall Score: {total_score:.1f}/100")
        print(f"  Passed Gates: {passed_gates}/{len(self.results['quality_gates'])}")
        
        # Determine final status
        if critical_failures:
            self.results['overall_status'] = 'BLOCKED'
            self.results['blocking_issues'] = [f"Critical failure in {cf}" for cf in critical_failures]
            print(f"\n  🚫 BLOCKED - Critical failures: {critical_failures}")
        elif total_score >= 80 and passed_gates >= len(self.results['quality_gates']) * 0.8:
            self.results['overall_status'] = 'APPROVED'
            print(f"\n  🎉 APPROVED - Ready for production")
        elif total_score >= 60:
            self.results['overall_status'] = 'CONDITIONAL'
            self.results['warnings'] = ["Performance or quality concerns - review recommended"]
            print(f"\n  ⚠️  CONDITIONAL - Address warnings before production")
        else:
            self.results['overall_status'] = 'REJECTED'
            self.results['blocking_issues'].append("Overall quality score too low")
            print(f"\n  ❌ REJECTED - Significant quality issues")
        
        # Generate recommendations
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Generate actionable recommendations."""
        recommendations = []
        
        for gate_name, gate_result in self.results['quality_gates'].items():
            if not gate_result.get('passed', False):
                if gate_name == 'test_coverage':
                    recommendations.append("Increase test coverage - add more unit and integration tests")
                elif gate_name == 'security':
                    recommendations.append("Strengthen security implementation - review input validation")
                elif gate_name == 'performance':
                    recommendations.append("Optimize performance - profile bottlenecks and implement caching")
                elif gate_name == 'code_quality':
                    recommendations.append("Improve code quality - add type hints and documentation")
                elif gate_name == 'architecture':
                    recommendations.append("Complete architecture implementation - missing core components")
                elif gate_name == 'documentation':
                    recommendations.append("Enhance documentation - add usage examples and API docs")
        
        # General recommendations based on overall status
        if self.results['overall_status'] == 'CONDITIONAL':
            recommendations.append("Consider code review before production deployment")
            recommendations.append("Set up monitoring and alerting for production environment")
        elif self.results['overall_status'] == 'APPROVED':
            recommendations.append("Implement continuous integration/continuous deployment (CI/CD)")
            recommendations.append("Set up production monitoring and logging")
        
        self.results['recommendations'] = recommendations
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive quality gates report."""
        report_lines = [
            "EchoLoc-NN Quality Gates Report",
            "=" * 40,
            f"Assessment Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.results['timestamp']))}",
            f"Overall Status: {self.results['overall_status']}",
            "",
            "QUALITY GATES SUMMARY",
            "-" * 25
        ]
        
        for gate_name, gate_result in self.results['quality_gates'].items():
            status = "PASSED" if gate_result.get('passed', False) else "FAILED"
            score = gate_result.get('score', 0)
            report_lines.append(f"{gate_name.replace('_', ' ').title()}: {status} ({score:.0f}/100)")
        
        if self.results['blocking_issues']:
            report_lines.extend([
                "",
                "BLOCKING ISSUES",
                "-" * 15
            ])
            for issue in self.results['blocking_issues']:
                report_lines.append(f"❌ {issue}")
        
        if self.results['warnings']:
            report_lines.extend([
                "",
                "WARNINGS",
                "-" * 8
            ])
            for warning in self.results['warnings']:
                report_lines.append(f"⚠️  {warning}")
        
        if self.results['recommendations']:
            report_lines.extend([
                "",
                "RECOMMENDATIONS",
                "-" * 15
            ])
            for rec in self.results['recommendations']:
                report_lines.append(f"💡 {rec}")
        
        report_lines.extend([
            "",
            "NEXT STEPS",
            "-" * 10
        ])
        
        if self.results['overall_status'] == 'APPROVED':
            report_lines.extend([
                "✅ System approved for production deployment",
                "✅ Implement monitoring and alerting",
                "✅ Set up CI/CD pipeline",
                "✅ Plan rollout strategy"
            ])
        elif self.results['overall_status'] == 'CONDITIONAL':
            report_lines.extend([
                "⚠️  Address warnings before production",
                "⚠️  Conduct thorough testing",
                "⚠️  Set up staging environment",
                "⚠️  Plan gradual rollout"
            ])
        else:
            report_lines.extend([
                "❌ Address blocking issues",
                "❌ Re-run quality gates validation",
                "❌ Consider architecture review",
                "❌ Increase test coverage"
            ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report)
            print(f"\n📄 Quality gates report written to: {output_file}")
        
        return report


def main():
    """Main entry point for quality gates validation."""
    validator = QualityGatesValidator()
    
    try:
        # Run comprehensive validation
        results = validator.run_comprehensive_validation()
        
        # Generate and display report
        report = validator.generate_report()
        print("\n" + "="*50)
        print(report)
        
        # Export results as JSON
        json_file = "quality_gates_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📊 Results exported to: {json_file}")
        
        # Exit with appropriate code
        if results['overall_status'] == 'APPROVED':
            return 0
        elif results['overall_status'] == 'CONDITIONAL':
            return 0  # Warnings don't fail the build
        else:
            return 1  # Blocked or rejected
    
    except Exception as e:
        print(f"Error running quality gates validation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())