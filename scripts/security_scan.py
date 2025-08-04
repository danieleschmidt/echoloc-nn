#!/usr/bin/env python3
"""
Security scanning and validation script for EchoLoc-NN.

Performs comprehensive security checks including:
- Input validation testing
- File path security validation
- Model integrity checking
- Configuration security analysis
- Dependency vulnerability scanning
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from echoloc_nn.utils.security import SecurityValidator, InputSanitizer, SecureConfigLoader
from echoloc_nn.utils.exceptions import SecurityError
from echoloc_nn.utils.logging_config import get_logger
import numpy as np
import torch


class SecurityScanner:
    """
    Comprehensive security scanner for EchoLoc-NN system.
    
    Performs automated security testing and validation across
    all system components.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.logger = get_logger('security_scanner')
        
        # Initialize security components
        self.validator = SecurityValidator()
        self.sanitizer = InputSanitizer()
        self.config_loader = SecureConfigLoader(self.validator)
        
        # Scan results
        self.scan_results = {
            'timestamp': time.time(),
            'project_root': str(self.project_root),
            'total_issues': 0,
            'critical_issues': 0,
            'warnings': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'tests': {}
        }
    
    def run_full_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scan."""
        self.logger.info("Starting comprehensive security scan")
        
        # Input validation tests
        self.scan_results['tests']['input_validation'] = self._test_input_validation()
        
        # File path security tests
        self.scan_results['tests']['file_path_security'] = self._test_file_path_security()
        
        # Configuration security tests
        self.scan_results['tests']['config_security'] = self._test_config_security()
        
        # Model integrity tests
        self.scan_results['tests']['model_integrity'] = self._test_model_integrity()
        
        # Code security analysis
        self.scan_results['tests']['code_analysis'] = self._analyze_code_security()
        
        # Dependency security
        self.scan_results['tests']['dependency_security'] = self._check_dependency_security()
        
        # Network security
        self.scan_results['tests']['network_security'] = self._test_network_security()
        
        # Calculate summary statistics
        self._calculate_summary_stats()
        
        self.logger.info(f"Security scan completed: {self.scan_results['total_issues']} issues found")
        
        return self.scan_results
    
    def _test_input_validation(self) -> Dict[str, Any]:
        """Test input validation and sanitization."""
        self.logger.info("Testing input validation")
        
        results = {
            'test_name': 'Input Validation',
            'passed': True,
            'issues': [],
            'tests_run': 0,
            'tests_passed': 0
        }
        
        # Test dangerous string patterns
        dangerous_inputs = [
            '../../../etc/passwd',
            '__import__("os").system("rm -rf /")',
            'eval("print(1)")',
            'exec("import os")',
            '<script>alert("xss")</script>',
            'javascript:alert(1)',
            '../../../../windows/system32',
            'subprocess.call(["rm", "-rf", "/"])'
        ]
        
        for dangerous_input in dangerous_inputs:
            results['tests_run'] += 1
            try:
                self.sanitizer.sanitize_string_input(dangerous_input)
                # Should not reach here - input should be rejected
                results['issues'].append(f"Dangerous input not caught: {dangerous_input[:50]}")
                results['passed'] = False
            except SecurityError:
                # Good - input was properly rejected
                results['tests_passed'] += 1
            except Exception as e:
                results['issues'].append(f"Unexpected error on input validation: {e}")
                results['passed'] = False
        
        # Test oversized inputs
        oversized_input = "A" * 10000
        results['tests_run'] += 1
        try:
            self.sanitizer.sanitize_string_input(oversized_input, max_length=1000)
            results['issues'].append("Oversized input not rejected")
            results['passed'] = False
        except SecurityError:
            results['tests_passed'] += 1
        except Exception as e:
            results['issues'].append(f"Unexpected error on oversized input: {e}")
            results['passed'] = False
        
        # Test NumPy array validation
        results['tests_run'] += 1
        try:
            # Create oversized array (should be rejected)
            large_array = np.ones((10000, 10000), dtype=np.float64)  # ~800MB
            self.sanitizer.validate_numpy_array(large_array, max_size_mb=100)
            results['issues'].append("Oversized NumPy array not rejected")
            results['passed'] = False
        except SecurityError:
            results['tests_passed'] += 1
        except Exception as e:
            results['issues'].append(f"Unexpected error on array validation: {e}")
            results['passed'] = False
        
        # Test array with non-finite values
        results['tests_run'] += 1
        try:
            bad_array = np.array([1.0, 2.0, np.inf, 4.0])
            self.sanitizer.validate_numpy_array(bad_array)
            results['issues'].append("Array with non-finite values not rejected")
            results['passed'] = False
        except SecurityError:
            results['tests_passed'] += 1
        except Exception as e:
            results['issues'].append(f"Unexpected error on non-finite array: {e}")
            results['passed'] = False
        
        return results
    
    def _test_file_path_security(self) -> Dict[str, Any]:
        """Test file path sanitization and validation."""
        self.logger.info("Testing file path security")
        
        results = {
            'test_name': 'File Path Security',
            'passed': True,
            'issues': [],
            'tests_run': 0,
            'tests_passed': 0
        }
        
        # Create test file for valid path tests
        test_file = self.project_root / "test_file.txt"
        test_file.write_text("test content")
        
        try:
            # Test directory traversal attempts
            dangerous_paths = [
                "../../../etc/passwd",
                "..\\..\\windows\\system32\\config",
                "/etc/shadow",
                "C:\\Windows\\System32\\config\\SAM",
                "file:///etc/passwd",
                "//server/share/file.txt"
            ]
            
            for dangerous_path in dangerous_paths:
                results['tests_run'] += 1
                try:
                    self.sanitizer.sanitize_file_path(dangerous_path)
                    results['issues'].append(f"Dangerous path not caught: {dangerous_path}")
                    results['passed'] = False
                except SecurityError:
                    results['tests_passed'] += 1
                except Exception as e:
                    results['issues'].append(f"Unexpected error on path validation: {e}")
                    results['passed'] = False
            
            # Test base directory restriction
            results['tests_run'] += 1
            try:
                # Try to access file outside project root
                self.sanitizer.sanitize_file_path(
                    str(test_file),
                    base_directory="/tmp"  # Different directory
                )
                results['issues'].append("Base directory restriction not enforced")
                results['passed'] = False
            except SecurityError:
                results['tests_passed'] += 1
            except Exception as e:
                results['issues'].append(f"Unexpected error on base directory test: {e}")
                results['passed'] = False
            
            # Test valid path (should pass)
            results['tests_run'] += 1
            try:
                sanitized_path = self.sanitizer.sanitize_file_path(
                    str(test_file),
                    base_directory=str(self.project_root)
                )
                if Path(sanitized_path).exists():
                    results['tests_passed'] += 1
                else:
                    results['issues'].append("Valid path sanitization failed")
                    results['passed'] = False
            except Exception as e:
                results['issues'].append(f"Valid path test failed: {e}")
                results['passed'] = False
        
        finally:
            # Clean up test file
            if test_file.exists():
                test_file.unlink()
        
        return results
    
    def _test_config_security(self) -> Dict[str, Any]:
        """Test configuration security validation."""
        self.logger.info("Testing configuration security")
        
        results = {
            'test_name': 'Configuration Security',
            'passed': True,
            'issues': [],
            'tests_run': 0,
            'tests_passed': 0
        }
        
        # Test dangerous configuration values
        dangerous_configs = [
            {
                'model_path': '../../../etc/passwd',
                'log_dir': '/tmp/$(rm -rf /)'
            },
            {
                'command': 'rm -rf /',
                'script': 'eval("malicious_code")'
            },
            {
                'path': '__import__("os").system("malicious")',
                'value': 1e20  # Extremely large value
            }
        ]
        
        for config in dangerous_configs:
            results['tests_run'] += 1
            validation_result = self.validator.validate_configuration(config)
            
            if validation_result['is_safe']:
                results['issues'].append(f"Dangerous configuration not caught: {list(config.keys())}")
                results['passed'] = False
            else:
                results['tests_passed'] += 1
        
        # Test safe configuration (should pass)
        safe_config = {
            'model_size': 'base',
            'batch_size': 32,
            'learning_rate': 0.001,
            'device': 'cuda'
        }
        
        results['tests_run'] += 1
        validation_result = self.validator.validate_configuration(safe_config)
        
        if validation_result['is_safe']:
            results['tests_passed'] += 1
        else:
            results['issues'].append("Safe configuration incorrectly flagged as dangerous")
            results['passed'] = False
        
        return results
    
    def _test_model_integrity(self) -> Dict[str, Any]:
        """Test model integrity validation."""
        self.logger.info("Testing model integrity")
        
        results = {
            'test_name': 'Model Integrity',
            'passed': True,
            'issues': [],
            'tests_run': 0,
            'tests_passed': 0
        }
        
        # Create test model file
        test_model_path = self.project_root / "test_model.pt"
        
        try:
            # Create legitimate model
            test_model = torch.nn.Linear(10, 1)
            torch.save(test_model.state_dict(), test_model_path)
            
            results['tests_run'] += 1
            try:
                integrity_result = self.validator.validate_model_integrity(str(test_model_path))
                
                if integrity_result['is_safe'] and integrity_result['is_pytorch_model']:
                    results['tests_passed'] += 1
                else:
                    results['issues'].append("Legitimate model failed integrity check")
                    results['passed'] = False
            except Exception as e:
                results['issues'].append(f"Model integrity check failed: {e}")
                results['passed'] = False
            
            # Test with potentially malicious content
            malicious_data = {
                'state_dict': test_model.state_dict(),
                'eval': 'malicious_code',
                '__import__': 'os'
            }
            
            malicious_model_path = self.project_root / "malicious_model.pt"
            torch.save(malicious_data, malicious_model_path)
            
            results['tests_run'] += 1
            try:
                integrity_result = self.validator.validate_model_integrity(str(malicious_model_path))
                
                if not integrity_result['is_safe']:
                    results['tests_passed'] += 1
                else:
                    results['issues'].append("Malicious model not detected")
                    results['passed'] = False
            except Exception as e:
                results['issues'].append(f"Malicious model test failed: {e}")
                results['passed'] = False
            finally:
                if malicious_model_path.exists():
                    malicious_model_path.unlink()
        
        finally:
            # Clean up test files
            if test_model_path.exists():
                test_model_path.unlink()
        
        return results
    
    def _analyze_code_security(self) -> Dict[str, Any]:
        """Analyze code for security issues."""
        self.logger.info("Analyzing code security")
        
        results = {
            'test_name': 'Code Security Analysis',
            'passed': True,
            'issues': [],
            'tests_run': 0,
            'tests_passed': 0
        }
        
        # Patterns to look for in code
        security_patterns = [
            (r'eval\s*\(', 'Use of eval() function'),
            (r'exec\s*\(', 'Use of exec() function'),
            (r'subprocess\.call\s*\(', 'Direct subprocess calls'),
            (r'os\.system\s*\(', 'Use of os.system()'),
            (r'__import__\s*\(', 'Dynamic imports'),
            (r'open\s*\([^)]*["\']w["\']', 'File writing operations'),
            (r'pickle\.loads?\s*\(', 'Pickle deserialization'),
        ]
        
        # Scan Python files in the project
        python_files = list(self.project_root.rglob("*.py"))
        files_scanned = 0
        
        for py_file in python_files:
            # Skip test files and this scanner
            if 'test' in py_file.name.lower() or py_file.name == 'security_scan.py':
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                files_scanned += 1
                
                for pattern, description in security_patterns:
                    import re
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        # Check if it's in a safe context (comments, strings, etc.)
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if re.search(pattern, line, re.IGNORECASE):
                                # Check if it's in a comment or docstring
                                stripped = line.strip()
                                if not (stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''")):
                                    results['issues'].append(
                                        f"{description} in {py_file.relative_to(self.project_root)}:{i+1}"
                                    )
                                    results['passed'] = False
            
            except Exception as e:
                results['issues'].append(f"Error scanning {py_file}: {e}")
        
        results['tests_run'] = files_scanned
        results['tests_passed'] = files_scanned - len([i for i in results['issues'] if 'Error scanning' not in i])
        
        # Check for secure coding practices
        security_files = [
            self.project_root / 'echoloc_nn' / 'utils' / 'security.py',
            self.project_root / 'echoloc_nn' / 'utils' / 'validation.py'
        ]
        
        security_implementations = 0
        for sec_file in security_files:
            if sec_file.exists():
                security_implementations += 1
        
        if security_implementations == 0:
            results['issues'].append("No security utility modules found")
            results['passed'] = False
        
        return results
    
    def _check_dependency_security(self) -> Dict[str, Any]:
        """Check for known vulnerabilities in dependencies."""
        self.logger.info("Checking dependency security")
        
        results = {
            'test_name': 'Dependency Security',
            'passed': True,
            'issues': [],
            'tests_run': 0,
            'tests_passed': 0
        }
        
        # Check for requirements files
        req_files = [
            self.project_root / 'requirements.txt',
            self.project_root / 'setup.py',
            self.project_root / 'pyproject.toml'
        ]
        
        found_requirements = False
        for req_file in req_files:
            if req_file.exists():
                found_requirements = True
                results['tests_run'] += 1
                
                # Basic check for pinned versions
                content = req_file.read_text()
                
                # Look for unpinned dependencies (security risk)
                lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
                unpinned_deps = []
                
                for line in lines:
                    if '==' not in line and '>=' not in line and '<=' not in line and '~=' not in line:
                        if not any(char in line for char in ['[', ']', '#']):  # Skip extras and comments
                            unpinned_deps.append(line)
                
                if unpinned_deps:
                    results['issues'].append(f"Unpinned dependencies in {req_file.name}: {unpinned_deps}")
                    results['passed'] = False
                else:
                    results['tests_passed'] += 1
        
        if not found_requirements:
            results['issues'].append("No requirements file found - dependency tracking recommended")
            results['passed'] = False
        
        # Check for known vulnerable packages (simplified check)
        try:
            import pkg_resources
            
            # List of packages with known common vulnerabilities
            vulnerable_packages = {
                'pillow': '8.0.0',  # Example: versions before 8.1.0 had vulnerabilities
                'requests': '2.20.0',  # Example: versions before 2.20.0 had vulnerabilities
            }
            
            for package_name, min_safe_version in vulnerable_packages.items():
                results['tests_run'] += 1
                try:
                    installed_version = pkg_resources.get_distribution(package_name).version
                    
                    # Simple version comparison (not robust, but for demo)
                    if installed_version < min_safe_version:
                        results['issues'].append(
                            f"Potentially vulnerable {package_name} version: {installed_version} < {min_safe_version}"
                        )
                        results['passed'] = False
                    else:
                        results['tests_passed'] += 1
                        
                except pkg_resources.DistributionNotFound:
                    # Package not installed, skip
                    results['tests_passed'] += 1
        
        except ImportError:
            results['issues'].append("Cannot check installed packages - pkg_resources not available")
        
        return results
    
    def _test_network_security(self) -> Dict[str, Any]:
        """Test network-related security configurations."""
        self.logger.info("Testing network security")
        
        results = {
            'test_name': 'Network Security',
            'passed': True,
            'issues': [],
            'tests_run': 0,
            'tests_passed': 0
        }
        
        # Check for hardcoded credentials or URLs
        config_files = list(self.project_root.rglob("*.yaml")) + list(self.project_root.rglob("*.yml")) + list(self.project_root.rglob("*.json"))
        
        dangerous_patterns = [
            (r'password\s*[:=]\s*["\'](?!changeme|password|secret)[^"\']+["\']', 'Hardcoded password'),
            (r'api_key\s*[:=]\s*["\'][^"\']+["\']', 'Hardcoded API key'),
            (r'secret\s*[:=]\s*["\'][^"\']+["\']', 'Hardcoded secret'),
            (r'token\s*[:=]\s*["\'][^"\']+["\']', 'Hardcoded token'),
            (r'http://[^/\s]+', 'Unencrypted HTTP URL'),
        ]
        
        for config_file in config_files:
            if config_file.name.startswith('.'):
                continue
                
            try:
                content = config_file.read_text(encoding='utf-8')
                results['tests_run'] += 1
                
                file_has_issues = False
                for pattern, description in dangerous_patterns:
                    import re
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        results['issues'].append(
                            f"{description} in {config_file.relative_to(self.project_root)}"
                        )
                        file_has_issues = True
                        results['passed'] = False
                
                if not file_has_issues:
                    results['tests_passed'] += 1
                    
            except Exception as e:
                results['issues'].append(f"Error scanning {config_file}: {e}")
        
        # Check for secure defaults
        results['tests_run'] += 1
        secure_config = self.validator.create_secure_environment_config()
        
        if secure_config.get('security', {}).get('input_validation', False):
            results['tests_passed'] += 1
        else:
            results['issues'].append("Secure environment configuration not properly implemented")
            results['passed'] = False
        
        return results
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics for the scan."""
        total_issues = 0
        critical_issues = 0
        warnings = 0
        passed_checks = 0
        failed_checks = 0
        
        for test_name, test_results in self.scan_results['tests'].items():
            if test_results['passed']:
                passed_checks += 1
            else:
                failed_checks += 1
            
            issue_count = len(test_results['issues'])
            total_issues += issue_count
            
            # Classify issues by severity (simplified)
            for issue in test_results['issues']:
                if any(keyword in issue.lower() for keyword in ['eval', 'exec', 'system', 'malicious', 'vulnerable']):
                    critical_issues += 1
                else:
                    warnings += 1
        
        self.scan_results.update({
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'warnings': warnings,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks
        })
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate security scan report."""
        report_lines = [
            "EchoLoc-NN Security Scan Report",
            "=" * 40,
            f"Scan Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.scan_results['timestamp']))}",
            f"Project Root: {self.scan_results['project_root']}",
            "",
            "SUMMARY",
            "-" * 20,
            f"Total Issues: {self.scan_results['total_issues']}",
            f"Critical Issues: {self.scan_results['critical_issues']}",
            f"Warnings: {self.scan_results['warnings']}",
            f"Passed Checks: {self.scan_results['passed_checks']}",
            f"Failed Checks: {self.scan_results['failed_checks']}",
            "",
            "DETAILED RESULTS",
            "-" * 20
        ]
        
        for test_name, test_results in self.scan_results['tests'].items():
            status = "✓ PASSED" if test_results['passed'] else "✗ FAILED"
            report_lines.extend([
                f"\n{test_results['test_name']}: {status}",
                f"Tests Run: {test_results.get('tests_run', 0)}",
                f"Tests Passed: {test_results.get('tests_passed', 0)}"
            ])
            
            if test_results['issues']:
                report_lines.append("Issues:")
                for issue in test_results['issues']:
                    report_lines.append(f"  - {issue}")
        
        # Add recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 20
        ])
        
        if self.scan_results['critical_issues'] > 0:
            report_lines.append("⚠️  CRITICAL: Address critical security issues immediately")
        
        if self.scan_results['warnings'] > 0:
            report_lines.append("⚠️  Review and address warning-level issues")
        
        if self.scan_results['total_issues'] == 0:
            report_lines.append("✓ No security issues detected - good security posture")
        
        report_lines.extend([
            "",
            "Next Steps:",
            "1. Address critical issues first",
            "2. Review and fix warnings",
            "3. Implement additional security measures as needed",
            "4. Run regular security scans",
            "5. Keep dependencies updated",
            ""
        ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report)
            self.logger.info(f"Security report written to: {output_file}")
        
        return report


def main():
    """Main entry point for security scanner."""
    parser = argparse.ArgumentParser(description="EchoLoc-NN Security Scanner")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--output",
        help="Output file for security report"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run scanner
    scanner = SecurityScanner(args.project_root)
    
    try:
        print("Running EchoLoc-NN Security Scan...")
        print("=" * 40)
        
        # Run the scan
        results = scanner.run_full_scan()
        
        if args.json:
            # Output as JSON
            json_output = json.dumps(results, indent=2, default=str)
            if args.output:
                Path(args.output).write_text(json_output)
            else:
                print(json_output)
        else:
            # Generate and output report
            report = scanner.generate_report(args.output)
            if not args.output:
                print(report)
        
        # Exit with appropriate code
        if results['critical_issues'] > 0:
            print(f"\n❌ Critical security issues found: {results['critical_issues']}")
            sys.exit(1)
        elif results['total_issues'] > 0:
            print(f"\n⚠️  Security warnings found: {results['warnings']}")
            sys.exit(0)  # Warnings don't fail the build
        else:
            print("\n✅ No security issues detected")
            sys.exit(0)
    
    except Exception as e:
        print(f"Error running security scan: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()