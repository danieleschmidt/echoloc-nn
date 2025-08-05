#!/usr/bin/env python3
"""
Security Scanner for EchoLoc-NN Quantum Planning System

Performs comprehensive security analysis including:
- Code injection vulnerabilities
- Insecure data handling
- Authentication and authorization issues
- Information disclosure risks
- Insecure dependencies
"""

import sys
import os
import re
import ast
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

@dataclass
class SecurityFinding:
    """Security finding with severity and details."""
    severity: Severity
    category: str
    title: str
    description: str
    file_path: str
    line_number: int = 0
    code_snippet: str = ""
    recommendation: str = ""

class SecurityScanner:
    """Comprehensive security scanner for Python codebases."""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.findings: List[SecurityFinding] = []
        
        # Security patterns to detect
        self.dangerous_patterns = {
            # Code injection
            r'eval\s*\(': ('Code Injection', 'Use of eval() function', 'CRITICAL'),
            r'exec\s*\(': ('Code Injection', 'Use of exec() function', 'CRITICAL'),
            r'subprocess\.call\s*\(.*shell\s*=\s*True': ('Command Injection', 'subprocess with shell=True', 'HIGH'),
            r'os\.system\s*\(': ('Command Injection', 'Use of os.system()', 'HIGH'),
            
            # Pickle vulnerabilities
            r'pickle\.loads\s*\(': ('Deserialization', 'Insecure pickle.loads()', 'HIGH'),
            r'cPickle\.loads\s*\(': ('Deserialization', 'Insecure cPickle.loads()', 'HIGH'),
            
            # Hardcoded secrets
            r'password\s*=\s*["\'][^"\']+["\']': ('Hardcoded Secrets', 'Hardcoded password', 'HIGH'),
            r'secret\s*=\s*["\'][^"\']+["\']': ('Hardcoded Secrets', 'Hardcoded secret', 'HIGH'),
            r'api_key\s*=\s*["\'][^"\']+["\']': ('Hardcoded Secrets', 'Hardcoded API key', 'HIGH'),
            r'token\s*=\s*["\'][^"\']+["\']': ('Hardcoded Secrets', 'Hardcoded token', 'MEDIUM'),
            
            # Insecure random
            r'random\.random\s*\(': ('Weak Crypto', 'Use of weak random generator', 'MEDIUM'),
            r'random\.randint\s*\(': ('Weak Crypto', 'Use of weak random generator', 'MEDIUM'),
            
            # SQL injection (if any SQL code)
            r'\.execute\s*\(\s*["\'][^"\']*%[sd][^"\']*["\']': ('SQL Injection', 'String formatting in SQL', 'HIGH'),
            r'\.format\s*\(\s*\).*sql': ('SQL Injection', 'Potential SQL injection', 'MEDIUM'),
            
            # Path traversal
            r'open\s*\(\s*.*\+.*\)': ('Path Traversal', 'Potential path traversal in file operations', 'MEDIUM'),
            
            # Insecure temporary files
            r'tempfile\.mktemp\s*\(': ('Insecure Files', 'Use of insecure mktemp()', 'MEDIUM'),
            
            # Debug mode in production
            r'debug\s*=\s*True': ('Information Disclosure', 'Debug mode enabled', 'LOW'),
            
            # Insecure protocols
            r'http://': ('Insecure Transport', 'Use of HTTP instead of HTTPS', 'LOW'),
        }
        
        # Safe patterns that should be encouraged
        self.safe_patterns = {
            r'secrets\.token_': 'Secure random token generation',
            r'secrets\.choice': 'Secure random choice',
            r'hashlib\.sha256': 'Strong hashing algorithm',
            r'bcrypt\.': 'Secure password hashing',
            r'cryptography\.': 'Using cryptography library',
            r'ssl\.create_default_context': 'Secure SSL context',
        }
    
    def scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a single Python file for security issues."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            # Pattern-based scanning
            for pattern, (category, title, severity) in self.dangerous_patterns.items():
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                    
                    finding = SecurityFinding(
                        severity=Severity(severity),
                        category=category,
                        title=title,
                        description=f"Detected {title.lower()} pattern",
                        file_path=str(file_path.relative_to(self.repo_path)),
                        line_number=line_num,
                        code_snippet=line_content.strip(),
                        recommendation=self._get_recommendation(category, title)
                    )
                    findings.append(finding)
            
            # AST-based analysis for more complex patterns
            try:
                tree = ast.parse(content, filename=str(file_path))
                ast_findings = self._analyze_ast(tree, file_path, lines)
                findings.extend(ast_findings)
            except SyntaxError:
                # File has syntax errors, skip AST analysis
                pass
                
        except Exception as e:
            # Log error but continue scanning
            pass
            
        return findings
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[SecurityFinding]:
        """Analyze AST for complex security patterns."""
        findings = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, scanner, file_path, lines):
                self.scanner = scanner
                self.file_path = file_path
                self.lines = lines
                self.findings = []
            
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec']:
                        self.findings.append(SecurityFinding(
                            severity=Severity.CRITICAL,
                            category='Code Injection',
                            title=f'Use of {node.func.id}() function',
                            description=f'Dynamic code execution using {node.func.id}()',
                            file_path=str(file_path.relative_to(self.scanner.repo_path)),
                            line_number=node.lineno,
                            code_snippet=self.lines[node.lineno - 1].strip() if node.lineno <= len(self.lines) else "",
                            recommendation="Use safe alternatives or validate input thoroughly"
                        ))
                
                # Check for subprocess calls with shell=True
                if (isinstance(node.func, ast.Attribute) and 
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == 'subprocess'):
                    
                    for keyword in node.keywords:
                        if (keyword.arg == 'shell' and 
                            isinstance(keyword.value, ast.Constant) and 
                            keyword.value.value is True):
                            
                            self.findings.append(SecurityFinding(
                                severity=Severity.HIGH,
                                category='Command Injection',
                                title='subprocess with shell=True',
                                description='Command injection vulnerability',
                                file_path=str(file_path.relative_to(self.scanner.repo_path)),
                                line_number=node.lineno,
                                code_snippet=self.lines[node.lineno - 1].strip() if node.lineno <= len(self.lines) else "",
                                recommendation="Use shell=False and pass command as list"
                            ))
                
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                # Check for hardcoded secrets in assignments
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    value = node.value.value.lower()
                    
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_name = target.id.lower()
                            
                            if any(secret in var_name for secret in ['password', 'secret', 'key', 'token']):
                                if len(node.value.value) > 8:  # Likely a real secret, not a placeholder
                                    severity = Severity.HIGH if 'password' in var_name else Severity.MEDIUM
                                    
                                    self.findings.append(SecurityFinding(
                                        severity=severity,
                                        category='Hardcoded Secrets',
                                        title=f'Hardcoded {var_name}',
                                        description='Sensitive data hardcoded in source',
                                        file_path=str(file_path.relative_to(self.scanner.repo_path)),
                                        line_number=node.lineno,
                                        code_snippet=self.lines[node.lineno - 1].strip() if node.lineno <= len(self.lines) else "",
                                        recommendation="Use environment variables or secure configuration"
                                    ))
                
                self.generic_visit(node)
        
        visitor = SecurityVisitor(self, file_path, lines)
        visitor.visit(tree)
        findings.extend(visitor.findings)
        
        return findings
    
    def _get_recommendation(self, category: str, title: str) -> str:
        """Get security recommendation for a finding."""
        recommendations = {
            'Code Injection': "Avoid dynamic code execution. Use safe alternatives or validate input thoroughly.",
            'Command Injection': "Use subprocess with shell=False and pass commands as lists. Validate all inputs.",
            'Deserialization': "Use safe serialization formats like JSON. Validate data before deserializing.",
            'Hardcoded Secrets': "Store secrets in environment variables or secure configuration systems.",
            'Weak Crypto': "Use cryptographically secure random generators from the secrets module.",
            'SQL Injection': "Use parameterized queries or ORM. Never use string formatting with SQL.",
            'Path Traversal': "Validate and sanitize file paths. Use safe path joining methods.",
            'Insecure Files': "Use secure temporary file creation methods.",
            'Information Disclosure': "Disable debug mode in production environments.",
            'Insecure Transport': "Always use HTTPS for sensitive data transmission."
        }
        return recommendations.get(category, "Review and remediate this security issue.")
    
    def scan_directory(self, directory: Path = None) -> List[SecurityFinding]:
        """Scan all Python files in a directory."""
        if directory is None:
            directory = self.repo_path
            
        findings = []
        
        # Find all Python files
        python_files = list(directory.rglob('*.py'))
        
        for py_file in python_files:
            file_findings = self.scan_file(py_file)
            findings.extend(file_findings)
        
        return findings
    
    def check_dependencies(self) -> List[SecurityFinding]:
        """Check for known vulnerable dependencies."""
        findings = []
        
        # Check requirements files
        req_files = ['requirements.txt', 'requirements-dev.txt', 'requirements-test.txt']
        
        for req_file in req_files:
            req_path = self.repo_path / req_file
            if req_path.exists():
                findings.extend(self._analyze_requirements(req_path))
        
        return findings
    
    def _analyze_requirements(self, req_file: Path) -> List[SecurityFinding]:
        """Analyze requirements file for vulnerable packages."""
        findings = []
        
        # Known vulnerable package patterns (simplified)
        vulnerable_packages = {
            'pickle': 'Use safer serialization formats',
            'pyyaml': 'Ensure version >= 5.1 to avoid yaml.load() issues',
            'requests': 'Ensure version >= 2.20.0 for security fixes',
            'jinja2': 'Ensure version >= 2.10.1 for XSS protection',
            'flask': 'Ensure recent version for security updates'
        }
        
        try:
            with open(req_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip().lower()
                    if line and not line.startswith('#'):
                        for pkg, recommendation in vulnerable_packages.items():
                            if pkg in line:
                                findings.append(SecurityFinding(
                                    severity=Severity.MEDIUM,
                                    category='Vulnerable Dependency',
                                    title=f'Potentially vulnerable package: {pkg}',
                                    description=f'Package {pkg} may have security issues',
                                    file_path=str(req_file.relative_to(self.repo_path)),
                                    line_number=line_num,
                                    code_snippet=line,
                                    recommendation=recommendation
                                ))
        except Exception:
            pass
        
        return findings
    
    def generate_report(self) -> str:
        """Generate comprehensive security report."""
        # Scan all files
        code_findings = self.scan_directory()
        dep_findings = self.check_dependencies()
        
        all_findings = code_findings + dep_findings
        
        # Group by severity
        by_severity = {sev: [] for sev in Severity}
        for finding in all_findings:
            by_severity[finding.severity].append(finding)
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("ECHOLOC-NN QUANTUM PLANNING - SECURITY SCAN REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Executive summary
        total_findings = len(all_findings)
        critical_count = len(by_severity[Severity.CRITICAL])
        high_count = len(by_severity[Severity.HIGH])
        medium_count = len(by_severity[Severity.MEDIUM])
        low_count = len(by_severity[Severity.LOW])
        
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append(f"Total findings: {total_findings}")
        report.append(f"Critical: {critical_count}")
        report.append(f"High: {high_count}")
        report.append(f"Medium: {medium_count}")
        report.append(f"Low: {low_count}")
        report.append("")
        
        # Security posture assessment
        if critical_count == 0 and high_count == 0:
            if medium_count <= 2:
                report.append("🟢 SECURITY POSTURE: GOOD")
                report.append("   No critical or high-severity issues found.")
            else:
                report.append("🟡 SECURITY POSTURE: ACCEPTABLE")
                report.append("   Some medium-risk issues should be addressed.")
        elif high_count <= 2 and critical_count == 0:
            report.append("🟡 SECURITY POSTURE: NEEDS ATTENTION")
            report.append("   High-severity issues require immediate attention.")
        else:
            report.append("🔴 SECURITY POSTURE: REQUIRES IMMEDIATE ACTION")
            report.append("   Critical security vulnerabilities must be fixed.")
        
        report.append("")
        
        # Detailed findings by severity
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            findings = by_severity[severity]
            if not findings:
                continue
                
            report.append(f"{severity.value} SEVERITY FINDINGS ({len(findings)})")
            report.append("-" * 40)
            
            for i, finding in enumerate(findings, 1):
                report.append(f"{i}. {finding.title}")
                report.append(f"   Category: {finding.category}")
                report.append(f"   File: {finding.file_path}:{finding.line_number}")
                if finding.code_snippet:
                    report.append(f"   Code: {finding.code_snippet}")
                report.append(f"   Description: {finding.description}")
                report.append(f"   Recommendation: {finding.recommendation}")
                report.append("")
        
        # Security best practices check
        report.append("SECURITY BEST PRACTICES ASSESSMENT")
        report.append("-" * 40)
        
        best_practices = self._assess_best_practices()
        for practice, status in best_practices.items():
            symbol = "✓" if status else "✗"
            report.append(f"{symbol} {practice}")
        
        report.append("")
        
        # Recommendations
        report.append("SECURITY RECOMMENDATIONS")
        report.append("-" * 30)
        report.append("1. Fix all CRITICAL and HIGH severity issues immediately")
        report.append("2. Implement input validation for all user inputs")
        report.append("3. Use secure coding practices for quantum algorithms")
        report.append("4. Implement proper authentication and authorization")
        report.append("5. Use secure communication protocols (HTTPS/TLS)")
        report.append("6. Regularly update dependencies to latest secure versions")
        report.append("7. Implement security logging and monitoring")
        report.append("8. Conduct regular security assessments")
        report.append("")
        
        return "\n".join(report)
    
    def _assess_best_practices(self) -> Dict[str, bool]:
        """Assess implementation of security best practices."""
        practices = {}
        
        # Check for presence of security-related files/configurations
        security_files = [
            'security.py',
            'auth.py', 
            'authentication.py',
            'authorization.py',
            'crypto.py',
            'encryption.py'
        ]
        
        has_security_module = any(
            list(self.repo_path.rglob(f'*{filename}')) 
            for filename in security_files
        )
        practices["Security module present"] = has_security_module
        
        # Check for secure configuration patterns
        config_files = list(self.repo_path.rglob('*config*.py'))
        has_secure_config = False
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                    if 'os.environ' in content or 'getenv' in content:
                        has_secure_config = True
                        break
            except Exception:
                pass
        
        practices["Environment-based configuration"] = has_secure_config
        
        # Check for secure random usage
        py_files = list(self.repo_path.rglob('*.py'))
        has_secure_random = False
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if 'secrets.' in content:
                        has_secure_random = True
                        break
            except Exception:
                pass
        
        practices["Secure random number generation"] = has_secure_random
        
        # Check for input validation
        has_validation = any(
            list(self.repo_path.rglob('*validation*.py'))
        )
        practices["Input validation implementation"] = has_validation
        
        # Check for error handling
        has_error_handling = any(
            list(self.repo_path.rglob('*error*.py')) or
            list(self.repo_path.rglob('*exception*.py'))
        )
        practices["Error handling implementation"] = has_error_handling
        
        return practices

def main():
    """Run security scan on the repository."""
    repo_path = Path(__file__).parent
    scanner = SecurityScanner(str(repo_path))
    
    print("Running security scan...")
    report = scanner.generate_report()
    
    # Write report to file
    report_file = repo_path / 'security_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nFull report saved to: {report_file}")
    
    # Return exit code based on severity
    all_findings = scanner.scan_directory() + scanner.check_dependencies()
    critical_count = sum(1 for f in all_findings if f.severity == Severity.CRITICAL)
    high_count = sum(1 for f in all_findings if f.severity == Severity.HIGH)
    
    if critical_count > 0:
        return 2  # Critical issues
    elif high_count > 0:
        return 1  # High severity issues
    else:
        return 0  # No major issues

if __name__ == "__main__":
    sys.exit(main())