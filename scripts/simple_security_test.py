#!/usr/bin/env python3
"""
Simple security validation test without heavy dependencies.
"""

import os
import sys
import re
from pathlib import Path

def test_dangerous_patterns():
    """Test for dangerous code patterns in the codebase."""
    print("🔍 Scanning for dangerous code patterns...")
    
    # Patterns to look for (excluding legitimate uses)
    patterns = [
        (r'(?<!model\.)eval\s*\(', 'eval() usage'),  # Exclude model.eval()
        (r'exec\s*\(', 'exec() usage'),
        (r'__import__\s*\(', 'dynamic import'),
        (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', 'shell injection risk'),
        (r'os\.system\s*\(', 'os.system() usage'),
        (r'pickle\.loads?\s*\([^)]*\)', 'pickle deserialization'),
    ]
    
    issues = []
    python_files = list(Path('.').rglob('*.py'))
    
    for py_file in python_files:
        if 'test' in py_file.name.lower() or py_file.name.startswith('.'):
            continue
            
        try:
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            
            for pattern, description in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Get line number
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = content.split('\n')[line_num - 1].strip()
                    
                    # Skip if in comment, docstring, or test pattern
                    if (line_content.startswith('#') or '"""' in line_content or "'''" in line_content or
                        'security_scan.py' in str(py_file) or 'simple_security_test.py' in str(py_file)):
                        continue
                    
                    issues.append(f"{description} in {py_file}:{line_num}")
        
        except Exception as e:
            print(f"Warning: Could not scan {py_file}: {e}")
    
    if issues:
        print("❌ Security issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ No dangerous patterns detected")
        return True

def test_file_permissions():
    """Test file permissions for security."""
    print("\n🔍 Checking file permissions...")
    
    issues = []
    
    # Check script files are not world-writable
    script_files = list(Path('.').rglob('*.py')) + list(Path('.').rglob('*.sh'))
    
    for script_file in script_files:
        try:
            stat = script_file.stat()
            mode = stat.st_mode
            
            # Check if world-writable (others can write)
            if mode & 0o002:
                issues.append(f"World-writable file: {script_file}")
            
            # Check if executable files are properly secured
            if mode & 0o111 and mode & 0o022:  # Executable but group/world writable
                issues.append(f"Executable file with loose permissions: {script_file}")
        
        except Exception as e:
            print(f"Warning: Could not check permissions for {script_file}: {e}")
    
    if issues:
        print("❌ Permission issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ File permissions look secure")
        return True

def test_hardcoded_secrets():
    """Test for hardcoded secrets."""
    print("\n🔍 Scanning for hardcoded secrets...")
    
    patterns = [
        (r'password\s*[:=]\s*["\'][^"\']{8,}["\']', 'hardcoded password'),
        (r'api_key\s*[:=]\s*["\'][^"\']{20,}["\']', 'hardcoded API key'),
        (r'secret\s*[:=]\s*["\'][^"\']{16,}["\']', 'hardcoded secret'),
        (r'token\s*[:=]\s*["\'][^"\']{20,}["\']', 'hardcoded token'),
        (r'private_key\s*[:=]\s*["\'][^"\']{32,}["\']', 'hardcoded private key'),
    ]
    
    issues = []
    config_files = (
        list(Path('.').rglob('*.py')) + 
        list(Path('.').rglob('*.yaml')) + 
        list(Path('.').rglob('*.yml')) + 
        list(Path('.').rglob('*.json')) +
        list(Path('.').rglob('*.env'))
    )
    
    for config_file in config_files:
        if config_file.name.startswith('.') or 'test' in config_file.name.lower():
            continue
            
        try:
            content = config_file.read_text(encoding='utf-8', errors='ignore')
            
            for pattern, description in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Skip obvious dummy/example values
                    matched_text = match.group()
                    if any(dummy in matched_text.lower() for dummy in [
                        'changeme', 'password', 'secret', 'example', 'test', 'dummy', 'placeholder'
                    ]):
                        continue
                    
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append(f"{description} in {config_file}:{line_num}")
        
        except Exception as e:
            print(f"Warning: Could not scan {config_file}: {e}")
    
    if issues:
        print("❌ Potential hardcoded secrets found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ No hardcoded secrets detected")
        return True

def test_security_implementation():
    """Test that security utilities are implemented."""
    print("\n🔍 Checking security implementation...")
    
    security_files = [
        'echoloc_nn/utils/security.py',
        'echoloc_nn/utils/validation.py',
        'echoloc_nn/utils/exceptions.py',
    ]
    
    missing_files = []
    for security_file in security_files:
        if not Path(security_file).exists():
            missing_files.append(security_file)
    
    if missing_files:
        print("❌ Missing security files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    # Check security.py has key components
    security_content = Path('echoloc_nn/utils/security.py').read_text()
    required_classes = ['InputSanitizer', 'SecurityValidator', 'SecureConfigLoader']
    missing_classes = []
    
    for cls in required_classes:
        if f'class {cls}' not in security_content:
            missing_classes.append(cls)
    
    if missing_classes:
        print("❌ Missing security classes:")
        for cls in missing_classes:
            print(f"  - {cls}")
        return False
    
    print("✅ Security implementation found")
    return True

def test_input_validation_patterns():
    """Test that input validation patterns are comprehensive."""
    print("\n🔍 Checking input validation patterns...")
    
    security_file = Path('echoloc_nn/utils/security.py')
    if not security_file.exists():
        print("❌ Security file not found")
        return False
    
    content = security_file.read_text()
    
    # Check for dangerous patterns list
    dangerous_patterns = [
        r'\\.\\./',  # Directory traversal
        r'eval\\s*\\(',  # eval calls
        r'exec\\s*\\(',  # exec calls
        r'import\\s+',  # import statements
        r'subprocess',  # subprocess calls
        r'<script',  # XSS attempts
    ]
    
    found_patterns = 0
    for pattern in dangerous_patterns:
        if pattern in content:
            found_patterns += 1
    
    if found_patterns < len(dangerous_patterns) * 0.8:  # At least 80% should be there
        print(f"❌ Input validation patterns incomplete: {found_patterns}/{len(dangerous_patterns)} found")
        return False
    
    print(f"✅ Input validation patterns comprehensive: {found_patterns}/{len(dangerous_patterns)} found")
    return True

def main():
    """Run all security tests."""
    print("EchoLoc-NN Security Validation")
    print("=" * 40)
    
    tests = [
        test_security_implementation,
        test_input_validation_patterns,
        test_dangerous_patterns,
        test_file_permissions,
        test_hardcoded_secrets,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
    
    print(f"\n{'='*40}")
    print(f"Security Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All security tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} security tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())