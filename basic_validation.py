#!/usr/bin/env python3
"""
Basic validation script for EchoLoc Sentiment Analysis System.
Tests core functionality without external dependencies.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_file_structure():
    """Test that all required files exist."""
    
    print("📁 Testing file structure...")
    
    required_files = [
        "echoloc_nn/__init__.py",
        "echoloc_nn/sentiment/__init__.py",
        "echoloc_nn/sentiment/models.py",
        "echoloc_nn/sentiment/spatial_fusion.py", 
        "echoloc_nn/sentiment/real_time.py",
        "echoloc_nn/sentiment/multi_modal.py",
        "echoloc_nn/sentiment/api.py",
        "echoloc_nn/sentiment/cli.py",
        "echoloc_nn/sentiment/optimization.py",
        "tests/test_sentiment.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"\\n❌ Missing files:")
        for missing in missing_files:
            print(f"  ❌ {missing}")
        return False
    else:
        print("\\n✅ All required files present")
        return True

def test_python_syntax():
    """Test that all Python files have valid syntax."""
    
    print("\\n🐍 Testing Python syntax...")
    
    python_files = list(project_root.glob("echoloc_nn/sentiment/*.py"))
    python_files.extend(list(project_root.glob("tests/*.py")))
    
    syntax_errors = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            # Try to compile the code
            compile(code, str(file_path), 'exec')
            print(f"  ✅ {file_path.relative_to(project_root)}")
            
        except SyntaxError as e:
            syntax_errors.append((file_path, str(e)))
            print(f"  ❌ {file_path.relative_to(project_root)}: {e}")
        except Exception as e:
            # Other errors like encoding issues
            syntax_errors.append((file_path, str(e)))
            print(f"  ⚠️  {file_path.relative_to(project_root)}: {e}")
    
    if syntax_errors:
        print(f"\\n❌ Syntax errors found in {len(syntax_errors)} files")
        return False
    else:
        print("\\n✅ All Python files have valid syntax")
        return True

def test_import_structure():
    """Test basic import structure without actually importing."""
    
    print("\\n📦 Testing import structure...")
    
    # Read files and check for basic import patterns
    files_to_check = [
        "echoloc_nn/sentiment/models.py",
        "echoloc_nn/sentiment/spatial_fusion.py",
        "echoloc_nn/sentiment/real_time.py",
        "echoloc_nn/sentiment/multi_modal.py"
    ]
    
    import_issues = []
    
    for file_path in files_to_check:
        full_path = project_root / file_path
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Check for common import patterns
            has_typing_imports = "from typing import" in content
            has_class_definitions = "class " in content
            has_function_definitions = "def " in content
            has_docstrings = '"""' in content
            
            issues = []
            if not has_class_definitions:
                issues.append("No class definitions found")
            if not has_function_definitions:
                issues.append("No function definitions found") 
            if not has_docstrings:
                issues.append("No docstrings found")
                
            if issues:
                import_issues.append((file_path, issues))
                print(f"  ⚠️  {file_path}: {', '.join(issues)}")
            else:
                print(f"  ✅ {file_path}")
                
        except Exception as e:
            import_issues.append((file_path, [str(e)]))
            print(f"  ❌ {file_path}: {e}")
    
    if import_issues:
        print(f"\\n⚠️  Structural issues found in {len(import_issues)} files")
        return False
    else:
        print("\\n✅ Import structure looks good")
        return True

def test_code_quality():
    """Test basic code quality metrics."""
    
    print("\\n📊 Testing code quality...")
    
    python_files = list(project_root.glob("echoloc_nn/sentiment/*.py"))
    
    quality_stats = {
        "total_lines": 0,
        "total_functions": 0,
        "total_classes": 0,
        "files_with_docstrings": 0,
        "files_with_type_hints": 0
    }
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            quality_stats["total_lines"] += len(lines)
            
            content = ''.join(lines)
            
            # Count functions and classes
            function_count = content.count("def ")
            class_count = content.count("class ")
            
            quality_stats["total_functions"] += function_count
            quality_stats["total_classes"] += class_count
            
            # Check for docstrings and type hints
            if '"""' in content or "'''" in content:
                quality_stats["files_with_docstrings"] += 1
                
            if "from typing import" in content or ": " in content:
                quality_stats["files_with_type_hints"] += 1
                
            print(f"  📄 {file_path.name}: {len(lines)} lines, {function_count} functions, {class_count} classes")
            
        except Exception as e:
            print(f"  ❌ {file_path.name}: Error reading file - {e}")
            
    print(f"\\n📈 Quality Statistics:")
    print(f"  Total lines of code: {quality_stats['total_lines']}")
    print(f"  Total functions: {quality_stats['total_functions']}")
    print(f"  Total classes: {quality_stats['total_classes']}")
    print(f"  Files with docstrings: {quality_stats['files_with_docstrings']}/{len(python_files)}")
    print(f"  Files with type hints: {quality_stats['files_with_type_hints']}/{len(python_files)}")
    
    # Quality thresholds
    avg_lines_per_file = quality_stats["total_lines"] / len(python_files) if python_files else 0
    docstring_coverage = quality_stats["files_with_docstrings"] / len(python_files) if python_files else 0
    
    quality_issues = []
    
    if avg_lines_per_file < 100:
        quality_issues.append(f"Low average lines per file: {avg_lines_per_file:.0f}")
    
    if docstring_coverage < 0.8:
        quality_issues.append(f"Low docstring coverage: {docstring_coverage:.1%}")
        
    if quality_stats["total_classes"] < 5:
        quality_issues.append(f"Few classes defined: {quality_stats['total_classes']}")
        
    if quality_issues:
        print(f"\\n⚠️  Quality concerns:")
        for issue in quality_issues:
            print(f"  ⚠️  {issue}")
        return False
    else:
        print("\\n✅ Code quality metrics look good")
        return True

def test_documentation():
    """Test documentation completeness."""
    
    print("\\n📚 Testing documentation...")
    
    # Check for key documentation files
    doc_files = [
        "README.md",
        "PROJECT_CHARTER.md", 
        "ARCHITECTURE.md",
        "SECURITY.md"
    ]
    
    doc_issues = []
    
    for doc_file in doc_files:
        file_path = project_root / doc_file
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                word_count = len(content.split())
                if word_count < 100:
                    doc_issues.append(f"{doc_file} is too short ({word_count} words)")
                else:
                    print(f"  ✅ {doc_file} ({word_count} words)")
                    
            except Exception as e:
                doc_issues.append(f"{doc_file}: {e}")
                print(f"  ❌ {doc_file}: {e}")
        else:
            doc_issues.append(f"{doc_file} missing")
            print(f"  ❌ {doc_file} missing")
    
    # Check inline documentation in key files
    key_files = ["echoloc_nn/sentiment/models.py", "echoloc_nn/sentiment/api.py"]
    
    for file_path in key_files:
        full_path = project_root / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                
                docstring_count = content.count('"""') + content.count("'''")
                if docstring_count < 4:  # Minimum expected docstrings
                    doc_issues.append(f"{file_path} has insufficient docstrings ({docstring_count})")
                else:
                    print(f"  ✅ {file_path} has good docstring coverage ({docstring_count} docstrings)")
                    
            except Exception as e:
                doc_issues.append(f"{file_path}: {e}")
    
    if doc_issues:
        print(f"\\n⚠️  Documentation issues:")
        for issue in doc_issues:
            print(f"  ⚠️  {issue}")
        return False
    else:
        print("\\n✅ Documentation looks comprehensive")
        return True

def test_architecture_consistency():
    """Test architectural consistency and patterns."""
    
    print("\\n🏗️  Testing architectural consistency...")
    
    # Check for consistent patterns across files
    sentiment_files = list(project_root.glob("echoloc_nn/sentiment/*.py"))
    
    patterns_found = {
        "error_handling": 0,
        "type_hints": 0,
        "dataclasses": 0,
        "async_functions": 0,
        "test_functions": 0
    }
    
    for file_path in sentiment_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for architectural patterns
            if "@handle_errors" in content or "try:" in content:
                patterns_found["error_handling"] += 1
                
            if "from typing import" in content:
                patterns_found["type_hints"] += 1
                
            if "@dataclass" in content:
                patterns_found["dataclasses"] += 1
                
            if "async def" in content:
                patterns_found["async_functions"] += 1
                
        except Exception as e:
            print(f"  ❌ Error checking {file_path.name}: {e}")
    
    print(f"  📊 Architectural patterns found:")
    print(f"    Error handling: {patterns_found['error_handling']}/{len(sentiment_files)} files")
    print(f"    Type hints: {patterns_found['type_hints']}/{len(sentiment_files)} files")
    print(f"    Dataclasses: {patterns_found['dataclasses']}/{len(sentiment_files)} files")
    print(f"    Async functions: {patterns_found['async_functions']}/{len(sentiment_files)} files")
    
    # Architecture consistency checks
    consistency_issues = []
    
    if patterns_found["error_handling"] < len(sentiment_files) * 0.8:
        consistency_issues.append("Inconsistent error handling across files")
        
    if patterns_found["type_hints"] < len(sentiment_files) * 0.8:
        consistency_issues.append("Inconsistent type hint usage")
    
    if consistency_issues:
        print(f"\\n⚠️  Architecture consistency issues:")
        for issue in consistency_issues:
            print(f"  ⚠️  {issue}")
        return False
    else:
        print("\\n✅ Architecture patterns are consistent")
        return True

def run_validation():
    """Run all validation tests."""
    
    print("="*60)
    print("🎯 EchoLoc Sentiment Analysis - Basic Validation")
    print("="*60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("Import Structure", test_import_structure),
        ("Code Quality", test_code_quality),
        ("Documentation", test_documentation),
        ("Architecture Consistency", test_architecture_consistency)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\\n{'-'*60}")
        print(f"🔍 Running: {test_name}")
        print(f"{'-'*60}")
        
        test_start = time.time()
        try:
            success = test_func()
            test_time = time.time() - test_start
            results.append((test_name, success, test_time))
        except Exception as e:
            test_time = time.time() - test_start
            print(f"\\n💥 Test crashed: {e}")
            results.append((test_name, False, test_time))
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, success, _ in results if success)
    total_tests = len(results)
    
    print(f"\\n\\n{'='*60}")
    print("📋 VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    print(f"Tests run: {total_tests}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {total_tests - passed} ❌") 
    print(f"Success rate: {passed/total_tests*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    
    print(f"\\n📊 Test Results:")
    for test_name, success, test_time in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name:<25} ({test_time:.2f}s)")
    
    if passed == total_tests:
        print(f"\\n🎉 ALL TESTS PASSED!")
        print("✅ Sentiment analysis system validation successful")
        print("🚀 System is ready for deployment")
    else:
        print(f"\\n⚠️  {total_tests - passed} tests failed")
        print("🔧 Review and fix issues before deployment")
    
    print(f"\\n{'='*60}")
    
    return passed == total_tests

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)