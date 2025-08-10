#!/usr/bin/env python3
"""
Validate the syntax and structure of the model files.
"""

import ast
import os

def validate_file_syntax(filepath):
    """Validate Python file syntax."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        # Parse the AST
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_class_structure(filepath, expected_classes):
    """Check if expected classes are defined."""
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        missing = set(expected_classes) - set(classes)
        if missing:
            return False, f"Missing classes: {missing}"
        
        return True, classes
    except Exception as e:
        return False, f"Error checking classes: {e}"

def validate_models():
    """Validate all model files."""
    files_to_check = [
        ('echoloc_nn/models/__init__.py', []),
        ('echoloc_nn/models/base.py', ['EchoLocBaseModel']),
        ('echoloc_nn/models/hybrid_architecture.py', [
            'ConvBlock', 'MultiPathConv', 'CNNEncoder', 'EchoPositionalEncoding',
            'EchoAttention', 'TransformerDecoder', 'CNNTransformerHybrid', 'EchoLocModel'
        ])
    ]
    
    results = {}
    
    for filepath, expected_classes in files_to_check:
        print(f"\nValidating {filepath}...")
        
        if not os.path.exists(filepath):
            results[filepath] = (False, "File does not exist")
            continue
        
        # Check syntax
        syntax_ok, syntax_error = validate_file_syntax(filepath)
        if not syntax_ok:
            results[filepath] = (False, syntax_error)
            continue
        
        # Check class structure
        if expected_classes:
            classes_ok, class_info = check_class_structure(filepath, expected_classes)
            if not classes_ok:
                results[filepath] = (False, class_info)
                continue
            else:
                results[filepath] = (True, f"All classes found: {class_info}")
        else:
            results[filepath] = (True, "Syntax valid")
    
    return results

if __name__ == "__main__":
    print("Validating EchoLoc-NN Model Files")
    print("=" * 50)
    
    results = validate_models()
    
    all_valid = True
    for filepath, (valid, message) in results.items():
        status = "✓" if valid else "✗"
        print(f"{status} {filepath}: {message}")
        if not valid:
            all_valid = False
    
    print("\n" + "=" * 50)
    if all_valid:
        print("All model files are syntactically valid! ✓")
        
        # Check imports in __init__.py
        init_file = 'echoloc_nn/models/__init__.py'
        with open(init_file, 'r') as f:
            init_content = f.read()
        
        expected_imports = [
            'EchoLocBaseModel',
            'EchoLocModel', 
            'CNNTransformerHybrid',
            'CNNEncoder',
            'TransformerDecoder'
        ]
        
        print("\nChecking __init__.py imports...")
        for imp in expected_imports:
            if imp in init_content:
                print(f"✓ {imp} imported")
            else:
                print(f"✗ {imp} missing from imports")
    else:
        print("Some validation errors found! ✗")