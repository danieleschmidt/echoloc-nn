#!/usr/bin/env python3
"""
Terragon Value Discovery Engine for EchoLoc-NN
Autonomous SDLC enhancement with continuous value discovery
"""

import os
import re
import yaml
import json
import subprocess
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TerragonValueDiscovery:
    """Autonomous value discovery and prioritization engine"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.config = self._load_config()
        self.discovered_items = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            return {}
    
    def run_discovery_cycle(self) -> Dict[str, Any]:
        """Execute complete value discovery cycle"""
        logger.info("Starting Terragon value discovery cycle")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "signals_harvested": 0,
            "items_discovered": 0,
            "value_items": []
        }
        
        try:
            # Phase 1: Signal Harvesting
            signals = self._harvest_signals()
            results["signals_harvested"] = len(signals)
            
            # Phase 2: Value Analysis
            value_items = self._analyze_value(signals)
            results["items_discovered"] = len(value_items)
            results["value_items"] = value_items
            
            # Phase 3: Prioritization
            prioritized_items = self._prioritize_items(value_items)
            
            # Phase 4: Autonomous Execution (placeholder)
            execution_results = self._plan_execution(prioritized_items)
            results["execution_plan"] = execution_results
            
            # Update discovery history
            self._update_history(results)
            
            logger.info(f"Discovery cycle complete: {results['items_discovered']} items discovered")
            return results
            
        except Exception as e:
            logger.error(f"Discovery cycle failed: {e}")
            return results
    
    def _harvest_signals(self) -> List[Dict[str, Any]]:
        """Harvest signals from multiple sources"""
        signals = []
        
        # Code analysis signals
        if self.config.get("signals", {}).get("code_analysis", {}).get("enabled", True):
            signals.extend(self._harvest_code_signals())
        
        # Git history signals  
        signals.extend(self._harvest_git_signals())
        
        # Static analysis signals
        if self.config.get("signals", {}).get("static_analysis", {}).get("enabled", True):
            signals.extend(self._harvest_static_analysis_signals())
        
        # Dependency signals
        if self.config.get("signals", {}).get("dependency_analysis", {}).get("enabled", True):
            signals.extend(self._harvest_dependency_signals())
        
        return signals
    
    def _harvest_code_signals(self) -> List[Dict[str, Any]]:
        """Extract signals from code comments and patterns"""
        signals = []
        patterns = self.config.get("signals", {}).get("code_analysis", {}).get("patterns", [])
        
        for pattern in patterns:
            try:
                result = subprocess.run([
                    "grep", "-r", "-n", pattern, str(self.repo_path),
                    "--include=*.py", "--exclude-dir=.git"
                ], capture_output=True, text=True)
                
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            signals.append({
                                "type": "code_comment",
                                "pattern": pattern,
                                "file": parts[0],
                                "line": parts[1],
                                "content": parts[2].strip(),
                                "priority": self._calculate_comment_priority(pattern, parts[2])
                            })
            except Exception as e:
                logger.warning(f"Failed to harvest code signals for pattern {pattern}: {e}")
        
        return signals
    
    def _harvest_git_signals(self) -> List[Dict[str, Any]]:
        """Extract signals from git history"""
        signals = []
        git_patterns = self.config.get("signals", {}).get("code_analysis", {}).get("git_patterns", [])
        
        try:
            # Get recent commit messages
            result = subprocess.run([
                "git", "log", "--oneline", "-50"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    for pattern in git_patterns:
                        if pattern.lower() in line.lower():
                            signals.append({
                                "type": "git_commit",
                                "pattern": pattern,
                                "commit": line,
                                "priority": 0.6
                            })
        except Exception as e:
            logger.warning(f"Failed to harvest git signals: {e}")
        
        return signals
    
    def _harvest_static_analysis_signals(self) -> List[Dict[str, Any]]:
        """Run static analysis tools and extract signals"""
        signals = []
        
        # Placeholder for static analysis integration
        # In real implementation, would run flake8, mypy, bandit, etc.
        logger.info("Static analysis signal harvesting (placeholder)")
        
        return signals
    
    def _harvest_dependency_signals(self) -> List[Dict[str, Any]]:
        """Analyze dependencies for vulnerabilities and updates"""
        signals = []
        
        # Check for pyproject.toml
        pyproject_path = self.repo_path / "pyproject.toml"
        if pyproject_path.exists():
            signals.append({
                "type": "dependency_management",
                "file": str(pyproject_path),
                "status": "configured",
                "priority": 0.3
            })
        
        return signals
    
    def _analyze_value(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze signals to identify value opportunities"""
        value_items = []
        
        # Group signals by type and analyze patterns
        signal_groups = {}
        for signal in signals:
            signal_type = signal["type"]
            if signal_type not in signal_groups:
                signal_groups[signal_type] = []
            signal_groups[signal_type].append(signal)
        
        # Generate value items based on signal analysis
        for signal_type, group_signals in signal_groups.items():
            if signal_type == "code_comment":
                value_items.extend(self._analyze_code_comments(group_signals))
            elif signal_type == "git_commit":
                value_items.extend(self._analyze_git_commits(group_signals))
            elif signal_type == "dependency_management":
                value_items.extend(self._analyze_dependencies(group_signals))
        
        return value_items
    
    def _analyze_code_comments(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze code comment signals for value opportunities"""
        value_items = []
        
        for signal in signals:
            pattern = signal["pattern"]
            if pattern in ["TODO:", "FIXME:"]:
                value_items.append({
                    "title": f"Address {pattern} in {Path(signal['file']).name}",
                    "description": signal["content"],
                    "type": "technical_debt",
                    "file": signal["file"],
                    "line": signal["line"],
                    "priority": signal["priority"],
                    "effort_estimate": "small",
                    "value_category": "code_quality"
                })
        
        return value_items
    
    def _analyze_git_commits(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze git commit signals for patterns"""
        value_items = []
        
        # Look for patterns indicating technical debt
        quick_fix_count = len([s for s in signals if "quick fix" in s["pattern"]])
        if quick_fix_count > 2:
            value_items.append({
                "title": "Review and refactor frequent quick fixes",
                "description": f"Found {quick_fix_count} recent commits with 'quick fix' pattern",
                "type": "refactoring",
                "priority": 0.7,
                "effort_estimate": "medium",
                "value_category": "technical_debt"
            })
        
        return value_items
    
    def _analyze_dependencies(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze dependency management signals"""
        value_items = []
        
        for signal in signals:
            if signal["status"] == "configured":
                value_items.append({
                    "title": "Setup automated dependency scanning",
                    "description": "Configure automated dependency vulnerability scanning",
                    "type": "security",
                    "priority": 0.8,
                    "effort_estimate": "small",
                    "value_category": "security"
                })
        
        return value_items
    
    def _prioritize_items(self, value_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize value items using configured factors"""
        prioritization_config = self.config.get("prioritization", {})
        factors = prioritization_config.get("factors", {})
        
        for item in value_items:
            # Calculate composite priority score
            priority_score = self._calculate_priority_score(item, factors)
            item["final_priority"] = priority_score
            item["priority_category"] = self._get_priority_category(priority_score)
        
        # Sort by priority
        return sorted(value_items, key=lambda x: x["final_priority"], reverse=True)
    
    def _calculate_priority_score(self, item: Dict[str, Any], factors: Dict[str, Any]) -> float:
        """Calculate weighted priority score for an item"""
        base_priority = item.get("priority", 0.5)
        
        # Apply domain-specific weighting
        category_weights = {
            "security": 1.2,
            "performance": 1.1,
            "technical_debt": 1.0,
            "feature": 0.9,
            "documentation": 0.8
        }
        
        category = item.get("value_category", "feature")
        weight_multiplier = category_weights.get(category, 1.0)
        
        return min(base_priority * weight_multiplier, 1.0)
    
    def _get_priority_category(self, score: float) -> str:
        """Convert priority score to category"""
        thresholds = self.config.get("execution", {}).get("execution_thresholds", {})
        
        if score >= thresholds.get("critical_priority", 0.8):
            return "critical"
        elif score >= thresholds.get("high_priority", 0.6):
            return "high"
        elif score >= thresholds.get("medium_priority", 0.4):
            return "medium"
        else:
            return "low"
    
    def _plan_execution(self, prioritized_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan autonomous execution of high-priority items"""
        execution_plan = {
            "immediate_actions": [],
            "scheduled_actions": [],
            "manual_review_required": []
        }
        
        for item in prioritized_items:
            if item["priority_category"] == "critical":
                execution_plan["immediate_actions"].append(item)
            elif item["priority_category"] == "high":
                execution_plan["scheduled_actions"].append(item)
            else:
                execution_plan["manual_review_required"].append(item)
        
        return execution_plan
    
    def _calculate_comment_priority(self, pattern: str, content: str) -> float:
        """Calculate priority based on comment pattern and content"""
        priority_map = {
            "FIXME:": 0.8,
            "TODO:": 0.6,
            "HACK:": 0.9,
            "DEPRECATED:": 0.7,
            "BUG:": 0.9,
            "OPTIMIZE:": 0.5
        }
        
        base_priority = priority_map.get(pattern, 0.5)
        
        # Boost priority for security-related content
        if any(keyword in content.lower() for keyword in ["security", "vulnerability", "unsafe"]):
            base_priority = min(base_priority + 0.2, 1.0)
        
        return base_priority
    
    def _update_history(self, results: Dict[str, Any]) -> None:
        """Update discovery history"""
        try:
            history_path = self.repo_path / ".terragon" / "discovery_history.json"
            
            history = []
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
            
            history.append(results)
            
            # Keep only last 100 entries
            history = history[-100:]
            
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update discovery history: {e}")

def main():
    """Main entry point for value discovery"""
    discovery = TerragonValueDiscovery()
    results = discovery.run_discovery_cycle()
    
    print(f"\n🔍 Terragon Value Discovery Results")
    print(f"📊 Signals harvested: {results['signals_harvested']}")
    print(f"💎 Value items discovered: {results['items_discovered']}")
    
    if results["value_items"]:
        print(f"\n🎯 Top Priority Items:")
        for item in results["value_items"][:5]:
            print(f"  • {item['title']} ({item.get('priority_category', 'medium')} priority)")
    
    print(f"\n✅ Discovery cycle completed at {results['timestamp']}")

if __name__ == "__main__":
    main()