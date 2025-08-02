import streamlit as st
import zipfile
import os
import tempfile
import shutil
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import transformers and torch for local LLM
try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        pipeline,
        BitsAndBytesConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("Please install transformers and torch: pip install transformers torch accelerate bitsandbytes")

@dataclass
class CodeIssue:
    file_path: str
    line_number: int
    severity: str  # 'high', 'medium', 'low'
    category: str
    description: str
    suggestion: str
    code_snippet: str

@dataclass
class AnalysisResult:
    total_files: int
    total_lines: int
    languages_detected: List[str]
    issues: List[CodeIssue]
    quality_score: float
    maintainability_score: float
    recommendations: List[str]

class LocalLLMManager:
    """Manages local LLM loading and inference"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    
    def check_model_cached(self, model_name: str) -> Dict[str, any]:
        """Check if model is already cached locally"""
        cache_info = {
            'is_cached': False,
            'cache_path': None,
            'cache_size': 0,
            'files_found': [],
            'required_files': ['config.json', 'tokenizer.json', 'tokenizer_config.json'],
            'model_files_found': [],
            'model_name': '',
        }
        
        try:
            # Check multiple possible cache locations
            possible_cache_dirs = [
                os.path.expanduser("~/.cache/huggingface/transformers"),
                os.path.expanduser("~/.cache/huggingface/hub"),
                self.cache_dir
            ]
            
            model_cache_path = None
            
            # Look for the model in cache directories
            for cache_base in possible_cache_dirs:
                if not os.path.exists(cache_base):
                    continue
                    
                # Search for model directory (could have hash suffix)
                for item in os.listdir(cache_base):
                    item_path = os.path.join(cache_base, item)
                    if os.path.isdir(item_path):
                        # Check if this directory contains our model
                        if (model_name.replace('/', '--') in item or 
                            any(model_part in item for model_part in model_name.split('/'))):
                            
                            # Verify it has required files
                            has_config = False
                            has_model_file = False
                            
                            for root, dirs, files in os.walk(item_path):
                                for file in files: 
                                    if file == 'config.json':
                                        has_config = True
                                    if any(ext in file for ext in ['.bin', '.safetensors', '.pth']):
                                        has_model_file = True
                            
                            if has_config and has_model_file:
                                model_cache_path = item_path
                                break
                
                if model_cache_path:
                    break
            
            if model_cache_path:
                cache_info['is_cached'] = True
                cache_info['cache_path'] = model_cache_path
                cache_info['model_name'] = '/'.join(model_cache_path.replace("--", "/").split('/')[4:])
                
                # Calculate cache size and list files
                total_size = 0
                files_found = []
                model_files = []
                
                for root, dirs, files in os.walk(model_cache_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            size = os.path.getsize(file_path)
                            total_size += size
                            file_info = {
                                'name': file,
                                'size': self._format_size(size),
                                'path': file_path,
                                'relative_path': os.path.relpath(file_path, model_cache_path)
                            }
                            files_found.append(file_info)
                            
                            # Track model files specifically
                            if any(ext in file for ext in ['.bin', '.safetensors', '.pth']):
                                model_files.append(file_info)
                
                cache_info['cache_size'] = total_size
                cache_info['files_found'] = files_found
                cache_info['model_files_found'] = model_files
                
                # Check for required files
                file_names = [f['name'] for f in files_found]
                missing_files = [f for f in cache_info['required_files'] if f not in file_names]
                cache_info['missing_files'] = missing_files
                cache_info['has_all_required'] = len(missing_files) == 0
                
        except Exception as e:
            cache_info['error'] = str(e)
        
        return cache_info
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    def estimate_download_size(self, model_name: str) -> str:
        """Estimate download size for different models"""
        size_estimates = {
            "deepseek-ai/deepseek-coder-1.3b-instruct": "1.2 GB",
            "codellama/CodeLlama-7b-Instruct-hf": "6.7 GB", 
            "deepseek-ai/deepseek-coder-6.7b-instruct": "6.2 GB",
            "microsoft/CodeBERT-base": "500 MB",
            "Salesforce/codet5-base": "850 MB"
        }
        return size_estimates.get(model_name, "Unknown size")
    
    def clear_model_cache(self, model_name: str) -> bool:
        """Clear cached model files"""
        try:
            cache_info = self.check_model_cached(model_name)
            if cache_info['is_cached'] and cache_info['cache_path']:
                import shutil
                shutil.rmtree(cache_info['cache_path'], ignore_errors=True)
                return True
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")
        return False
        
    @st.cache_resource
    def load_model(_self, model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct"):
        """Load the local LLM model"""
        try:
            # Check if model is cached
            cache_info = _self.check_model_cached(model_name)
            
            if cache_info['is_cached'] and cache_info.get('has_all_required', False):
                st.success(f"✅ Found cached model at {cache_info['cache_path']}")
                st.info(f"📁 Cache size: {_self._format_size(cache_info['cache_size'])}")
                use_local_files = True
                model_path = cache_info['cache_path']
            else:
                if cache_info['is_cached']:
                    missing = cache_info.get('missing_files', [])
                    st.warning(f"⚠️ Incomplete cache found. Missing: {', '.join(missing)}")
                else:
                    estimated_size = _self.estimate_download_size(model_name)
                    st.warning(f"📥 Model not cached. Will download ~{estimated_size}")
                st.info("💡 Download may take 5-15 minutes depending on your internet speed")
                use_local_files = False
                model_path = model_name
            
            # Configure for your VRAM (15.8GB)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load tokenizer
            st.info("🔤 Loading tokenizer...")
            try:
                if use_local_files:
                    _self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                else:
                    _self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=_self.cache_dir,
                        trust_remote_code=True
                    )
            except Exception as e:
                st.warning(f"Failed to load tokenizer from cache: {e}")
                st.info("Trying to download tokenizer...")
                _self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=_self.cache_dir,
                    trust_remote_code=True
                )
            
            # Load model
            st.info("🧠 Loading model...")
            try:
                if use_local_files:
                    _self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                else:
                    _self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        cache_dir=_self.cache_dir
                    )
            except Exception as e:
                st.warning(f"Failed to load model from cache: {e}")
                st.info("Trying to download model...")
                _self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    cache_dir=_self.cache_dir
                )
            
            # Create pipeline
            st.info("🔧 Creating pipeline...")
            _self.pipeline = pipeline(
                "text-generation",
                model=_self.model,
                tokenizer=_self.tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=_self.tokenizer.eos_token_id
            )
            
            # Display model info
            model_info = _self._get_model_info()
            st.success("✅ Model loaded successfully!")
            
            with st.expander("📊 Model Information"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Device:** {_self.device}")
                    st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
                    st.write(f"**Source:** {'Local Cache' if use_local_files else 'Downloaded'}")
                with col2:
                    st.write(f"**Parameters:** {model_info.get('num_parameters', 'Unknown')}")
                    st.write(f"**Memory Usage:** {model_info.get('memory_usage', 'Unknown')}")
                    st.write(f"**Cache Path:** {cache_info['cache_path'] if cache_info['is_cached'] else 'Default'}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            
            # Provide helpful troubleshooting
            if "Connection error" in str(e) or "ConnectTimeout" in str(e):
                st.error("🌐 **Network Connection Issue**")
                st.info("💡 **Troubleshooting Steps:**")
                st.write("1. Check your internet connection")
                st.write("2. Try using a VPN if Hugging Face is blocked")
                st.write("3. Use offline mode if model is already downloaded")
                
                # Check if we have a cached model
                cache_info = _self.check_model_cached(model_name)
                if cache_info['is_cached']:
                    st.info("📂 **Local cache found!** Try enabling offline mode.")
                    with st.expander("🔧 Manual Offline Setup"):
                        st.code(f"""
# Try setting environment variable:
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Or use local path directly:
model_path = "{cache_info['cache_path']}"
                        """)
            
            elif "local_files_only" in str(e):
                st.error("📂 **Cache Issue** - Model files incomplete or corrupted")
                st.info("💡 Try clearing cache and re-downloading")
            
            return False
    
    def _get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model"""
        info = {}
        try:
            if self.model:
                info['model_type'] = getattr(self.model.config, 'model_type', 'Unknown')
                
                # Estimate parameters
                total_params = sum(p.numel() for p in self.model.parameters())
                if total_params > 1e9:
                    info['num_parameters'] = f"{total_params/1e9:.1f}B"
                elif total_params > 1e6:
                    info['num_parameters'] = f"{total_params/1e6:.1f}M"
                else:
                    info['num_parameters'] = f"{total_params/1e3:.1f}K"
                
                # Estimate memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    info['memory_usage'] = f"{memory_allocated:.1f} GB"
                else:
                    info['memory_usage'] = "CPU"
                    
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def analyze_code_with_llm(self, code_content: str, file_path: str, language: str) -> List[CodeIssue]:
        """Analyze code using the local LLM"""
        if not self.pipeline:
            return []
        
        # Enhanced prompt for better C# analysis
        language_specific_guidance = {
            'csharp': """
Focus on C# specific issues:
- Async/await usage (.Result, .Wait() usage)
- Exception handling (empty catch blocks, generic Exception catching)
- Memory management (IDisposable, using statements)
- Performance (string concatenation, HttpClient usage)
- Security (Process.Start, file operations)
- Best practices (PascalCase, dependency injection)
""",
            'python': """
Focus on Python specific issues:
- Import practices (wildcard imports)
- Exception handling (bare except clauses)
- Code style (PEP 8 violations)
- Performance (string concatenation, list comprehensions)
""",
            'javascript': """
Focus on JavaScript specific issues:
- Variable declarations (var vs let/const)
- Equality comparisons (== vs ===)
- Security (eval usage, XSS vulnerabilities)
- Async patterns (callback hell, promise usage)
"""
        }
        
        specific_guidance = language_specific_guidance.get(language, "")

        prompt = f"""You are a senior code reviewer specializing in {language}. Analyze the following code for issues:

{specific_guidance}

Code file: {file_path}
```{language}
{code_content[:2000]}  # Limit code length for context
```

Find issues related to:
1. Security vulnerabilities
2. Performance problems  
3. Best practices violations
4. Code maintainability
5. Potential bugs

Provide analysis in this JSON format:
{{
    "issues": [
        {{
            "line_number": 5,
            "severity": "high|medium|low",
            "category": "security|performance|style|maintainability|best_practice",
            "description": "Brief description of the issue",
            "suggestion": "Specific suggestion to fix this issue"
        }}
    ]
}}

Focus on finding real, actionable issues. Be specific about line numbers and provide concrete suggestions."""

        try:
            response = self.pipeline(prompt, max_new_tokens=512, temperature=0.1)[0]['generated_text']
            # Extract JSON from response (look for content after the prompt)
            response_content = response[len(prompt):].strip()
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    analysis = json.loads(json_match.group())
                    issues = []
                    for issue in analysis.get('issues', []):
                        # Find the actual code line
                        lines = code_content.split('\n')
                        line_num = issue.get('line_number', 1)
                        
                        # Ensure line number is valid
                        if line_num < 1:
                            line_num = 1
                        elif line_num > len(lines):
                            line_num = len(lines)
                            
                        code_snippet = lines[line_num-1] if 0 < line_num <= len(lines) else ""
                        
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=line_num,
                            severity=issue.get('severity', 'low'),
                            category=issue.get('category', 'general'),
                            description=issue.get('description', ''),
                            suggestion=issue.get('suggestion', ''),
                            code_snippet=code_snippet.strip()
                        ))
                    return issues
                except json.JSONDecodeError as e:
                    st.warning(f"JSON decode error for {file_path}: {str(e)}")
                    # Try to extract issues from text response
                    return self._parse_text_response(response_content, code_content, file_path)
        except Exception as e:
            st.warning(f"LLM analysis failed for {file_path}: {str(e)}")
        
        return []
    
    def _parse_text_response(self, response: str, code_content: str, file_path: str) -> List[CodeIssue]:
        """Fallback method to parse non-JSON LLM responses"""
        issues = []
        lines = code_content.split('\n')
        
        # Look for common patterns in text responses
        issue_patterns = [
            r'line\s+(\d+)[:\s]+(.*?)(?:suggestion|fix|recommendation)[:\s]+(.*?)(?:\n|$)',
            r'(\d+)[:\s]+(high|medium|low)[:\s]+(.*?)(?:\n|$)'
        ]
        
        for pattern in issue_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                try:
                    line_num = int(match.group(1))
                    if len(match.groups()) >= 3:
                        description = match.group(2).strip()
                        suggestion = match.group(3).strip()
                        severity = 'medium'
                    else:
                        severity = match.group(2) if match.group(2).lower() in ['high', 'medium', 'low'] else 'medium'
                        description = match.group(3).strip()
                        suggestion = "Review and fix this issue"
                    
                    if 0 < line_num <= len(lines):
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=line_num,
                            severity=severity,
                            category='general',
                            description=description,
                            suggestion=suggestion,
                            code_snippet=lines[line_num-1].strip()
                        ))
                except (ValueError, IndexError):
                    continue
        
        return issues

class CodeAnalyzer:
    """Main code analysis engine"""
    
    def __init__(self):
        self.llm_manager = LocalLLMManager()
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cs': 'csharp',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.sql': 'sql',
            '.sh': 'bash',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css'
        }
        
       # Basic pattern-based rules for quick analysis
        self.pattern_rules = {
            'python': {
                r'print\s*\(': ('low', 'style', 'Consider using logging instead of print'),
                r'except\s*:': ('high', 'best_practice', 'Avoid bare except clauses'),
                r'from\s+\w+\s+import\s+\*': ('medium', 'best_practice', 'Avoid wildcard imports'),
                r'TODO|FIXME|HACK': ('medium', 'maintenance', 'Unresolved technical debt'),
                r'def\s+\w+\([^)]*\):[^{]*\n(\s{4}[^\n]*\n){25,}': ('medium', 'maintainability', 'Function is too long (>25 lines)')
            },
            'javascript': {
                r'console\.log': ('low', 'style', 'Remove console.log in production'),
                r'var\s+': ('medium', 'best_practice', 'Use let or const instead of var'),
                r'==(?!=)': ('medium', 'best_practice', 'Use strict equality (===)'),
                r'eval\s*\(': ('high', 'security', 'Avoid eval() - security risk'),
                r'function\s+\w+\s*\([^)]*\)\s*{[^}]{500,}': ('medium', 'maintainability', 'Function is too long'),
                r'innerHTML\s*=': ('medium', 'security', 'Potential XSS vulnerability with innerHTML')
            },
            'typescript': {
                r'console\.log': ('low', 'style', 'Remove console.log in production'),
                r'var\s+': ('medium', 'best_practice', 'Use let or const instead of var'),
                r'==(?!=)': ('medium', 'best_practice', 'Use strict equality (===)'),
                r'any\s+\w+': ('medium', 'best_practice', 'Avoid using "any" type, use specific types'),
                r'@ts-ignore': ('medium', 'best_practice', 'Avoid @ts-ignore, fix the underlying issue')
            },
            'java': {
                r'System\.out\.print': ('low', 'style', 'Use logging framework instead of System.out'),
                r'catch\s*\([^)]*\)\s*\{\s*\}': ('high', 'best_practice', 'Empty catch block - handle exceptions properly'),
                r'String\s+\w+\s*=\s*"[^"]*"\s*\+': ('medium', 'performance', 'Use StringBuilder for string concatenation'),
                r'public\s+class\s+\w+\s*{[^}]{1000,}': ('medium', 'maintainability', 'Class is too large'),
                r'\.printStackTrace\(\)': ('medium', 'best_practice', 'Use proper logging instead of printStackTrace')
            },
            'csharp': {
                r'Console\.WriteLine?': ('low', 'style', 'Use proper logging framework (ILogger, Serilog, NLog)'),
                r'Console\.Write(?!Line)': ('low', 'style', 'Use proper logging framework instead of Console.Write'),
                r'catch\s*\([^)]*\)\s*\{\s*\}': ('high', 'best_practice', 'Empty catch block - handle exceptions properly'),
                r'catch\s*\(\s*Exception\s+\w*\s*\)\s*\{[^}]*\}': ('medium', 'best_practice', 'Catching generic Exception - use specific exception types'),
                r'public\s+class\s+\w+[^{]*\{[^}]{800,}': ('medium', 'maintainability', 'Class is too large - consider breaking into smaller classes'),
                r'string\s+\w+\s*=\s*"[^"]*"\s*\+': ('medium', 'performance', 'Use StringBuilder or string interpolation for concatenation'),
                r'\.ToString\(\)\s*\+': ('medium', 'performance', 'Use string interpolation instead of ToString() + concatenation'),
                r'new\s+Exception\s*\(': ('low', 'best_practice', 'Use specific exception types instead of generic Exception'),
                r'Thread\.Sleep\s*\(': ('medium', 'best_practice', 'Use async/await with Task.Delay instead of Thread.Sleep'),
                r'ConfigurationManager\.AppSettings': ('medium', 'best_practice', 'Consider using IConfiguration for dependency injection'),
                r'DateTime\.Now': ('low', 'best_practice', 'Consider using DateTime.UtcNow for UTC times'),
                r'\.Result\b': ('medium', 'best_practice', 'Avoid .Result - use await instead to prevent deadlocks'),
                r'\.Wait\(\)': ('medium', 'best_practice', 'Avoid .Wait() - use await instead to prevent deadlocks'),
                r'Task\.Run\s*\(\s*\(\)\s*=>\s*{[^}]*}\s*\)\.Result': ('high', 'best_practice', 'Potential deadlock - use async/await properly'),
                r'public\s+\w+\s+\w+\s*{[^}]*get\s*;[^}]*set\s*;[^}]*}': ('low', 'style', 'Consider using auto-properties or init-only setters'),
                r'#region': ('low', 'style', 'Regions often indicate code organization issues'),
                r'goto\s+': ('high', 'best_practice', 'Avoid goto statements - use structured programming'),
                r'Activator\.CreateInstance': ('medium', 'performance', 'Consider using dependency injection instead of Activator'),
                r'GC\.Collect\s*\(': ('medium', 'performance', 'Avoid manual GC.Collect() calls'),
                r'Assembly\.LoadFrom|Assembly\.LoadFile': ('medium', 'security', 'Loading assemblies can be a security risk'),
                r'Process\.Start\s*\(': ('medium', 'security', 'Process.Start can be a security risk - validate inputs'),
                r'File\.ReadAllText\s*\([^)]*\)': ('low', 'performance', 'Consider using async File.ReadAllTextAsync for large files'),
                r'HttpClient\s+\w+\s*=\s*new\s+HttpClient': ('medium', 'performance', 'HttpClient should be reused or use IHttpClientFactory'),
                r'using\s*\(\s*var\s+\w+\s*=\s*new\s+HttpClient': ('medium', 'performance', 'HttpClient in using statement can cause socket exhaustion')
            },
        }
    
    def extract_zip(self, zip_file) -> str:
        """Extract ZIP file to temporary directory"""
        temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            return temp_dir
        except Exception as e:
            st.error(f"Error extracting ZIP file: {str(e)}")
            return None
    
    def get_code_files(self, directory: str) -> Dict[str, str]:
        """Get all code files from directory"""
        code_files = {}
        for root, dirs, files in os.walk(directory):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.vscode', '.idea'}]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.supported_extensions:
                    try:
                        relative_path = str(file_path.relative_to(directory))
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        if content.strip():  # Only non-empty files
                            code_files[relative_path] = content
                    except Exception as e:
                        st.warning(f"Could not read file {file_path}: {str(e)}")
        
        return code_files
    
    def detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        return self.supported_extensions.get(ext, 'unknown')
    
    def analyze_with_patterns(self, content: str, language: str, file_path: str) -> List[CodeIssue]:
        """Quick pattern-based analysis"""
        issues = []
        if language not in self.pattern_rules:
            return issues
        
        lines = content.split('\n')
        
        # Add debug info for C# files
        if language == 'csharp':
            st.info(f"🔍 Analyzing C# file: {file_path} ({len(lines)} lines)")
        
        for i, line in enumerate(lines, 1):
            for pattern, (severity, category, description) in self.pattern_rules[language].items():
                try:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=i,
                            severity=severity,
                            category=category,
                            description=description,
                            suggestion="Review and apply best practices",
                            code_snippet=line.strip()
                        ))
                        
                        # Debug output for C# matches
                        if language == 'csharp':
                            st.write(f"✅ Found issue on line {i}: {description}")
                            
                except re.error as e:
                    # Log regex errors but continue
                    if language == 'csharp':
                        st.warning(f"Regex error in pattern '{pattern}': {e}")
                    continue
        
        # Add some generic C# checks
        if language == 'csharp':
            issues.extend(self._additional_csharp_checks(content, file_path, lines))
        
        return issues
    
    def _additional_csharp_checks(self, content: str, file_path: str, lines: List[str]) -> List[CodeIssue]:
        """Additional C# specific checks"""
        issues = []
        
        # Check for missing using statements
        if 'System' not in content and any(word in content for word in ['Console', 'String', 'DateTime']):
            issues.append(CodeIssue(
                file_path=file_path,
                line_number=1,
                severity='low',
                category='style',
                description='Missing using System; statement',
                suggestion='Add using System; at the top of the file',
                code_snippet='// Missing using System;'
            ))
        
        # Check for long methods (basic heuristic)
        in_method = False
        method_start_line = 0
        method_lines = 0
        brace_count = 0
        
        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Simple method detection
            if re.search(r'(public|private|protected|internal)\s+.*\s+\w+\s*\([^)]*\)\s*{?', stripped_line):
                if not in_method:
                    in_method = True
                    method_start_line = i
                    method_lines = 0
                    brace_count = stripped_line.count('{') - stripped_line.count('}')
                    
            if in_method:
                method_lines += 1
                brace_count += stripped_line.count('{') - stripped_line.count('}')
                
                if brace_count <= 0 and method_lines > 30:
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=method_start_line,
                        severity='medium',
                        category='maintainability',
                        description=f'Method is too long ({method_lines} lines)',
                        suggestion='Consider breaking this method into smaller methods',
                        code_snippet=lines[method_start_line-1].strip()
                    ))
                    in_method = False
                elif brace_count <= 0:
                    in_method = False
        
        # Check for hardcoded strings
        for i, line in enumerate(lines, 1):
            if re.search(r'"[^"]{20,}"', line) and 'const' not in line.lower():
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i,
                    severity='low',
                    category='maintainability',
                    description='Long hardcoded string found',
                    suggestion='Consider using constants or resource files for long strings',
                    code_snippet=line.strip()
                ))
        
        return issues
    def calculate_metrics(self, code_files: Dict[str, str], all_issues: List[CodeIssue]) -> Tuple[float, float]:
        """Calculate quality and maintainability scores"""
        total_lines = sum(len(content.split('\n')) for content in code_files.values())
        
        # Quality score based on issues per line
        high_issues = len([i for i in all_issues if i.severity == 'high'])
        medium_issues = len([i for i in all_issues if i.severity == 'medium'])
        low_issues = len([i for i in all_issues if i.severity == 'low'])
        
        issue_density = (high_issues * 3 + medium_issues * 2 + low_issues) / max(total_lines, 1) * 1000
        quality_score = max(0, 100 - issue_density * 10)
        
        # Maintainability score
        avg_file_size = total_lines / max(len(code_files), 1)
        maintainability_score = max(0, 100 - (avg_file_size / 50) - (high_issues * 5))
        
        return quality_score, maintainability_score
    
    def analyze_project(self, zip_file, use_llm: bool = True) -> AnalysisResult:
        """Main analysis function"""
        # Extract ZIP
        temp_dir = self.extract_zip(zip_file)
        if not temp_dir:
            return None
        
        try:
            # Get code files
            code_files = self.get_code_files(temp_dir)
            if not code_files:
                st.warning("No code files found in the ZIP archive")
                return None
            
            # Detect languages
            languages = set()
            all_issues = []
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(code_files)
            
            for i, (file_path, content) in enumerate(code_files.items()):
                progress = (i + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {file_path}... ({i+1}/{total_files})")
                
                language = self.detect_language(file_path)
                if language != 'unknown':
                    languages.add(language)
                
                # Pattern-based analysis (fast)
                pattern_issues = self.analyze_with_patterns(content, language, file_path)
                all_issues.extend(pattern_issues)
                
                # LLM analysis (slower, more comprehensive)
                if use_llm and self.llm_manager.pipeline and len(content) < 5000:
                    llm_issues = self.llm_manager.analyze_code_with_llm(content, file_path, language)
                    all_issues.extend(llm_issues)
            
            progress_bar.empty()
            status_text.empty()
            
            # Calculate metrics
            quality_score, maintainability_score = self.calculate_metrics(code_files, all_issues)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(languages, all_issues)
            
            return AnalysisResult(
                total_files=len(code_files),
                total_lines=sum(len(content.split('\n')) for content in code_files.values()),
                languages_detected=list(languages),
                issues=all_issues,
                quality_score=quality_score,
                maintainability_score=maintainability_score,
                recommendations=recommendations
            )
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def generate_recommendations(self, languages: set, issues: List[CodeIssue]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Language-specific recommendations
        if 'python' in languages:
            recommendations.extend([
                "Follow PEP 8 style guidelines",
                "Use type hints for better code documentation",
                "Implement proper error handling with specific exception types"
            ])
        
        if 'javascript' in languages:
            recommendations.extend([
                "Use ESLint for consistent code style",
                "Implement proper async/await error handling",
                "Use const/let instead of var for variable declarations"
            ])
        
        if 'csharp' in languages:
            recommendations.extend([
                "Follow C# coding conventions and naming standards",
                "Use async/await pattern consistently",
                "Implement proper exception handling with specific exception types",
                "Consider using dependency injection for better testability"
            ])

        # Issue-based recommendations
        high_issues = [i for i in issues if i.severity == 'high']
        if high_issues:
            recommendations.append(f"Priority: Fix {len(high_issues)} high-severity issues first")
        
        security_issues = [i for i in issues if i.category == 'security']
        if security_issues:
            recommendations.append("Implement security code review process")
        
        performance_issues = [i for i in issues if i.category == 'performance']
        if performance_issues:
            recommendations.append("Review and optimize performance-related issues")

        return recommendations

def create_visualizations(result: AnalysisResult):
    """Create visualizations for the analysis results"""
    
    # Issues by severity
    severity_counts = {}
    for issue in result.issues:
        severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
    
    if severity_counts:
        fig_severity = px.pie(
            values=list(severity_counts.values()),
            names=list(severity_counts.keys()),
            title="Issues by Severity",
            color_discrete_map={'high': '#ff4444', 'medium': '#ffaa00', 'low': '#44ff44'}
        )
        st.plotly_chart(fig_severity, use_container_width=True)
    
    # Issues by category
    category_counts = {}
    for issue in result.issues:
        category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
    
    if category_counts:
        fig_category = px.bar(
            x=list(category_counts.keys()),
            y=list(category_counts.values()),
            title="Issues by Category",
            labels={'x': 'Category', 'y': 'Number of Issues'}
        )
        st.plotly_chart(fig_category, use_container_width=True)
    # Issues by file (top 10)
    file_counts = {}
    for issue in result.issues:
        file_counts[issue.file_path] = file_counts.get(issue.file_path, 0) + 1
    
    if file_counts:
        # Get top 10 files with most issues
        top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_files:
            fig_files = px.bar(
                x=[count for _, count in top_files],
                y=[file for file, _ in top_files],
                orientation='h',
                title="Top Files by Issue Count",
                labels={'x': 'Number of Issues', 'y': 'File Path'}
            )
            fig_files.update_layout(height=400)
            st.plotly_chart(fig_files, use_container_width=True)

    # Metrics gauge
    col1, col2 = st.columns(2)
    
    with col1:
        fig_quality = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = result.quality_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Code Quality Score"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        fig_maintainability = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = result.maintainability_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Maintainability Score"},
            delta = {'reference': 75},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        st.plotly_chart(fig_maintainability, use_container_width=True)

def filter_issues(issues: List[CodeIssue], severity_filter: str, category_filter: str, file_filter: str) -> List[CodeIssue]:
    """Filter issues based on severity, category, and file"""
    filtered = issues
    
    # Filter by severity
    if severity_filter and severity_filter.lower() != "all":
        filtered = [issue for issue in filtered if issue.severity.lower() == severity_filter.lower()]
    
    # Filter by category  
    if category_filter and category_filter.lower() != "all":
        filtered = [issue for issue in filtered if issue.category.lower() == category_filter.lower()]
    
    # Filter by file
    if file_filter and file_filter.lower() != "all":
        filtered = [issue for issue in filtered if file_filter.lower() in issue.file_path.lower()]
    
    return filtered

def main():
    st.set_page_config(
        page_title="AI Code Review Tool",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 AI-Powered Code Review Tool")
    st.markdown("Upload your project ZIP file for comprehensive code analysis using local LLM")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "Deepseek Coder 1.3B (Recommended)": "deepseek-ai/deepseek-coder-1.3b-instruct",
            "CodeLlama 7B": "codellama/CodeLlama-7b-Instruct-hf",
            "Deepseek Coder 6.7B": "deepseek-ai/deepseek-coder-6.7b-instruct"
        }
        
        selected_model_name = st.selectbox(
            "Select LLM Model",
            options=list(model_options.keys()),
            help="Choose based on your VRAM. 1.3B model recommended for your setup."
        )
        
        selected_model = model_options[selected_model_name]
        
        # Check cache status
        if TRANSFORMERS_AVAILABLE:
            analyzer = CodeAnalyzer()
            cache_info = analyzer.llm_manager.check_model_cached(selected_model)
            
            if cache_info['is_cached']:
                if cache_info.get('has_all_required', False):
                    st.success("✅ Model Available Locally")
                    st.session_state.model_loaded = True
                    st.info(f"📁 Size: {analyzer.llm_manager._format_size(cache_info['cache_size'])}")
                else:
                    st.warning("⚠️ Incomplete Model Cache")
                    missing = cache_info.get('missing_files', [])
                    st.caption(f"Missing: {', '.join(missing)}")
                
                with st.expander("📂 Cache Details"):
                    st.write(f"**Cache Path:** {cache_info['cache_path']}")
                    st.write(f"**Files Count:** {len(cache_info['files_found'])}")
                    
                    # Show model files specifically
                    model_files = cache_info.get('model_files_found', [])
                    if model_files:
                        st.write("**Model Files:**")
                        for file in model_files[:3]:  # Show first 3 model files
                            st.write(f"• {file['name']} ({file['size']})")
                    
                    # Show config files
                    config_files = [f for f in cache_info['files_found'] 
                                  if 'config' in f['name'] or 'tokenizer' in f['name']]
                    if config_files:
                        st.write("**Config Files:**")
                        for file in config_files[:3]:
                            st.write(f"• {file['name']} ({file['size']})")
                
                # Option to clear cache
                if st.button("🗑️ Clear Model Cache", help="Remove cached model to free space"):
                    if analyzer.llm_manager.clear_model_cache(selected_model):
                        st.success("Cache cleared successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to clear cache")
                        
                # Offline mode toggle
                if cache_info.get('has_all_required', False):
                    st.info("🔒 **Offline Mode Available** - Model can run without internet")
                        
            else:
                st.warning("⬇️ Model Not Cached")
                estimated_size = analyzer.llm_manager.estimate_download_size(selected_model)
                st.info(f"📥 Download size: ~{estimated_size}")
                st.caption("💡 First download will take 5-15 minutes")
                
                # Check for partial downloads
                if 'error' not in cache_info and cache_info['files_found']:
                    st.caption(f"⚠️ Found {len(cache_info['files_found'])} partial files")
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            offline_mode = st.checkbox(
                "Force Offline Mode", 
                help="Only use locally cached files, don't download anything"
            )
            
            if offline_mode:
                st.info("🔒 Offline mode enabled - will only use cached models")
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_HUB_OFFLINE'] = '1'
            else:
                # Remove offline environment variables if they exist
                os.environ.pop('TRANSFORMERS_OFFLINE', None)
                os.environ.pop('HF_HUB_OFFLINE', None)
        
        use_llm = st.checkbox(
            "Enable LLM Analysis", 
            value=True,
            help="Uncheck for faster pattern-based analysis only"
        )
        
        # Model loading section
        st.subheader("🚀 Model Management")
        
        # Initialize session state if not exists
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'current_model' not in st.session_state:
            st.session_state.current_model = None
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = None
        
        # Check if current model matches selected model
        if st.session_state.current_model == None:
            model_changed = cache_info.get("model_name", "") != selected_model
        else: 
            model_changed = st.session_state.current_model != selected_model
        if model_changed and st.session_state.model_loaded:
            st.warning("⚠️ Model changed - please reload")
            st.session_state.model_loaded = False
            st.session_state.analyzer = None
        else:
             st.session_state.model_loaded = True
             st.session_state.current_model = selected_model
             st.session_state.analyzer = analyzer
        
        # Show current model status
        if st.session_state.get('model_loaded', False):
            st.success("✅ Model Ready")
            st.info(f"📋 Current: {st.session_state.current_model.split('/')[-1]}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reload Model"):
                    st.session_state.model_loaded = False
                    st.session_state.analyzer = None
                    st.rerun()
            with col2:
                if st.button("❌ Unload Model"):
                    st.session_state.model_loaded = False
                    st.session_state.current_model = None
                    st.session_state.analyzer = None
                    # Clear GPU memory if available
                    if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    st.rerun()
        else:
            if st.button("🚀 Load Model", type="primary"):
                if not TRANSFORMERS_AVAILABLE:
                    st.error("Please install required packages first!")
                    st.code("pip install transformers torch accelerate bitsandbytes")
                else:
                    with st.spinner("Loading model... Please wait..."):
                        # Create analyzer instance
                        analyzer = CodeAnalyzer()
                        
                        # Show progress information
                        cache_info = analyzer.llm_manager.check_model_cached(selected_model)
                        if not cache_info.get('is_cached', False) or not cache_info.get('has_all_required', False):
                            st.info("📥 Downloading model for first time...")
                            st.info("⏳ This may take 5-15 minutes")
                        else:
                            st.info("📂 Loading from cache...")
                            st.info("⏳ This should take 1-2 minutes")
                        
                        success = analyzer.llm_manager.load_model(selected_model)
                        if success:
                            # Store in session state
                            st.session_state.model_loaded = True
                            st.session_state.current_model = selected_model
                            st.session_state.analyzer = analyzer
                            st.success("🎉 Model loaded successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to load model")
                            st.session_state.model_loaded = False
                            st.session_state.analyzer = None
        
        # System information
        st.subheader("💻 System Info")
        
        # GPU info
        if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            st.success(f"🎮 GPU: {gpu_name}")
            st.info(f"📊 VRAM: {gpu_memory:.1f} GB")
            
            if torch.cuda.memory_allocated() > 0:
                memory_used = torch.cuda.memory_allocated() / 1024**3
                st.info(f"🔥 GPU Memory Used: {memory_used:.1f} GB")
        else:
            st.warning("⚠️ No GPU detected - using CPU")
        
        # Cache directory info
        if TRANSFORMERS_AVAILABLE:
            cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
            if os.path.exists(cache_dir):
                cache_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(cache_dir)
                    for filename in filenames
                ) / 1024**3
                st.info(f"💾 Total Cache: {cache_size:.1f} GB")
            else:
                st.info("💾 No cache directory found")
    
    # Main interface
    uploaded_file = st.file_uploader(
        "Choose a ZIP file containing your code project",
        type=['zip'],
        help="Upload a ZIP file containing your source code"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            analyze_button = st.button("🔍 Analyze Code", type="primary")
        
        if analyze_button:
            if use_llm and not st.session_state.get('model_loaded', False):
                st.warning("⚠️ Please load the LLM model first, or disable LLM analysis for pattern-based review.")
                return
            
            with st.spinner("Analyzing your code... This may take a few minutes."):
                analyzer = CodeAnalyzer()
                
                # Analysis
                start_time = time.time()
                result = analyzer.analyze_project(uploaded_file, use_llm=use_llm)
                analysis_time = time.time() - start_time
                
                if result:
                    st.success(f"✅ Analysis completed in {analysis_time:.1f} seconds!")
                    # Store result in session state for filtering
                    st.session_state.analysis_result = result
                    
                    # Display results
                    st.header("📊 Analysis Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Files Analyzed", result.total_files)
                    with col2:
                        st.metric("Lines of Code", f"{result.total_lines:,}")
                    with col3:
                        st.metric("Issues Found", len(result.issues))
                    with col4:
                        st.metric("Languages", len(result.languages_detected))
                    
                    # Languages detected
                    if result.languages_detected:
                        st.subheader("🔤 Languages Detected")
                        for lang in result.languages_detected:
                            st.caption(f":blue[{lang.upper()}]")
                    
                    # Visualizations
                    if result.issues:
                        st.subheader("📈 Visual Analysis")
                        create_visualizations(result)
                    
                    # Issues breakdown
                    if result.issues:
                        st.subheader("🚨 Issues Found")
                        
                     # Create filter columns
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            # Severity filter
                            severity_options = ["All"] + sorted(list(set(issue.severity for issue in result.issues)))
                            severity_filter = st.selectbox(
                                "Filter by Severity:",
                                options=severity_options,
                                index=0
                            )
                        
                        with col2:
                            # Category filter
                            category_options = ["All"] + sorted(list(set(issue.category for issue in result.issues)))
                            category_filter = st.selectbox(
                                "Filter by Category:",
                                options=category_options,
                                index=0
                            )
                        
                        with col3:
                            # File filter
                            file_options = ["All"] + sorted(list(set(issue.file_path for issue in result.issues)))
                            file_filter = st.selectbox(
                                "Filter by File:",
                                options=file_options,
                                index=0
                            )
                        
                        with col4:
                            # Search filter
                            search_filter = st.text_input(
                                "Search Issues:",
                                placeholder="Search in descriptions...",
                                help="Search in issue descriptions and suggestions"
                            )
                        
                        # Apply filters
                        filtered_issues = filter_issues(
                            result.issues, 
                            severity_filter, 
                            category_filter, 
                            file_filter
                        )
                        
                        # Apply search filter
                        if search_filter:
                            search_lower = search_filter.lower()
                            filtered_issues = [
                                issue for issue in filtered_issues 
                                if (search_lower in issue.description.lower() or 
                                    search_lower in issue.suggestion.lower() or
                                    search_lower in issue.code_snippet.lower())
                            ]
                        
                        # Display filter results
                        st.info(f"Showing {len(filtered_issues)} of {len(result.issues)} total issues")
                        
                        # Pagination
                        issues_per_page = 20
                        total_pages = (len(filtered_issues) + issues_per_page - 1) // issues_per_page
                        
                        if total_pages > 1:
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                page = st.selectbox(
                                    "Page:",
                                    options=list(range(1, total_pages + 1)),
                                    index=0,
                                    format_func=lambda x: f"Page {x} of {total_pages}"
                                )
                            start_idx = (page - 1) * issues_per_page
                            end_idx = start_idx + issues_per_page
                            page_issues = filtered_issues[start_idx:end_idx]
                        else:
                            page_issues = filtered_issues
                        
                        # Display issues
                        for i, issue in enumerate(page_issues):
                            severity_color = {
                                'high': '🔴',
                                'medium': '🟡', 
                                'low': '🟢'
                            }.get(issue.severity, '⚪')
                            
                            # Create expandable issue display
                            with st.expander(
                                f"{severity_color} **{issue.severity.upper()}** | {issue.description} | {issue.file_path}:{issue.line_number}",
                                expanded=False
                            ):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**📁 File:** `{issue.file_path}`")
                                    st.write(f"**📍 Line:** {issue.line_number}")
                                    st.write(f"**🔖 Category:** {issue.category}")
                                
                                with col2:
                                    st.write(f"**⚠️ Severity:** {issue.severity}")
                                    st.write(f"**📝 Description:** {issue.description}")
                                
                                st.write(f"**💡 Suggestion:** {issue.suggestion}")
                                
                                if issue.code_snippet:
                                    st.write("**📄 Code Snippet:**")
                                    st.code(issue.code_snippet, language=analyzer.detect_language(issue.file_path))
                        
                        # Show pagination info at bottom
                        if total_pages > 1:
                            st.info(f"Showing page {page} of {total_pages} ({len(page_issues)} issues on this page)")
                    
                    # Recommendations
                    if result.recommendations:
                        st.subheader("💡 Recommendations")
                        for i, rec in enumerate(result.recommendations, 1):
                            st.write(f"{i}. {rec}")
                    
                    # Export results
                    st.subheader("📥 Export Results")
                    
                    # Prepare export data
                    export_data = {
                        'summary': {
                            'total_files': result.total_files,
                            'total_lines': result.total_lines,
                            'languages': result.languages_detected,
                            'quality_score': result.quality_score,
                            'maintainability_score': result.maintainability_score,
                            'analysis_date': datetime.now().isoformat(),
                            'total_issues': len(result.issues),
                            'high_severity_issues': len([i for i in result.issues if i.severity == 'high']),
                            'medium_severity_issues': len([i for i in result.issues if i.severity == 'medium']),
                            'low_severity_issues': len([i for i in result.issues if i.severity == 'low'])
                        },
                        'issues': [
                            {
                                'file': issue.file_path,
                                'line': issue.line_number,
                                'severity': issue.severity,
                                'category': issue.category,
                                'description': issue.description,
                                'suggestion': issue.suggestion,
                                'code': issue.code_snippet
                            }
                            for issue in result.issues
                        ],
                        'recommendations': result.recommendations
                    }
                    
                    st.download_button(
                        label="📄 Download Detailed Report (JSON)",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"code_review_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                with col2:
                        # Create CSV export for issues
                        import io
                        csv_buffer = io.StringIO()
                        
                        # Create DataFrame for issues
                        issues_data = []
                        for issue in result.issues:
                            issues_data.append({
                                'File': issue.file_path,
                                'Line': issue.line_number,
                                'Severity': issue.severity,
                                'Category': issue.category,
                                'Description': issue.description,
                                'Suggestion': issue.suggestion,
                                'Code Snippet': issue.code_snippet
                            })
                        
                        if issues_data:
                            df = pd.DataFrame(issues_data)
                            csv_content = df.to_csv(index=False)
                            
                            st.download_button(
                                label="📊 Download Issues (CSV)",
                                data=csv_content,
                                file_name=f"code_issues_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    # Additional statistics
                with st.expander("📈 Detailed Statistics"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Issues by Severity:**")
                            severity_counts = {}
                            for issue in result.issues:
                                severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
                            for severity, count in sorted(severity_counts.items()):
                                st.write(f"• {severity.title()}: {count}")
                        
                        with col2:
                            st.write("**Issues by Category:**")
                            category_counts = {}
                            for issue in result.issues:
                                category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
                            for category, count in sorted(category_counts.items()):
                                st.write(f"• {category.title()}: {count}")
                        
                        with col3:
                            st.write("**Files with Most Issues:**")
                            file_counts = {}
                            for issue in result.issues:
                                file_counts[issue.file_path] = file_counts.get(issue.file_path, 0) + 1
                            top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                            for file_path, count in top_files:
                                st.write(f"• {file_path}: {count}")
                    
                    # Clear results button
                if st.button("🗑️ Clear Results", help="Clear analysis results to start fresh"):
                        if 'analysis_result' in st.session_state:
                            del st.session_state.analysis_result
                        st.rerun()

    # Show previous results if available
    elif 'analysis_result' in st.session_state:
        result = st.session_state.analysis_result
        st.info("📋 Showing previous analysis results. Upload a new file to analyze again.")
        
        # Display results (same as above but without the analysis step)
        st.header("📊 Previous Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Files Analyzed", result.total_files)
        with col2:
            st.metric("Lines of Code", f"{result.total_lines:,}")
        with col3:
            st.metric("Issues Found", len(result.issues))
        with col4:
            st.metric("Languages", len(result.languages_detected))
        
        # Languages detected
        if result.languages_detected:
            st.subheader("🔤 Languages Detected")
            for lang in result.languages_detected:
                st.caption(f":blue[{lang.upper()}]")
        
        # Issues with filtering (same filtering logic as above)
        if result.issues:
            st.subheader("🚨 Issues Found")
            
            # Create filter columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                severity_options = ["All"] + sorted(list(set(issue.severity for issue in result.issues)))
                severity_filter = st.selectbox(
                    "Filter by Severity:",
                    options=severity_options,
                    index=0,
                    key="prev_severity"
                )
            
            with col2:
                category_options = ["All"] + sorted(list(set(issue.category for issue in result.issues)))
                category_filter = st.selectbox(
                    "Filter by Category:",
                    options=category_options,
                    index=0,
                    key="prev_category"
                )
            
            with col3:
                file_options = ["All"] + sorted(list(set(issue.file_path for issue in result.issues)))
                file_filter = st.selectbox(
                    "Filter by File:",
                    options=file_options,
                    index=0,
                    key="prev_file"
                )
            
            with col4:
                search_filter = st.text_input(
                    "Search Issues:",
                    placeholder="Search in descriptions...",
                    help="Search in issue descriptions and suggestions",
                    key="prev_search"
                )
            
            # Apply filters
            filtered_issues = filter_issues(
                result.issues, 
                severity_filter, 
                category_filter, 
                file_filter
            )
            
            # Apply search filter
            if search_filter:
                search_lower = search_filter.lower()
                filtered_issues = [
                    issue for issue in filtered_issues 
                    if (search_lower in issue.description.lower() or 
                        search_lower in issue.suggestion.lower() or
                        search_lower in issue.code_snippet.lower())
                ]
            
            # Display filter results
            st.info(f"Showing {len(filtered_issues)} of {len(result.issues)} total issues")
            
            # Display first 20 issues (simplified for previous results)
            for issue in filtered_issues[:20]:
                severity_color = {
                    'high': '🔴',
                    'medium': '🟡', 
                    'low': '🟢'
                }.get(issue.severity, '⚪')
                
                with st.expander(
                    f"{severity_color} **{issue.severity.upper()}** | {issue.description} | {issue.file_path}:{issue.line_number}",
                    expanded=False
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**📁 File:** `{issue.file_path}`")
                        st.write(f"**📍 Line:** {issue.line_number}")
                        st.write(f"**🔖 Category:** {issue.category}")
                    
                    with col2:
                        st.write(f"**⚠️ Severity:** {issue.severity}")
                        st.write(f"**📝 Description:** {issue.description}")
                    
                    st.write(f"**💡 Suggestion:** {issue.suggestion}")
                    
                    if issue.code_snippet:
                        st.write("**📄 Code Snippet:**")
                        analyzer = CodeAnalyzer()  # Create temporary analyzer for language detection
                        st.code(issue.code_snippet, language=analyzer.detect_language(issue.file_path))
            
            if len(filtered_issues) > 20:
                st.info(f"Showing first 20 of {len(filtered_issues)} filtered issues")

if __name__ == "__main__":
    main()