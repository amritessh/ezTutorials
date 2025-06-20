import re
import hashlib
import hmac
from typing import Set

class SecurityManager:
    """Comprehensive security and privacy management."""
    
    def __init__(self, config: VideoGPTConfig, logger: VideoGPTLogger):
        self.config = config
        self.logger = logger
        self.secret_key = secrets.token_hex(32)
        self.setup_security_patterns()
        self.rate_limiter = RateLimiter()
        self.blocked_patterns = set()
    
    def setup_security_patterns(self):
        """Initialize security patterns for detection."""
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b')
        }
        
        self.dangerous_patterns = [
            r'<script.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
            r'<iframe.*?>',
            r'<object.*?>',
            r'<embed.*?>'
        ]
        
        self.logger.info("Security patterns initialized", "SECURITY")
    
    def sanitize_input(self, text: str, user_id: str = None) -> Dict[str, Any]:
        """Comprehensive input sanitization."""
        try:
            original_length = len(text)
            sanitized_text = text
            issues_found = []
            
            # Check input length
            max_length = 2000
            if len(text) > max_length:
                sanitized_text = text[:max_length]
                issues_found.append("input_truncated")
                self.logger.debug(f"Input truncated from {original_length} to {max_length}", "SECURITY")
            
            # Remove dangerous patterns
            for pattern in self.dangerous_patterns:
                if re.search(pattern, sanitized_text, re.IGNORECASE):
                    sanitized_text = re.sub(pattern, '[REMOVED]', sanitized_text, flags=re.IGNORECASE)
                    issues_found.append("dangerous_pattern_removed")
                    self.logger.warning(f"Dangerous pattern removed from input", "SECURITY")
            
            # Detect and mask PII
            pii_found = []
            for pii_type, pattern in self.pii_patterns.items():
                matches = pattern.findall(sanitized_text)
                if matches:
                    pii_found.append(pii_type)
                    sanitized_text = pattern.sub(f'[{pii_type.upper()}_REDACTED]', sanitized_text)
                    issues_found.append(f"pii_{pii_type}_masked")
            
            if pii_found:
                self.logger.warning(f"PII detected and masked: {pii_found}", "SECURITY")
            
            # Create user hash for logging (if user_id provided)
            user_hash = None
            if user_id:
                user_hash = self.create_user_hash(user_id)
            
            return {
                'sanitized_text': sanitized_text.strip(),
                'original_length': original_length,
                'final_length': len(sanitized_text),
                'issues_found': issues_found,
                'pii_detected': pii_found,
                'user_hash': user_hash,
                'is_safe': len(issues_found) == 0 or all('truncated' in issue for issue in issues_found)
            }
            
        except Exception as e:
            self.logger.error(f"Input sanitization failed: {str(e)}", "SECURITY")
            return {
                'sanitized_text': '',
                'error': str(e),
                'is_safe': False
            }
    
    def create_user_hash(self, user_id: str) -> str:
        """Create anonymous user hash for tracking."""
        return hmac.new(
            self.secret_key.encode(),
            user_id.encode(),
            hashlib.sha256
        ).hexdigest()[:16]
    
    def validate_youtube_url(self, url: str) -> Dict[str, Any]:
        """Validate YouTube URL for safety."""
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'https?://(?:www\.)?youtu\.be/[\w-]+',
            r'https?://(?:www\.)?youtube\.com/embed/[\w-]+'
        ]
        
        is_valid = any(re.match(pattern, url) for pattern in youtube_patterns)
        
        if not is_valid:
            self.logger.warning(f"Invalid YouTube URL format: {url[:50]}", "SECURITY")
        
        return {
            'is_valid': is_valid,
            'url': url,
            'validation_time': datetime.now()
        }

class RateLimiter:
    """Rate limiting for API calls."""
    
    def __init__(self, max_calls: int = 60, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = {}
    
    def can_make_call(self, user_id: str) -> Dict[str, Any]:
        """Check if user can make a call."""
        now = datetime.now()
        
        if user_id not in self.calls:
            self.calls[user_id] = []
        
        # Remove old calls outside time window
        cutoff_time = now - timedelta(seconds=self.time_window)
        self.calls[user_id] = [
            call_time for call_time in self.calls[user_id] 
            if call_time > cutoff_time
        ]
        
        current_calls = len(self.calls[user_id])
        can_call = current_calls < self.max_calls
        
        if can_call:
            self.calls[user_id].append(now)
        
        return {
            'can_call': can_call,
            'current_calls': current_calls,
            'max_calls': self.max_calls,
            'reset_time': (now + timedelta(seconds=self.time_window)).isoformat()
        }

class SystemMonitor:
    """System monitoring and health checks."""
    
    def __init__(self, config: VideoGPTConfig, logger: VideoGPTLogger, metrics: VideoGPTMetrics):
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.health_checks = {}
        self.alerts = []
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_status = {
            'timestamp': datetime.now(),
            'overall_status': 'healthy',
            'components': {},
            'alerts': []
        }
        
        # Check API keys
        api_key_status = self._check_api_keys()
        health_status['components']['api_keys'] = api_key_status
        
        # Check memory usage
        memory_status = self._check_memory_usage()
        health_status['components']['memory'] = memory_status
        
        # Check error rates
        error_rate_status = self._check_error_rates()
        health_status['components']['error_rates'] = error_rate_status
        
        # Determine overall status
        component_statuses = [comp['status'] for comp in health_status['components'].values()]
        if 'critical' in component_statuses:
            health_status['overall_status'] = 'critical'
        elif 'warning' in component_statuses:
            health_status['overall_status'] = 'warning'
        
        return health_status
    
    def _check_api_keys(self) -> Dict[str, Any]:
        """Check if required API keys are present."""
        required_keys = ['GEMINI_API_KEY']
        missing_keys = []
        
        for key in required_keys:
            if not os.getenv(key):
                missing_keys.append(key)
        
        status = 'healthy' if not missing_keys else 'critical'
        
        return {
            'status': status,
            'missing_keys': missing_keys,
            'checked_at': datetime.now()
        }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage."""
        import psutil
        
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=1)
            
            memory_threshold = 85  # 85% threshold
            cpu_threshold = 80     # 80% threshold
            
            status = 'healthy'
            if memory.percent > memory_threshold or cpu > cpu_threshold:
                status = 'warning'
            if memory.percent > 95 or cpu > 95:
                status = 'critical'
            
            return {
                'status': status,
                'memory_percent': memory.percent,
                'cpu_percent': cpu,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'checked_at': datetime.now()
            }
            
        except ImportError:
            return {
                'status': 'unknown',
                'error': 'psutil not available',
                'checked_at': datetime.now()
            }
    
    def _check_error_rates(self) -> Dict[str, Any]:
        """Check application error rates."""
        total_requests = self.metrics.metrics.get('total_requests', 1)
        failed_requests = self.metrics.metrics.get('failed_requests', 0)
        
        error_rate = (failed_requests / total_requests) * 100
        
        status = 'healthy'
        if error_rate > 10:  # 10% error rate threshold
            status = 'warning'
        if error_rate > 25:  # 25% error rate threshold
            status = 'critical'
        
        return {
            'status': status,
            'error_rate_percent': round(error_rate, 2),
            'total_requests': total_requests,
            'failed_requests': failed_requests,
            'checked_at': datetime.now()
        }
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], user_hash: str = None):
        """Log security-related events."""
        security_event = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'details': details,
            'user_hash': user_hash,
            'severity': details.get('severity', 'info')
        }
        
        self.logger.info(f"Security event: {event_type} - {details}", "SECURITY_MONITOR")
        
        # Store for analysis
        if not hasattr(self, 'security_events'):
            self.security_events = []
        
        self.security_events.append(security_event)
        
        # Keep only recent events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]

# Test Security & Monitoring
def test_security_monitoring():
    """Test security and monitoring functionality."""
    print("🧪 Testing Security & Monitoring System...")
    
    config = VideoGPTConfig()
    logger = VideoGPTLogger(config)
    metrics = VideoGPTMetrics()
    
    # Test security manager
    security = SecurityManager(config, logger)
    
    # Test input sanitization
    test_inputs = [
        "What is machine learning?",  # Safe input
        "Tell me about AI <script>alert('xss')</script>",  # Dangerous input
        "My email is john@example.com and phone is 555-123-4567",  # PII input
        "A" * 3000  # Too long input
    ]
    
    print("Testing input sanitization...")
    for i, test_input in enumerate(test_inputs, 1):
        result = security.sanitize_input(test_input, f"user_{i}")
        print(f"Input {i}: {'✅ Safe' if result['is_safe'] else '⚠️ Issues found'}")
        if result.get('issues_found'):
            print(f"  Issues: {result['issues_found']}")
    
    # Test URL validation
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Valid
        "https://youtu.be/dQw4w9WgXcQ",  # Valid short
        "https://malicious-site.com/video",  # Invalid
    ]
    
    print("\nTesting URL validation...")
    for url in test_urls:
        result = security.validate_youtube_url(url)
        print(f"URL: {'✅ Valid' if result['is_valid'] else '❌ Invalid'}")
    
    # Test rate limiter
    rate_limiter = RateLimiter(max_calls=3, time_window=60)
    
    print("\nTesting rate limiter...")
    for i in range(5):
        result = rate_limiter.can_make_call("test_user")
        status = "✅ Allowed" if result['can_call'] else "❌ Rate limited"
        print(f"Call {i+1}: {status} ({result['current_calls']}/{result['max_calls']})")
    
    # Test system monitor
    monitor = SystemMonitor(config, logger, metrics)
    
    print("\nTesting system health check...")
    health = monitor.check_system_health()
    print(f"Overall Status: {health['overall_status']}")
    
    for component, status in health['components'].items():
        print(f"  {component}: {status['status']}")
    
    print("✅ Security & Monitoring test completed!")

if __name__ == "__main__":
    test_security_monitoring()