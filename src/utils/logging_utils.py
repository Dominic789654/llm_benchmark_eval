"""
Logging utilities for benchmark evaluation.
"""
import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "./logs"
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file name (if None, uses timestamp)
        log_dir: Directory to save log files
    
    Returns:
        Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"benchmark_eval_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("benchmark_eval")
    logger.info(f"Logging initialized. Log file: {log_path}")
    
    return logger


def get_logger(name: str = "benchmark_eval") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize logging context.
        
        Args:
            log_level: Temporary logging level
            log_file: Temporary log file
        """
        self.log_level = log_level
        self.log_file = log_file
        self.original_level = None
        self.original_handlers = []
    
    def __enter__(self):
        """Enter the logging context."""
        # Save original configuration
        root_logger = logging.getLogger()
        self.original_level = root_logger.level
        self.original_handlers = root_logger.handlers.copy()
        
        # Apply temporary configuration
        if self.log_file:
            setup_logging(self.log_level, self.log_file)
        else:
            root_logger.setLevel(getattr(logging, self.log_level.upper()))
        
        return get_logger()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the logging context and restore original configuration."""
        root_logger = logging.getLogger()
        
        # Remove temporary handlers
        for handler in root_logger.handlers[:]:
            if handler not in self.original_handlers:
                handler.close()
                root_logger.removeHandler(handler)
        
        # Restore original configuration
        root_logger.setLevel(self.original_level)
        for handler in self.original_handlers:
            if handler not in root_logger.handlers:
                root_logger.addHandler(handler)
