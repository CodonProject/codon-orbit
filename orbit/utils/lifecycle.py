import atexit
import signal
import sys
from typing import Callable, Any

class ExitManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExitManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._callbacks = []
        self._has_executed = False
        self._setup_handlers()
        self._initialized = True

    def _setup_handlers(self):
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError: pass
            
        atexit.register(self._execute_callbacks)

    def _signal_handler(self, signum: int, frame):
        sig_name = signal.Signals(signum).name
        
        sys.exit(0)

    def _execute_callbacks(self):
        if self._has_executed: return
            
        self._has_executed = True
        if not self._callbacks: return
        
        for func, args, kwargs in reversed(self._callbacks):
            try:
                func(*args, **kwargs)
            except Exception as e:
                pass

    def register(self, func: Callable, *args: Any, **kwargs: Any):
        self._callbacks.append((func, args, kwargs))
        return func


exit_manager = ExitManager()

register_exit = exit_manager.register