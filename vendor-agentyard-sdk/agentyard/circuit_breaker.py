"""Circuit breaker pattern for protecting external service calls.

Prevents cascading failures by tracking error rates and temporarily
rejecting calls to unhealthy services. Transitions through three states:

- CLOSED: Normal operation, calls pass through.
- OPEN: Too many failures, calls are rejected immediately.
- HALF_OPEN: Recovery probe — a limited number of calls are allowed
  to test whether the service has recovered.

Usage:
    breaker = CircuitBreaker("payment-service", failure_threshold=3)

    try:
        result = await breaker.call(http_client.post, url, json=payload)
    except CircuitOpenError as e:
        # Service is down, use fallback
        print(f"Circuit open, retry after {e.retry_after:.1f}s")
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("agentyard.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when the circuit is open and a call is rejected."""

    def __init__(self, name: str, retry_after: float) -> None:
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit '{name}' is open. Retry after {retry_after:.1f}s."
        )


class CircuitBreaker:
    """Circuit breaker for protecting external service calls.

    Args:
        name: Identifier for the protected service.
        failure_threshold: Consecutive failures before opening the circuit.
        recovery_timeout: Seconds to wait in OPEN state before probing.
        half_open_max_calls: Max concurrent probe calls in HALF_OPEN state.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._failure_count: int = 0
        self._success_count: int = 0
        self._half_open_calls: int = 0
        self._half_open_successes: int = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = 0.0
        self._state = CircuitState.CLOSED
        self._total_calls: int = 0
        self._total_failures: int = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state, accounting for recovery timeout expiry."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute func through the circuit breaker.

        Raises CircuitOpenError if the circuit is open.
        Tracks success/failure to manage state transitions.
        """
        async with self._lock:
            current_state = self.state
            self._total_calls += 1

            if current_state == CircuitState.OPEN:
                retry_after = self._recovery_timeout - (
                    time.monotonic() - self._opened_at
                )
                raise CircuitOpenError(self.name, max(0.0, retry_after))

            if (
                current_state == CircuitState.HALF_OPEN
                and self._half_open_calls >= self._half_open_max_calls
            ):
                raise CircuitOpenError(self.name, self._recovery_timeout)

            if current_state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                self.record_success()
            return result
        except Exception:
            async with self._lock:
                self.record_failure()
            raise

    def record_success(self) -> None:
        """Record a successful call. Resets to CLOSED after consecutive successes in HALF_OPEN."""
        self._failure_count = 0
        self._success_count += 1
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self._half_open_max_calls:
                self.reset()

    def record_failure(self) -> None:
        """Record a failed call. Opens the circuit if threshold is exceeded."""
        self._failure_count += 1
        self._total_failures += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def reset(self) -> None:
        """Reset the circuit to its initial CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
        self._half_open_successes = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Return circuit state, failure count, and timing info."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "last_failure_time": self._last_failure_time,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout": self._recovery_timeout,
        }
