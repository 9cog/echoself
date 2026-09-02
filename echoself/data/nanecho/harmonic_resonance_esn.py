"""Frequency-domain reservoir: phases and amplitudes, not random recurrence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class OscillatorStateError(ValueError):
    """Harmonic reservoir requires non-empty oscillator state."""


@dataclass(frozen=True)
class OscillatorState:
    phases: tuple[float, ...]
    amplitudes: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.phases or not self.amplitudes:
            raise OscillatorStateError(
                "harmonic reservoir requires non-empty phases and amplitudes"
            )
        if len(self.phases) != len(self.amplitudes):
            raise OscillatorStateError("phases and amplitudes must be the same length")
        if any(amp < 0.0 for amp in self.amplitudes):
            raise OscillatorStateError("amplitudes must be non-negative")


class HarmonicResonanceESN:
    """Step oscillators in the frequency domain (phase rotation + amplitude leak)."""

    def __init__(
        self,
        state: OscillatorState,
        *,
        leak: float = 0.05,
        frequencies: Sequence[float] | None = None,
    ) -> None:
        self._state = state
        if not 0.0 <= leak < 1.0:
            raise OscillatorStateError("leak must be in [0, 1)")
        self._leak = leak
        n = len(state.phases)
        if frequencies is None:
            self._frequencies = tuple(1.0 + 0.25 * i for i in range(n))
        else:
            if len(frequencies) != n:
                raise OscillatorStateError("frequencies must match oscillator count")
            self._frequencies = tuple(float(f) for f in frequencies)

    @property
    def state(self) -> OscillatorState:
        return self._state

    def step(self, drive: Sequence[float] | None = None) -> OscillatorState:
        if drive is not None and len(drive) != len(self._state.phases):
            raise OscillatorStateError("drive must match oscillator count")
        phases: list[float] = []
        amplitudes: list[float] = []
        for i, (phase, amp, freq) in enumerate(
            zip(self._state.phases, self._state.amplitudes, self._frequencies)
        ):
            injection = 0.0 if drive is None else float(drive[i])
            next_phase = (phase + freq + 0.15 * injection) % (2.0 * math.pi)
            next_amp = (1.0 - self._leak) * amp + self._leak * abs(injection)
            phases.append(next_phase)
            amplitudes.append(next_amp)
        self._state = OscillatorState(tuple(phases), tuple(amplitudes))
        return self._state

    def readout(self) -> float:
        return sum(
            amp * math.cos(phase)
            for phase, amp in zip(self._state.phases, self._state.amplitudes)
        )


def persona_oscillators(weights: Sequence[float]) -> OscillatorState:
    if not weights:
        raise OscillatorStateError("persona weights cannot be empty")
    phases = tuple((i * math.pi / 4.0) % (2.0 * math.pi) for i in range(len(weights)))
    amplitudes = tuple(max(0.0, float(w)) for w in weights)
    return OscillatorState(phases, amplitudes)
