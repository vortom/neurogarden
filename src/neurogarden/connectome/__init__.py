"""Real wiring for small brains: the MaleCNS connectome as a graph a fly can think with.

The data says who connects to whom, with how many synapses and which transmitter. The
dynamics, the signs, the sensory encoding and the readout are modelling choices made
here — see docs/superpowers/specs/2026-10-03-connectome-fly-design.md.
"""

from .model import Encoding, Projection, RateNetwork, Wiring, make_network

__all__ = ["Encoding", "Projection", "RateNetwork", "Wiring", "make_network"]
