"""Example brains."""

from .base import Brain, brain_seed, run_episode
from .connectome import ConnectomeBrain, ConnectomeGenome, RandomGraphBrain
from .evolved import EvolvedBrain, Genome
from .random_brain import RandomBrain
from .scripted import ScriptedBrain

BRAINS = {
    "random": RandomBrain,
    "scripted": ScriptedBrain,
    "evolved": EvolvedBrain,
    "connectome": ConnectomeBrain,  # needs the cached MaleCNS graph: `neurogarden connectome …`
    "connectome-random": RandomGraphBrain,  # its control, on the row-shuffled graph
}
# The brains whose weights come from a file.
WEIGHTED = {
    "evolved": EvolvedBrain,
    "connectome": ConnectomeBrain,
    "connectome-random": RandomGraphBrain,
}

__all__ = [
    "BRAINS",
    "WEIGHTED",
    "Brain",
    "ConnectomeBrain",
    "ConnectomeGenome",
    "EvolvedBrain",
    "RandomGraphBrain",
    "Genome",
    "RandomBrain",
    "ScriptedBrain",
    "brain_seed",
    "run_episode",
]
