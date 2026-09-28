"""Example brains."""

from .base import Brain, brain_seed, run_episode
from .evolved import EvolvedBrain, Genome
from .random_brain import RandomBrain
from .scripted import ScriptedBrain

BRAINS = {"random": RandomBrain, "scripted": ScriptedBrain, "evolved": EvolvedBrain}

__all__ = [
    "BRAINS",
    "Brain",
    "EvolvedBrain",
    "Genome",
    "RandomBrain",
    "ScriptedBrain",
    "brain_seed",
    "run_episode",
]
