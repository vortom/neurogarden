"""Example brains."""

from .base import Brain, brain_seed, run_episode
from .random_brain import RandomBrain
from .scripted import ScriptedBrain

BRAINS = {"random": RandomBrain, "scripted": ScriptedBrain}

__all__ = ["BRAINS", "Brain", "RandomBrain", "ScriptedBrain", "brain_seed", "run_episode"]
