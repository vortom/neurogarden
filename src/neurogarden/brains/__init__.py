"""Example brains."""

from .base import Brain, run_episode
from .random_brain import RandomBrain
from .scripted import ScriptedBrain

BRAINS = {"random": RandomBrain, "scripted": ScriptedBrain}

__all__ = ["BRAINS", "Brain", "RandomBrain", "ScriptedBrain", "run_episode"]
