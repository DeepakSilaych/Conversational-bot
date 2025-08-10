# stt_base.py
from typing import Callable
import abc

class BaseSTT(abc.ABC):
    def __init__(self, language: str = "en-IN"):
        self.language = language
        self.callback: Callable[[str, bool], None] = lambda text, final: None

    def set_callback(self, callback: Callable[[str, bool], None]):
        self.callback = callback

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def feed_audio(self, data: bytes):
        pass

    @abc.abstractmethod
    def finish(self):
        pass
