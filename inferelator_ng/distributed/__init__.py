# This is the required package init. Everything in this package must implement the abstract class AbstractController

from abc import abstractmethod


class AbstractController:
    client = None
    is_master = False
    chunk = 25

    @classmethod
    @abstractmethod
    def connect(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def map(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def sync_processes(cls, *args, **kwargs):
        raise NotImplementedError
