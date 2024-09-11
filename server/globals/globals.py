from enum import Enum


class ServerStatus(Enum):
    INITIALIZING = "initializing"
    DOWNLOADING_MODEL = "downloading_model"
    MISSING_MODEL_FILES = "missing model files"
    ERROR = "error"
    READY = "ready"


_current_status = ServerStatus.INITIALIZING


def get_server_status():
    return _current_status


def set_server_status(status: ServerStatus):
    global _current_status
    _current_status = status
