from app.logger import logger

class ModelsCache:
    """Cache loaded models to avoid reloading."""
    def __init__(self):
        self.transcription_model = None
        self.diarization_model = None
        self.align_model = None
        self.align_metadata = None

models_cache = ModelsCache()