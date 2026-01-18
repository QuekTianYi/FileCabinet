from .base import FileHandler

class ImageHandler(FileHandler):
    def can_handle(self, path):
        return super().can_handle(path)
    
    def extract(self, path):
        return super().extract(path)