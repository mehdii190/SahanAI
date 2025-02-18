
from hazm import Normalizer
from cleantext import clean

class TextPreprocessor:
    def __init__(self):
        self.normalizer = Normalizer()
    
    def clean(self, text: str) -> str:
        text = self.normalizer.normalize(text)
        return clean(
            text,
            fix_unicode=True,
            to_ascii=False,
            no_emoji=True,
            no_urls=True
        )
        
        
        
        
        