import string
from datetime import datetime
from transformers import BarkModel, AutoProcessor
from scipy.io.wavfile import write as write_wav

class TTSGenerator:
    def __init__(self, model: BarkModel, processor: AutoProcessor, device, text: string):
        self.voice_preset = "v2/pt_speaker_6"
        self.text = text
        self.model = model
        self.processor = processor
        self.device = device

    def executar(self):        
        self.model.to(self.device)

        inputs = self.processor(
            self.text,
            voice_preset=self.voice_preset
        )
    
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)

        audio_array = self.model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        
        sample_rate = self.model.generation_config.sample_rate
        
        return audio_array, sample_rate