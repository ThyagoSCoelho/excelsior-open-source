import os
import string
import sys
import mlflow
from datetime import datetime
from scipy.io.wavfile import write as write_wav

from helpers.device import Device
from service.tss_generator import TTSGenerator
from ai_startup.initializer_excelsior import InitializerExcelsior

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

class Excelsior():
    
    def __init__(self):
        self.device_worker = Device()
        self.device_worker.info_device()
        self.device_worker.status()
        self.device = self.device_worker.torch_device()
        
    def initialize(self):
        model_pretreined = parent_dir + "/model"
        excelsior = InitializerExcelsior(model_path=model_pretreined)
        
        processor, model = excelsior.initialize()
        
        self.processor = processor
        self.model = model
    
    def process(self, uuid: string, text: string):        
        generator = TTSGenerator(
            model=self.model, 
            processor=self.processor,
            device=self.device,
            text=text
        )
        audio_array, sample_rate = generator.executar()
        return self.save(audio_array, sample_rate, uuid)
        
    def save(self, audio_array, sample_rate, uuid):
        try:
            filename = f"{parent_dir}/outputs/{uuid}.wav"
            write_wav(filename, rate=sample_rate, data=audio_array)
            return True
        except Exception as e:
            print(f"[ERROR] Unable to save the model on the machine. Error: {e}")
            return False
    # def log(self):
    #     mlflow.log_param("device", device)
    #     mlflow.log_param("model", type(model).__name__)        
