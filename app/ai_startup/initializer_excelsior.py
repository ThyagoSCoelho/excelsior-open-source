import os
import string
from transformers import AutoProcessor, BarkModel

os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

class InitializerExcelsior:
    
    def __init__(self, model_path=string):
        self.model_path = model_path
        print(self.model_path)
    
    def initialize(self):
        # config = BarkConfig.from_json_file(model_pretreined + "/config.json")
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=self.model_path)
        self.model = BarkModel.from_pretrained(pretrained_model_name_or_path=self.model_path)
        #     config=config,
        #     local_files_only=True,
        #     
        # ).to(device)
        return self.processor, self.model