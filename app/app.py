import os
import sys
import time
import torch
import subprocess
import mlflow
from pprint import pprint
from datetime import datetime
from scipy.io.wavfile import write as write_wav
from optimum.bettertransformer import BetterTransformer
from transformers import AutoProcessor, BarkModel, BarkConfig, BarkProcessor

# TRANSFORMERS_OFFLINE=1
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

def init():
    
    global model
    global device
    
    model_pretreined = parent_dir + "/model"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] CUDA version: {torch.cuda.get_device_name()}")
        print(f"[INFO] ID of current CUDA device: {torch.cuda.current_device()}")
        print("[INFO] nvidia-smi output:")
        pprint(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode(
                "utf-8"
            )
        )
    else:
        device = torch.device("cpu")
        print(
            "[WARN] CUDA acceleration is not available. This model takes hours to run on medium size data."
        )

    # config = BarkConfig.from_json_file(model_pretreined + "/config.json")
    
    # Load the model
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_pretreined)
    model = BarkModel.from_pretrained(pretrained_model_name_or_path=model_pretreined)
    # ,
    #     config=config,
    #     local_files_only=True,
    #     torch_dtype=torch.float16
    # ).to(device)
        
    
    model.to(device)

    voice_preset = "v2/pt_speaker_6"

    inputs = processor("O céu por cima do porto tinha a cor de uma TV que saiu do ar.", voice_preset=voice_preset)
    
    for key in inputs.keys():
        inputs[key] = inputs[key].to(device)

    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()

    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{parent_dir}/outputs/bark_out_{timestamp_str}.wav"

    sample_rate = model.generation_config.sample_rate
    write_wav(filename, rate=sample_rate, data=audio_array)

    mlflow.log_param("device", device)
    mlflow.log_param("model", type(model).__name__)

# Start the clock
start_time = time.time()

init()

# Stop the clock and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"The process took {elapsed_time:.2f} seconds.")