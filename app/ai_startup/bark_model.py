import os
import sys
import torch
import subprocess
from pprint import pprint
from transformers import BarkModel, BarkProcessor

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def init():
    global model
    global model_name
    global output_dir
    global device
    
    cuda_available = torch.cuda.is_available()
    print(cuda_available)
    
    device = "cuda" if cuda_available else "cpu"

    if cuda_available:
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] ID of current CUDA device: {torch.cuda.current_device()}")
        print("[INFO] nvidia-smi output:")
        pprint(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode(
                "utf-8"
            )
        )
    else:
        print(
            "[WARN] CUDA acceleration is not available. This model takes hours to run on medium size data."
        )
    
    # Load the model
    model_name = "suno/bark"
    output_dir = parent_dir + "/model"
    
    try:
        model = BarkModel.from_pretrained(model_name)
        
        model.save_pretrained(output_dir)
        
        processor = BarkProcessor.from_pretrained("suno/bark")
        processor.save_pretrained(
            save_directory=output_dir,
            speaker_embeddings_dict_path=output_dir + "/speaker_embeddings_path.json",
            speaker_embeddings_directory=output_dir + "/speaker_embeddings",
            push_to_hub=False,
        )
        
        print(
            f"[SAVED] Modelo foi salvo em: {output_dir}"
        )
    except Exception as e:
        print(
            f"[ERROR] Unable to save the model on the machine. Error: {e}"
        )        

init()