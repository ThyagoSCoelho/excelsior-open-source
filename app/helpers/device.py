import torch
import subprocess

class Device:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            self.currently_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            self.available_memory = self.total_memory - self.currently_allocated
            self.cuda = True
        else:
            print("Nenhuma GPU disponível.")
            self.device = torch.device("cpu")
            
    def torch_device(self):
        return self.device
    
    def status(self):
        if self.cuda == False: 
            print(f"[WARN] CUDA acceleration is not available. This model takes hours to run on medium size data.")
        else:
            print(f"[INFO] Total de memória GPU: {self.total_memory:.2f} GB")
            print(f"[INFO] Memória GPU atualmente alocada: {self.currently_allocated:.2f} GB")
            print(f"[INFO] Memória GPU disponível: {self.available_memory:.2f} GB")
    
    def info_device(self):
        if self.cuda == False: 
            print(f"[WARN] CUDA acceleration is not available. This model takes hours to run on medium size data.")
        else:
            print(f"[INFO] ID of current CUDA device: {torch.cuda.current_device()}")
            print("[INFO] nvidia-smi output:")
            print(
                subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode(
                    "utf-8"
                )
            )