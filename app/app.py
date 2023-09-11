import torch
import subprocess
import mlflow
from pprint import pprint
from optimum.bettertransformer import BetterTransformer
from transformers import AutoProcessor, AutoModel, AutoTokenizer, BartForConditionalGeneration, BarkModel
from optimum.bettertransformer import BetterTransformer
import torch
import scipy

def init():
    global model
    global device

    cuda_available = torch.cuda.is_available()
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
    try:
        model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
        model = BetterTransformer.transform(model, keep_original_model=False).to(device)
        model.enable_cpu_offload()
    except Exception as e:
        print(
            f"[ERROR] Error happened when loading the model on GPU or the default device. Error: {e}"
        )
        print("[INFO] Trying on CPU.")
        model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16)
        model = BetterTransformer.transform(model, keep_original_model=False)
        device = "cpu"

    processor = AutoProcessor.from_pretrained("suno/bark-small", force_download=True)
    model = AutoModel.from_pretrained("suno/bark-small", force_download=True)

    inputs = processor(
        text=["O céu por cima do porto tinha a cor de uma TV que saiu do ar."],
        return_tensors="pt",
    )

    speech_values = model.generate(**inputs, do_sample=True)
    sampling_rate = model.generation_config.sample_rate

    scipy.io.wavfile.write("outputs/teste2.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())

    mlflow.log_param("device", device)
    mlflow.log_param("model", type(model).__name__)

init()