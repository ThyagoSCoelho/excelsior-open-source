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

    model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(device)
    model = BetterTransformer.transform(model, keep_original_model=False)
    model.enable_cpu_offload()

    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small")

    inputs = processor(
        text=["Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
        return_tensors="pt",
    )

    speech_values = model.generate(**inputs, do_sample=True)
    sampling_rate = model.generation_config.sample_rate

    scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())

    mlflow.log_param("device", device)
    mlflow.log_param("model", type(model).__name__)

init()