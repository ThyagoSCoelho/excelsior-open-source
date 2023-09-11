import torchaudio
from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
import torch
import re
import sys
import torch
from espnet2.bin.tts_inference import Text2Speech
from espnet2.bin.tts_inference import str_or_none


tag = 'models--alefiury--wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-gain-normalization-sna'

# Replace the tag with any of the following acoustic models below
# kan-bayashi/ljspeech_tacotron2, kan-bayashi/ljspeech_fastspeech ,
# kan-bayashi/ljspeech_fastspeech2, kan-bayashi/ljspeech_conformer_fastspeech2, 
# kan-bayashi/ljspeech_vits

# Since VITS doesnt require a vocoder we arent utilizing one.
# That doesnt stop you from using one though and experimenting.
vocoder_tag = "none"

# Replace the tag with any of the following vocoders 
# parallel_wavegan/ljspeech_parallel_wavegan.v1, parallel_wavegan/ljspeech_full_band_melgan.v2
# parallel_wavegan/ljspeech_multi_band_melgan.v2, parallel_wavegan/ljspeech_hifigan.v1 ,
# parallel_wavegan/ljspeech_style_melgan.v1


chars_to_ignore_regex = '[\,\?\.\!\;\:\"]'  # noqa: W605
wer = load_metric("wer")
device = "cuda"

model_name = 'lgris/wav2vec2-large-xlsr-open-brazilian-portuguese'
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_name)

def map_to_pred(batch):
    features = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0], padding=True, return_tensors="pt")
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    pred_ids = torch.argmax(logits, dim=-1)
    batch["predicted"] = processor.batch_decode(pred_ids)
    batch["predicted"] = [pred.lower() for pred in batch["predicted"]]
    batch["target"] = batch["sentence"]
    return batch


# Adicione o texto de entrada que você deseja reconhecer
input_text = "Insira o texto que você deseja reconhecer aqui."

# Crie um dicionário simulando um lote (batch)
batch = {
    "speech": [input_text],
    "sampling_rate": [16000],  # Taxa de amostragem do áudio (ajuste conforme necessário)
    "sentence": [""],  # A sentença não é usada aqui, apenas para simular
}

# Execute a função map_to_pred para obter o resultado da inferência
result_batch = map_to_pred(batch)

text2speech = Text2Speech.from_pretrained(
    model_tag=str_or_none(tag),
    vocoder_tag=str_or_none(vocoder_tag),
    device="cuda",
    
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1.0,
    
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

# Texto reconhecido obtido após a inferência
recognized_text = result_batch["predicted"][0]

# Sintetize o áudio a partir do texto reconhecido
with torch.no_grad():
    wav = text2speech(recognized_text)["wav"]

# Salve o áudio em um arquivo
output_audio_file = "output.wav"
torch.save(wav.cpu(), output_audio_file)
print("Áudio salvo em:", output_audio_file)