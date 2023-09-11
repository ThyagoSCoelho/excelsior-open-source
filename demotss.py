from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import time
import torch
import torchaudio
from pydub import AudioSegment

# Other languages are support like Japanese, Chinese Mandarin and many others....
lang = 'Portuguese'

# Currently utilzing the Pre-trained VITs LJSpeech model, but can easily replace...
tag = 'kan-bayashi/ljspeech_tacotron2-pt'

# Replace the tag with any of the following acoustic models below
# kan-bayashi/ljspeech_tacotron2, kan-bayashi/ljspeech_fastspeech ,
# kan-bayashi/ljspeech_fastspeech2, kan-bayashi/ljspeech_conformer_fastspeech2, 
# kan-bayashi/ljspeech_vits

# Since VITS doesnt require a vocoder we arent utilizing one.
# That doesnt stop you from using one though and experimenting.
vocoder_tag = "parallel_wavegan/ljspeech_parallel_wavegan.v1"

# Replace the tag with any of the following vocoders 
# parallel_wavegan/ljspeech_parallel_wavegan.v1, parallel_wavegan/ljspeech_full_band_melgan.v2
# parallel_wavegan/ljspeech_multi_band_melgan.v2, parallel_wavegan/ljspeech_hifigan.v1 ,
# parallel_wavegan/ljspeech_style_melgan.v1

text2speech = Text2Speech.from_pretrained(
    model_tag='kan-bayashi/ljspeech_tacotron2',
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

# speech synthesis
input_text = 'O céu por cima do porto tinha a cor de uma TV que saiu do ar. — Não é que eu queira — Case ouviu enquanto abria caminho pela multidão que estava na porta do Chat. — Mas é como se o meu corpo tivesse criado, por si mesmo, esta enorme dependência da droga.'

with torch.no_grad():
    start = time.time()
    wav = text2speech(input_text)["wav"]
rtf = (time.time() - start) / (len(wav) / text2speech.fs)
print(f"RTF = {rtf:5f}")

wav = wav.unsqueeze(0)

# Salve o áudio em um arquivo WAV
torchaudio.save("output.wav", wav.cpu(), text2speech.fs)

# # Converta o arquivo WAV para MP3
sound = AudioSegment.from_wav("output.wav")
sound.export("output.mp3", format="mp3")