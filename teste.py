import torch
import torchaudio
from transformers import LibriTTSForConditionalGeneration, AutoTokenizer

# Coleta o texto que deseja converter em áudio.
text = "Olá, mundo!"

# Cria uma instância do modelo.
model = LibriTTSForConditionalGeneration.from_pretrained("viniciuslima/librivox-pt-br")
tokenizer = AutoTokenizer.from_pretrained("viniciuslima/librivox-pt-br")

# Converte o texto em um vetor de tokens.
tokens = tokenizer(text)

# Gera um arquivo de áudio.
output = model.generate(tokens)

# Salva o arquivo de áudio.
torchaudio.save(output, "output.wav")