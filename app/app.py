# app/app.py
import pyttsx3

engine = pyttsx3.init()

rate = engine.getProperty('rate')  # Velocidade de fala (padrão é 200)
engine.setProperty('rate', rate - 50)  # Reduza a velocidade em 50 unidades

# Defina uma entonação diferente usando propriedades de pitch
engine.setProperty('pitch', 150)  # Valor padrão é 50, aumente para 150 para entonação mais alta

voices = engine.getProperty('voices')
# for voice in voices:
#     print("Voice:", voice.name)
#     print(" - ID:", voice.id)
    
# Escolha um ID de voz com base na lista gerada no passo anterior
selected_voice_id = "brazil"
engine.setProperty('voice', selected_voice_id)

texto = "O Céu sobre o porto tinha cor de televisão num canal fora do ar. Considerada a obra precursora do movimento cyberpunk e um clássico da ficção científica moderna, Neuromancer conta a história de Case, um cowboy do ciberespaço e hacker da matrix. Como punição por tentar enganar os patrões, seu sistema nervoso foi contaminado por uma toxina que o impede de entrar no mundo virtual."
# engine.say(texto)
engine.save_to_file(texto, "neuromancer_ia_fala_saida.mp3")
engine.runAndWait()