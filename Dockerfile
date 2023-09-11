# Use a imagem base do Python 3.8
FROM python:3.8

# Atualize o pip
RUN pip install --upgrade pip

# Instale o pacote espeak e as dependências necessárias
RUN apt-get update && apt-get install -y espeak alsa-utils ffmpeg

# Crie e defina o diretório de trabalho dentro do container
WORKDIR /app

# Copie os arquivos de requisitos primeiro para aproveitar o cache de camadas
COPY requirements.txt .

# Instale as dependências do projeto
RUN pip install -r requirements.txt

# Copie o restante dos arquivos do diretório atual para o diretório de trabalho no container
COPY . .

# Expõe a porta 80 para o tráfego externo (se o aplicativo realmente precisar disso)
EXPOSE 80

# Comando para executar o aplicativo quando o container for iniciado
CMD ["python", "./app/app.py"]