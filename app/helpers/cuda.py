import torch

# Verifica se você possui uma GPU disponível
if torch.cuda.is_available():
    # Obtém o dispositivo GPU padrão
    device = torch.device("cuda")
    
    # Obtém a quantidade total de memória da GPU
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # Em gigabytes (GB)
    
    # Obtém a quantidade de memória atualmente em uso
    currently_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Em gigabytes (GB)
    
    # Obtém a quantidade de memória disponível
    available_memory = total_memory - currently_allocated
    
    print(f"Total de memória GPU: {total_memory:.2f} GB")
    print(f"Memória GPU atualmente alocada: {currently_allocated:.2f} GB")
    print(f"Memória GPU disponível: {available_memory:.2f} GB")
else:
    print("Nenhuma GPU disponível.")