import os
import sys
import time

from excelsior import Excelsior

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


excelsior = Excelsior()

excelsior.initialize()

# Start the clock
start_time = time.time()

text = "O céu por cima do porto tinha a cor de uma TV que saiu do ar. — Não é que eu queira — Case ouviu enquanto abria caminho pela multidão que estava na porta do Chat. — Mas é como se o meu corpo tivesse criado, por si mesmo, esta enorme dependência da droga."
excelsior.process(f"{start_time.hex.__str__}", text)

# Stop the clock and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"The process took {elapsed_time:.2f} seconds.")
