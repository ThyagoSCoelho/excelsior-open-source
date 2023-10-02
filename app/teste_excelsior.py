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
excelsior.process("O céu por cima do porto tinha a cor de uma TV que saiu do ar.")

# Stop the clock and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"The process took {elapsed_time:.2f} seconds.")
