import os
import sys
from flask import Flask, request, jsonify

from excelsior import Excelsior

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

app = Flask(__name__)

excelsior = Excelsior()
excelsior.initialize()

def ia_excelsior(input_text):
    return excelsior.process(input_text)

@app.route('/ia', methods=['POST'])
def executar_ia():
    if request.method == 'POST':
        # Obtenha os dados da solicitação POST (por exemplo, texto de entrada)
        data = request.get_json()
        input_text = data.get('input_text')

        # Execute a função de IA
        resultado = ia_excelsior(input_text)
        print(resultado)

        # Retorne a resposta como JSON
        return jsonify({"resposta": resultado})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)