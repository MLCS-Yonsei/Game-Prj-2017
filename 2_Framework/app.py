import sys
sys.path.insert(0, './routes')

from blockchain import Blockchain

from uuid import uuid4

from flask import Flask, jsonify, request
from overtake import checkOvertake

from urllib.parse import urlparse

# Generate a globally unique address for this node
node_identifier = str(uuid4()).replace('-', '')

# Instantiate the Blockchain
blockchain = Blockchain()

# Instantiate our Node
app = Flask(__name__)
app.register_blueprint(checkOvertake)

@app.route('/status', methods=['GET'])
def status():
    return True
    #return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)