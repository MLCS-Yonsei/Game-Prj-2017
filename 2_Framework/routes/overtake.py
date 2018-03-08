import sys
sys.path.insert(0, '../controllers')

from flask import Flask, jsonify, request, Blueprint

get_overtake_blueprint = Blueprint('get_overtake', __name__)
@get_overtake_blueprint.route('/api/overtake', methods=['POST'])
def new_transaction():
    # values = request.get_json()

    # # Check that the required fields are in the POST'ed data
    # required = ['sender', 'recipient', 'amount']
    # if not all(k in values for k in required):
    #     return 'Missing values', 400

    # # Create a new Transaction
    # index = blockchain.new_transaction(values['sender'], values['recipient'], values['amount'])

    # response = {'message': f'Transaction will be added to Block {index}'}
    # return jsonify(response), 201
    return True
