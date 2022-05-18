from unicodedata import name
from urllib import response
from flask import Flask
from flask_cors import CORS
import json
import iris

# init server
app = Flask(__name__)


@app.route('/test')
def test():
    return "test"


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080, debug=True)