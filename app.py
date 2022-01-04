from flask import Flask
import base64
from flask_cors import CORS

import logging

import psycopg2.extras

from flask_sqlalchemy import SQLAlchemy

# Flask App
app = Flask(__name__)

cors = CORS(app)

#logging.basicConfig(filename="proof.log",level=logging.DEBUG,format='PROOF %(asctime)s  %(name)s  %(levelname)s: %(message)s')

STATUS = {
    "ACTIVE" : "A",
    "DEACTIVE" : "D",
    "DELETE" : "C"
}

# Postgres database connection
#db_url = 'postgresql://postgres:root@localhost:5432/proof_verification'
db_url = 'postgresql://postgres:postgres@10.10.106.95:5432/proof_verification'
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_POOL_SIZE"] = 10
db = SQLAlchemy(app)


#"postgres://postgres:root@localhost:5432/master"