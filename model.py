import datetime
from sqlalchemy import and_
#from sqlalchemy.sql.sqltypes import BLOB
from app import app,  db, STATUS

class Config(db.Model):
    """Config Table"""
    __tablename__ = "tb_config"
    config_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    id_type = db.Column(db.String(1000), nullable=False)
    id_version = db.Column(db.Integer, nullable=False)
    params = db.Column(db.String(5000), nullable=False)
    params_dist = db.Column(db.Float, nullable=False, default=0.0)
    params_ratio = db.Column(db.Float, nullable=False, default=0.0)
    image_breath = db.Column(db.Float, nullable=False, default=0.0)
    image_length = db.Column(db.Float, nullable=False, default=0.0)
    image_logo = db.Column(db.Float, nullable=False, default=0.0)
    key_breath = db.Column(db.String(1000), nullable=False, default="Photo Breath")
    key_length = db.Column(db.String(1000), nullable=False, default="Photo Length")
    key_logo = db.Column(db.String(1000), nullable=False, default="Photo Logo")
    status = db.Column(db.String(1), nullable=False, default=STATUS["ACTIVE"])
    created_date = db.Column(db.DateTime, nullable=True)
    updated_date = db.Column(db.DateTime, nullable=True)

    def __init__(self,id_type,id_version,params,params_dist,params_ratio,image_breath,image_length, image_logo, key_breath= "Photo Breath", key_length= "Photo Length", key_logo= "Photo Logo"):
        self.id_type = id_type
        self.id_version = id_version
        self.params = params
        self.params_dist = params_dist
        self.params_ratio = params_ratio
        self.image_breath = image_breath
        self.image_length = image_length
        self.image_logo = image_logo
        self.key_breath = key_breath
        self.key_length = key_length
        self.key_logo = key_logo
        self.created_date = datetime.datetime.now()
        
class Img(db.Model):
    __tablename__ = "tb_image"
    id = db.Column(db.Integer, primary_key=True)
    id_type = db.Column(db.String(1000), nullable=False)
    img = db.Column(db.String(64), nullable=False, default="") 
    status = db.Column(db.String(1), nullable=False, default=STATUS["ACTIVE"])
    created_date = db.Column(db.DateTime, nullable=True)
    updated_date = db.Column(db.DateTime, nullable=True)
    
    
    def __init__(self,id_type,img):
        self.id_type = id_type
        self.img = img
        self.created_date = datetime.datetime.now()
    
    