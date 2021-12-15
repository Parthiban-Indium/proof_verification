from app import app ,logging, STATUS,db
from model import *
from flask import jsonify, request, sessions
from sqlalchemy.sql.expression import distinct

import os

import json
import cv2
import pytesseract
import math
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\Indium Software\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

from pytesseract import Output

import datetime
import pandas as pd

IMAGE_PATH = "C:\\Users\\Indium Software\\Documents\\develop\\document_proof\\images"

@app.route("/")
def home():
    logging.info("start")
    return "Document Proof API Services"

@app.route("/id-proof", methods=['POST'])
def id_proof():
    """
    Verify User Proof Service 
    To Find Co-Ordinates of Text and Image Location 
    Return the Score of Matching Text and Image Co-ordinates
    
    """
    logging.info("id_proof : Start")
    resp_dict={"object":None,"status":False}
    
    image = request.files['image']
    input_json = request.form.get('id_type')
    
    if image:
        str_time =  datetime.datetime.now().strftime('%d%m%Y%H%M%S')
        image_file_name = str_time+".jpg"
        
        logging.info(os.path.isdir(IMAGE_PATH))
        if os.path.isdir(IMAGE_PATH)==False:
            os.mkdir(IMAGE_PATH)
            
        # image save
        image.save(os.path.join(IMAGE_PATH,image_file_name))
        image_read = cv2.imread(IMAGE_PATH+"/"+image_file_name)
        
        type_id = input_json
        
        config_obj =  Config.query.filter(Config.id_type ==input_json).all()
        
        if input_json:
            verified = verify(config_obj,image_read,type_id)
        else:
            resp_dict["msg"] = "Id Type Required"
            
        resp_dict["object"] = verified
        resp_dict["status"] = True
    else:
        resp_dict["object"] = "Image Required"
 
    os.remove(IMAGE_PATH+"/"+image_file_name)
        
    resp = jsonify(resp_dict)
    logging.debug("id_proof : end")
    return resp

def verify(config_obj,image,type_id):
    """
    Verify Text Method 
    To Find Co-Ordinates of Text Location 
    Calculate the Distance Between Two Texts and Ratio of Text
    
    """
    logging.info("verify : Start")
    try:
        verified_image = verify_image(image)
        
        breath = verified_image[0]
        logging.info(f"breath:{breath}")
        length = verified_image[1]
        logging.info(f"length:{length}")
        
        #OCR
        original_text_extract = pytesseract.image_to_data(image, output_type=Output.DICT)
        n_boxes = len(original_text_extract['level'])
        for config in config_obj:
            id_type=config.id_type
        
        # Find unique id version    
        version_all = db.session.query(Config.id_version).filter(Config.id_type==id_type).distinct(Config.id_version).all()
        # print(f"version_all:{version_all}")
        
        total_list = []
        update_mean_dict ={}
        result_list = []
        for version in version_all:
            id_version = version[0]
            dict_params ={}
            list_of_params = []
            # Select a params
            version_numbers = Config.query.filter(and_(Config.id_type==id_type),(Config.id_version==id_version)).all()
            
            for version_number in version_numbers:
                params_text = version_number.params
                list_of_params.append(params_text)
            
            # Find coordinates  
            for i in range(n_boxes):
                if(original_text_extract['text'][i] != ""):
                    list_of_original_params = original_text_extract['text'][i]
                    for params in list_of_params:
                        if params == list_of_original_params:
                            (x, y, w, h) = (original_text_extract['left'][i], original_text_extract['top'][i], original_text_extract['width'][i], original_text_extract['height'][i])
                            value = (x, y, w, h)
                            dict_params[params] = value
                            
            # Find distance and ratio        
            distance_value = []
            ratio_value = []
            
            length_dict_params =len(dict_params) 
            if length_dict_params>1:
                params_key = list(dict_params.keys())
                params_value = list(dict_params.values())
                for i in range(0,length_dict_params,2):
                    x1 = params_value[i][0]
                    y1 = params_value[i][1]
                    x2 = params_value[i+1][0]
                    y2 = params_value[i+1][1]
                    r1 = params_value[i][2]
                    r2 = params_value[i][3]
                    r3 = params_value[i+1][2]
                    r4 = params_value[i+1][3]
                    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    ratio_1 = r1/r2
                    ratio_2 = r3/r4
                    dist = round(dist,4)  
                      
                    distance_value.append({'key': params_key[i], 'value':dist })
                    distance_value.append({'key': params_key[i+1], 'value':dist })
                    
                    ratio_value.append({'key': params_key[i], 'value':round(ratio_1,4)})
                    ratio_value.append({'key': params_key[i+1], 'value':round(ratio_2,4)})
            
            result_dist = []
            for i in range(len(version_numbers)):
                
                if version_numbers[i].params_dist== distance_value[i]['value']:
                    result_dist.append({'id': version_numbers[i].id_version, 'type': 'distance', 'key': version_numbers[i].params, 'value':100})
                else:
                    result_dist.append({'id': version_numbers[i].id_version, 'type': 'distance', 'key': version_numbers[i].params, 'value':0})
                
                if version_numbers[i].params_ratio == ratio_value[i]['value']:
                    result_dist.append({'id': version_numbers[i].id_version, 'type': 'ratio', 'key': version_numbers[i].params, 'value':100})
                else:
                    result_dist.append({'id': version_numbers[i].id_version, 'type': 'ratio', 'key': version_numbers[i].params, 'value':0})
                
                if version_numbers[i].image_breath == breath and version_numbers[i].image_length == length :
                    result_dist.append({'id': version_numbers[i].id_version, 'type': 'image', 'key': version_numbers[i].image_key, 'value':100, 'key1': version_numbers[i].image_key, 'value1':100})
                else:
                    result_dist.append({'id': version_numbers[i].id_version, 'type': 'image', 'key': version_numbers[i].image_key, 'value':0,'key1': version_numbers[i].image_key, 'value1':100})
                    
            # print(f"result_dist:{result_dist}")
            result_df = pd.DataFrame(result_dist)
            print(f"result_df:{result_df}")
            
            df = result_df.loc[:,["type","key","value"]]
            key_values = dict(zip(zip((df['type'] + df['key'])), (df["value"])))
            dict_copy = {key[0]: value for key, value in key_values.items()}
            
            copy = list(dict_copy.values())
            total = sum(copy)/len(copy)
            print("total",total)
            
            dict_copy['total'] = round(total)
            print("dict_copy",dict_copy)
            result_list.append(dict_copy)
            
            grouped = result_df.groupby(['id'])
            mean = grouped['value'].agg(np.mean)
            
            mean_dict = mean.to_dict()
            update_mean_dict.update(mean_dict)
            
        result_score_dict= list(update_mean_dict.values())
        result = max(result_score_dict) 
        
        for i in result_list:
            total_list.append(i['total']) 
        
        total_max = max(total_list)
        print("total_max",total_max)
        
        key_dict ={}
        for j in result_list:
            if total_max == j['total']:
                key_dict = j
        
        proof_dict={"score":result, "key_score":key_dict,"id_type":type_id}
        return proof_dict
        
    except Exception as e:
        logging.exception("verify : exception : {}".format(e))
    logging.debug("verify : end")
    return 0

def verify_image(img):
    """
    Verify Image Method 
    To Find Co-Ordinates of Image Location 
    Calculate the Length and Breath of Image
    
    """
    logging.debug("verify_image : start")
    try:
        #Image
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=[40, 60],
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        logging.info(f"faces:{faces}")
        
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]
        
        cor1 = (y ,x)
        logging.info(f"cor1:{cor1}")
        cor2 = (w,x)
        logging.info(f"cor2:{cor2}")
        cor3 = (y,h)
        logging.info(f"cor3:{cor3}")
        cor4 = (w,h)
        logging.info(f"cor4:{cor4}")
        
        #breath = y - w
        #length = x - h  
        breath = cor3[0] - cor4[0]
        length = cor2[1] - cor4[1]
        return breath, length   
    except Exception as e:
        logging.error("verify_image : exception : {}".format(e))
    logging.debug("verify_image : end")
    return []

@app.route("/value", methods=["POST"])
def value():
    logging.debug("value : start")
    resp_dict = {"status":False, "msg":"", "object":None}

    image = request.files['image']
    input_json = request.form.get('id_type')
    
    type_id = input_json
    
    if image:
        str_time =  datetime.datetime.now().strftime('%d%m%Y%H%M%S')
        image_file_name = str_time+".jpg"
        
        logging.info(os.path.isdir(IMAGE_PATH))
        if os.path.isdir(IMAGE_PATH)==False:
            os.mkdir(IMAGE_PATH)
            
        # image save
        image.save(os.path.join(IMAGE_PATH,image_file_name))
        
    img = cv2.imread(IMAGE_PATH+"/"+image_file_name)
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['level'])
    logging.info(d['text'])
    
    response_dict = {}
    
    for i in range(n_boxes):
        value_dict = {}
        value_dict['left'] = d['left'][i]
        value_dict['top'] = d['top'][i]
        value_dict['width'] = d['width'][i]
        value_dict['height'] = d['height'][i]
        
        response_dict[d['text'][i]] = value_dict
        
    image = verify_image(img)
    lst = ["breath", int(image[0]),"length",int(image[1])]
    
    def Convert(lst):
        res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
        return res_dct
    
    length_breath = Convert(lst)
    
    result_dict={"text":response_dict, "image":length_breath,"id_type":type_id}
    resp_dict["object"] = result_dict
    resp_dict ["status"] = True
    
    os.remove(IMAGE_PATH+"/"+image_file_name)
    
    logging.debug("value : end")
    return jsonify(resp_dict)

@app.route("/add-config", methods=["POST"])
def add_config():
    """Add Config"""
    logging.debug("add_config : start")
    resp_dict = {"status":False, "msg":"", "object":None}
    try:
        id_type = request.json.get("id_type")
        dict_params = request.json.get("dict_params")
        breath = request.json.get("breath")
        length = request.json.get("length")
        
        # Find distance and ratio        
        distance_value = []
        ratio_value = []
        
        print("dict_params",dict_params)    
        
        length_dict_params =len(dict_params) 
        if length_dict_params>1:
            params_key = list(dict_params.keys())
            params_value = list(dict_params.values())
            for i in range(0,length_dict_params,2):
                x1 = params_value[i][0]
                y1 = params_value[i][1]
                x2 = params_value[i+1][0]
                y2 = params_value[i+1][1]
                r1 = params_value[i][2]
                r2 = params_value[i][3]
                r3 = params_value[i+1][2]
                r4 = params_value[i+1][3]
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                ratio_1 = r1/r2
                ratio_2 = r3/r4
                dist = round(dist,4)  
                    
                distance_value.append({'key': params_key[i], 'value':dist })
                distance_value.append({'key': params_key[i+1], 'value':dist })
                
                ratio_value.append({'key': params_key[i], 'value':round(ratio_1,4)})
                ratio_value.append({'key': params_key[i+1], 'value':round(ratio_2,4)})
            
            version_all = db.session.query(Config.id_version).filter(Config.id_type==id_type).distinct(Config.id_version).all()
            version_types = [i[0] for i in version_all]
            
            if version_types:
                id_version = max(version_types) + 1
            else:
                id_version = 1
            
            for i in range(len(distance_value)):
                for j in range(len(ratio_value)):
                    if i == j:
                        config = Config(id_type,id_version,distance_value[i]['key'],distance_value[i]['value'],ratio_value[j]['value'],breath,length) 
                        db.session.add(config)
                        db.session.commit()
        
        resp_dict["msg"] = "Config Added Successfully"
        resp_dict ["status"] = True
    except Exception as e:
        logging.error("add_config : exception : {}".format(e))
        resp_dict["msg"] = "Internal Server Error"
    logging.debug("add_config : end")
    return jsonify(resp_dict)

@app.route("/document-types", methods=["POST"])
def document_types():
    """Document Types"""
    logging.debug("document_types : start")
    resp_dict = {"status":False, "object":None}
    try:
        types = db.session.query(Config.id_type).distinct().all()
        config_types = [i[0] for i in types]
        resp_dict["object"] = config_types
        resp_dict ["status"] = True
    except Exception as e:
        logging.error("document_types : exception : {}".format(e))
        resp_dict["msg"] = "Internal Server Error"
    logging.debug("document_types : end")
    return jsonify(resp_dict)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5001)