from app import app ,logging, STATUS,db
from model import *
from flask import jsonify, request, sessions
from sqlalchemy.sql.expression import distinct

import matplotlib.pyplot as plt

import os,base64

import cv2 
import pytesseract
import math
import numpy as np

import glob

#pytesseract.pytesseract.tesseract_cmd = "/home/indium/Documents/develop/tesseract\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\Indium Software\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

from pytesseract import Output

import datetime
import pandas as pd

#IMAGE_PATH = "/home/indium/Documents/develop/id_verify/backend/images"

IMAGE_PATH = "C:\\Users\\Indium Software\\Documents\\develop\\proof-backend\\images"

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
    key_list = []
    try:
        verified_image = verify_image(image)
        
        breath = verified_image[0]
        length = verified_image[1]
        
        image_logo = Img.query.filter(Img.id_type==type_id).first()
        simage = app.config["LOGO_IMAGES"]+image_logo.img
        
        print("searchimage",simage)
            
        searchimage =  simage
        save_path = ''
        
        str_time =  datetime.datetime.now().strftime('%d%m%Y%H%M%S')
        image_file_name = str_time+".jpg"
        
        files = (IMAGE_PATH+"/"+image_file_name)
        
        image_holder = imagesearch(searchimage, files, save_path)
        image_holder.detect()
        
        image_logo = image_holder.search_feature_match(searchimage,files)
        print("---image_logo---",image_logo)
        
        if image_logo == 'Logo Not found':
            logo_image = 0
        else:
            logo_image = image_logo
        
        #OCR
        original_text_extract = pytesseract.image_to_data(image, output_type=Output.DICT)
        n_boxes = len(original_text_extract['level'])
        for config in config_obj:
            id_type=config.id_type
        
        # Find unique id version    
        version_all = db.session.query(Config.id_version).filter(Config.id_type==id_type).distinct(Config.id_version).all()
        
        total_list = list()
        update_mean_dict = dict()
        result_list = list()
        for version in version_all:
            id_version = version[0]
            dict_params = dict()
            list_of_params = list()
            
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
            
            print("---list_of_params---",list_of_params)
            
            print("dict_params",dict_params)
                            
            # ratio
            ratio_value = list()
            config_params = [version.params for version in version_numbers]
            print("--config_params--",config_params)
            
            length_dict_params =len(dict_params) 
            if length_dict_params>1:
                params_key = list(dict_params.keys())
                params_value = list(dict_params.values()) 
                for config_par in config_params:
                    print("--config_par--",config_par)
                    if config_par in params_key:
                        for key, value in dict_params.items():
                            if key == config_par:
                                r1 = value[2]
                                r2 = value[3]
                                ratio = r1/r2
                                ratio_value.append({'key': key, 'value':round(ratio,4)})
                    else:
                        ratio_value.append({'key': config_par , 'value':0})
            else:
                for version in version_numbers:
                    ratio_value.append({'key': version.params , 'value':0})
                
            result_dist = list()
            
            print("ratio_value",ratio_value)
            
            for i in range(len(version_numbers)):
                
                per = [version_numbers[i].params_ratio, ratio_value[i]['value']]
                per_min = min(per)
                per_max = max(per)
                percentage_ratio = (per_min/per_max)*100
                
                per_breath = [version_numbers[i].image_breath, breath]
                per_b_min = min(per_breath)
                per_b_max = max(per_breath)
                per_image_breath = (per_b_min/per_b_max)*100
                
                per_length = [version_numbers[i].image_length, length]
                per_l_min = min(per_length)
                per_l_max = max(per_length)
                per_image_length = (per_l_min/per_l_max)*100
                
                per_logo = [version_numbers[i].image_logo, logo_image]
                per_l_min = min(per_logo)
                per_l_max = max(per_logo)
                per_image_logo = (per_l_min/per_l_max)*100
            
                if version_numbers[i].params_ratio == (ratio_value[i]['value']):
                    result_dist.append({'id': version_numbers[i].id_version, 'param_type': 'Ratio', 'param_value': version_numbers[i].params, 'value':100, 'actual_dimension': ratio_value[i]['value'],'expected_dimension':version_numbers[i].params_ratio, 'percentage':round(percentage_ratio)})
                else:
                    result_dist.append({'id': version_numbers[i].id_version, 'param_type': 'Ratio', 'param_value': version_numbers[i].params, 'value':round(percentage_ratio), 'actual_dimension': ratio_value[i]['value'],'expected_dimension':version_numbers[i].params_ratio, 'percentage':round(percentage_ratio)})
                    
            if version_numbers[i].image_breath == breath :
                result_dist.append({'id': version_numbers[i].id_version, 'param_type': 'Image', 'param_value': version_numbers[i].key_breath, 'value':100, 'actual_dimension': breath,'expected_dimension':version_numbers[i].image_breath,'percentage':round(per_image_breath)})
            else:
                result_dist.append({'id': version_numbers[i].id_version, 'param_type': 'Image', 'param_value': version_numbers[i].key_breath, 'value':round(per_image_breath), 'actual_dimension': breath,'expected_dimension':version_numbers[i].image_breath,'percentage':round(per_image_breath)})
                
            if version_numbers[i].image_length == length :
                result_dist.append({'id': version_numbers[i].id_version, 'param_type': 'Image', 'param_value': version_numbers[i].key_length, 'value':100, 'actual_dimension': length, 'expected_dimension':version_numbers[i].image_length,'percentage':round(per_image_length)})
            else:
                result_dist.append({'id': version_numbers[i].id_version, 'param_type': 'Image', 'param_value': version_numbers[i].key_length, 'value':round(per_image_length), 'actual_dimension': length, 'expected_dimension':version_numbers[i].image_length,'percentage':round(per_image_length)})
                
            if version_numbers[i].image_logo == logo_image :
                result_dist.append({'id': version_numbers[i].id_version, 'param_type': 'Logo', 'param_value': version_numbers[i].key_logo, 'value':100, 'actual_dimension': logo_image,'expected_dimension':version_numbers[i].image_logo,'percentage':round(per_image_logo)})
            else:
                result_dist.append({'id': version_numbers[i].id_version, 'param_type': 'Logo', 'param_value': version_numbers[i].key_logo, 'value':round(per_image_logo), 'actual_dimension': logo_image,'expected_dimension':version_numbers[i].image_logo,'percentage':round(per_image_logo)})
                
            print("result_dict",result_dist)
                    
            result_df = pd.DataFrame(result_dist)
            
            df = result_df.loc[:,["param_value","value","actual_dimension","expected_dimension","percentage"]]
            data = df.to_dict('records')
            
            copy = [ sub['value'] for sub in data ]
            total = sum(copy)/len(copy)
    
            a_dictionary = {"total" : round(total)}
            data.append(a_dictionary)
            result_list.append(data)
            
            # 
            grouped = result_df.groupby(['id'])
            mean = grouped['value'].agg(np.mean)
            
            mean_dict = mean.to_dict()
            update_mean_dict.update(mean_dict)
            
        result_score_dict= list(update_mean_dict.values())
        result = max(result_score_dict) 
        #
        for i in result_list:
            total_list.append(i[-1]['total']) 
        
        total_max = max(total_list)
        
        for j in result_list:
            if total_max == j[-1]['total']:
                key_list = j
        
        proof_dict={"score":round(result), "key_score":key_list[:-1],"id_type":type_id}
        return proof_dict
        
    except Exception as e:
        logging.exception("verify : exception : {}".format(e))
    logging.debug("verify : end")
    return {"score":0, "key_score":key_list[:-1],"id_type":type_id}

def verify_image(img):
    """
    Verify Image Method 
    To Find Co-Ordinates of Image Location 
    Calculate the Length and Breath of Image
    
    """
    logging.debug("verify_image : start")
    try:
        face_cascade = cv2.CascadeClassifier("C:\\Users\\Indium Software\\AppData\\Local\\Programs\\Python\\Python38\\haarcascade_frontalface_default.xml")
        img_size = img.shape
        print("image dimensions", img.shape) # width , height
        # width = img_size[0]/2
        # height = img_size[1]/2
        
        # width = img_size[0]
        # height = img_size[1]
        # print("width",width) # 319
        # print("height",height)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            print("xxx",x) # 795
            print("yyy",y)
            print("www",w)
            print("hhh",h)
            
            #xxx 795
            #yyy 439
            #www 109
            #hhh 109
            
            breath = abs(y - w)
            length = abs(x - h)  
        
            # breath = abs(cor3[0] - cor4[0])
            # length = abs(cor2[1] - cor4[1])
            
            # ratio = w/h
            
            # print("ratio",ratio)
            
            # if x > width:
            #     id_type = "PAN"
            # else:
            #     id_type = ""
            
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            # roi_gray = gray[y:y+h, x:x+w]
            # roi_color = img[y:y+h, x:x+w]
            # sub_face = img[y:y+h, x:x+w]
            # face_file_name = "sample.jpg"
            # plt.imsave(face_file_name, sub_face)
            # plt.imshow(sub_face)
            
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
    id = request.form.get('id_type')
    logo_image_data = request.form.get("logo")
    
    logo_image = logo_image_data.split(",")[1]
    str_time =  datetime.datetime.now().strftime('%d%m%Y%H%M%S')
    logo_image_file_name = str_time+".jpg"
    
    with open(os.path.join(app.config["LOGO_IMAGES"], logo_image_file_name), "wb+") as f:
        f.write(base64.b64decode(logo_image))

    type_id = id
    
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
    response_list = []
    
    for i in range(n_boxes):
        value_dict = {}
        value_dict['left'] = d['left'][i]
        value_dict['top'] = d['top'][i]
        value_dict['width'] = d['width'][i]
        value_dict['height'] = d['height'][i]
        
        response_dict[d['text'][i]] = value_dict
        
    for key , value in response_dict.items():
        response_list.append({'param_key':key, 'left':value['left'], 'top':value['top'], 'width':value['width'], 'height':value['height']})
        
    image = verify_image(img)
    lst = ["breath", int(image[0]),"length",int(image[1])]
    
    def Convert(lst):
        res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
        return res_dct
    
    length_breath = Convert(lst)
    
    logo_present = db.session.query(Img.img).filter(Img.id_type==type_id).all()
    print("logo_present",len(logo_present))
    
    if len(logo_present) == 0:
        img = Img(type_id, logo_image_file_name)
        db.session.add(img)
        db.session.commit()
        
    else:
        Img.img = logo_image_file_name
        db.session.commit()
    
    image_logo = Img.query.filter(Img.id_type==type_id).first()
    
    simage = app.config["LOGO_IMAGES"]+image_logo.img
    
    print("searchimage",simage)
    
    searchimage =  simage
    save_path = ''
    
    files = (IMAGE_PATH+"/"+image_file_name)
    
    image_holder = imagesearch(searchimage, files, save_path)
    image_holder.detect()
    
    logo_ratio = image_holder.search_feature_match(searchimage,files)
    print("---logo_ratio---",logo_ratio)
    
    #"image": {'breath': 0, 'length': 0},
    
    result_dict={"text":response_list,"image":length_breath, "image_logo":logo_ratio,"id_type":type_id}
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
        image_breath = request.json.get("breath")
        image_length = request.json.get("length")
        image_logo = request.json.get("logo")
        
        if image_logo == "Logo Not found":
            image_logo_value = 0
        else:
            image_logo_value = image_logo
        
        # Ratio        
        ratio_value = list()
        
        length_dict_params =len(dict_params) 
        print("length_dict_params",length_dict_params)
        if length_dict_params>1:
            params_key = list(dict_params.keys())
            print("params_key",params_key)
            params_value = list(dict_params.values())
            print("params_value",params_value)
            for i in range(0,length_dict_params,1):
                r1 = params_value[i][2]
                r2 = params_value[i][3] 
                ratio = r1/r2
                ratio_value.append({'key': params_key[i], 'value':round(ratio,4)})

            print("ratio_value", ratio_value)
            
            version_all = db.session.query(Config.id_version).filter(Config.id_type==id_type).distinct(Config.id_version).all()
            version_types = [i[0] for i in version_all]
            
            if version_types:
                id_version = max(version_types) + 1
            else:
                id_version = 1
            
            for j in range(len(ratio_value)):
                    config = Config(id_type,id_version,ratio_value[j]['key'],0,ratio_value[j]['value'],abs(image_breath),abs(image_length),image_logo_value)
                    db.session.add(config)
                    db.session.commit()
                        
        config = db.session.query(Config.config_id).filter(and_(Config.id_type==id_type, Config.id_version==id_version, Config.status=='A')).distinct(Config.config_id).all()
        config_all = [i[0] for i in config]
        #config_id = len(config_all)/2
        
        final_dict = {"config":f"{len(config_all)} Config  Added", "id_version": id_version}
        
        resp_dict["msg"] = "Config Added Successfully"
        resp_dict['object'] = final_dict
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

class imagesearch():
    def __init__(self, queryimage, searchimage, save_path):
            self.queryimage = queryimage
            self.searchimage = searchimage
            self.save_path = save_path
            self.image_file =  os.path.basename(searchimage)
            self.initialize() 
                
    def initialize(self):
        self.order_id = 1
    
    def search_feature_match(self,queryimage, searchimage):
        try:
            img1 = cv2.imread(queryimage,0)     
            print("-------------------------")    
            img2 = cv2.imread(searchimage,0)
            sift = cv2.xfeatures2d.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)
            print("No of Key Points", len(kp2))
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
            search_params = dict(checks=1000)  
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2,k=2)

            good = []
            contorurarray = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    contorurarray.append([m.distance,n.distance])
                    good.append(m)

            print('No.of Good Points',len(good))
            print('No.of Matching points',len(matches))
            
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            print('No.of dst_pts',len(dst_pts))
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = img1.shape[:2]
            h1,w1 = img2.shape[:2]
            print("height",h1) #638
            print("width",w1) #995
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            print("pts",pts)

            dst = cv2.perspectiveTransform(pts,M)
            draw_params = dict(matchColor = (0,255,0), singlePointColor = None,  matchesMask = matchesMask, flags = 2)
            img3 = cv2.polylines(img2, [np.int32(dst)], True, (0,255,0),10, cv2.LINE_AA)
            print("---Destination points---",dst)
            
            ratio = w1/h1
            
            # dst_pos_0 = dst[0][0][0] # 493.32 
            # # dst_pos_1 = dst[1][0][0]
            # # dst_pos_2 = dst[2][0][0]
            # # dst_pos_3 = dst[3][0][0]
            
            # print("dst_pos_0",dst_pos_0)
            # value = 100
            # position = round(w1)/2  # 497.5
            # pos1 = position + value
            # pos2 = position - value

            # print("pos1",pos1)
            # print("pos2",pos2)
            
            # if pos2 <= dst_pos_0 <= pos1:
            #     id_type = "PAN"
            # else:
            #     id_type = " "   
            return round(ratio,4)
    
        except Exception as e:
            print(e) 
        return "Logo Not found"
        
    #cv2.imwrite(os.path.join(self.save_path,self.image_file+"search_feature_match.jpg"), img3)

    def rectangle_focus(self,queryimage, searchimage):
        print('------Rectangle Focus Initated-----')
        image = cv2.imread(searchimage)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        edge_enh = cv2.Laplacian(gray, ddepth = cv2.CV_8U, 
                                ksize = 3, scale = 1, delta = 0)
        blurred = cv2.bilateralFilter(edge_enh, 13, 50, 50)
        (_, thresh) = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations = 4)
        closed = cv2.dilate(closed, None, iterations = 4)
        (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        retval = cv2.imwrite(os.path.join(self.save_path,self.image_file+self.image_file+"rectangle_focus.jpg"), image)

    def detect(self,**kwargs):
        self.search_feature_match(self.queryimage, self.searchimage)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5001)
