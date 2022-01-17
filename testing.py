from app import app,db,logging

import application
import unittest
import json

from io import BytesIO

import ast

app.route(application)

class Flask_Test(unittest.TestCase):

    def setUp(self):
        app.config['Testing'] = True
        # example connections -'sqlite:///C:\\path\\to\\database.db'
        self.db_uri = 'postgresql://postgres:root@localhost:5432/proof_verification'
        app.config ['SQLALCHEMY_DATABASE_URI']=self.db_uri
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        app.config['Debug'] = False
        self.app = app.test_client()
        db.create_all()

    def tearDown(self):
        pass

    # Test User Proof
    def test_proof_original(self):
        url='/user-proof/PAN'
        
        with open('pan_originapng.png','rb') as img1:
            imgStringIO1 = BytesIO(img1.read())
        
        mock_request_data={'image': (imgStringIO1, 'img1.jpg')}

        response=self.app.post(url,data=mock_request_data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200) 
        
        resp = response.data
        dict_str = resp.decode("UTF-8")
        mydata = ast.literal_eval(dict_str)
        #print(mydata['object'])
        self.assertEqual(mydata['object']['score'], 100) 
        
    def test_proof_tampared(self):
        url='/user-proof/PAN'
        
        with open('aadhar.jpg','rb') as img1:
            imgStringIO1 = BytesIO(img1.read())
        
        mock_request_data={'image': (imgStringIO1, 'img1.jpg')}

        response=self.app.post(url,data=mock_request_data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200) 
        
        resp = response.data
        dict_str = resp.decode("UTF-8")
        mydata = ast.literal_eval(dict_str)
        #print(mydata['object'])
        self.assertEqual(mydata['object']['score'], 0)
        
    # def test_no_image(self):
    #     url='/user-proof/PAN'
        
    #     with open('aadhar.jpg','rb') as img1:
    #         imgStringIO1 = BytesIO(img1.read())
        
    #     mock_request_data={}

    #     response=self.app.post(url,data=mock_request_data, content_type='multipart/form-data')
    #     self.assertEqual(response.status_code, 200) 
    #     resp = response.data
    #     self.assertEqual(resp, 0)
              
if __name__ == "__main__":
    unittest.main()