# Prevent ImportError w/ flask
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from tkinter import E
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.datastructures import FileStorage
# ML/Data processing
import torch
import torchvision
from torchvision import transforms as T
# RESTful API packages
from flask_restplus import Api, Resource
from flask import Flask, jsonify
from collections import OrderedDict, MutableMapping
# Utility Functions
import oralmodel
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
from flask import send_file
import dictionary_map
import cv2

# input and output result image folders

mask_image_folder='./mask_images/'
resized_image_folder ='./resized_images/'


application = app = Flask(__name__)
api = Api(app, version="1.0", title="Disease and Hygiene Detect Flask API", 
        description="Identifying if a image contains certain diseases and Tartar")
ns = api.namespace(
    "ArtificialIntelligence", 
    description="Represents the image category by the AI."
)

# Use Flask-RESTPlus argparser to process user-uploaded images
arg_parser = api.parser()
arg_parser.add_argument('image', location='files',
                           type=FileStorage, required=True)

# loading the disease model
model1 = smp.FPN('timm-efficientnet-b8', encoder_weights='imagenet', classes=10, activation=None)
model1.load_state_dict(torch.load('Best_Disease_data234_FPN_224_timE8_balanced.pt', map_location ='cpu'))
model1.eval()

# loading hygiene model
model2 = smp.UnetPlusPlus('efficientnet-b7', encoder_weights='imagenet', classes=10, activation=None)
model2.load_state_dict(torch.load('Hygiene_eff_new_4768.pt', map_location ='cpu'))
model2.eval()

print("Loaded model from disk")


@app.route('/download/<string:url>/<string:filename>')
def downloadFile (url,filename):
    return send_file('/'+url+'/'+filename, as_attachment=True)

# Add the route to run inference
@ns.route("/prediction3")
class CNNPrediction(Resource):
    """Takes in the image, to pass to the CNN"""
    @api.doc(parser=arg_parser, 
             description="Let EAPO predict if you have any tooth problem.")
    def post(self):
        # A: get the image
        image, fileName = oralmodel.get_image(arg_parser)
       
        # B: preprocess the image

        final_image, final_image_448, tensor_image, tensor_image_448 = oralmodel.preprocess_image(image)
        opencv_final_image = cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR)
        opencv_final_image_448 = cv2.cvtColor(np.array(final_image_448), cv2.COLOR_RGB2BGR)


        cv2.imwrite(resized_image_folder+'diseaseImagecv_'+fileName+'.png', opencv_final_image)
        cv2.imwrite(resized_image_folder+'hygieneImagecv_'+fileName+'.png', opencv_final_image_448)
        
        
        masked1 = oralmodel.predict_disease(model1, final_image)
        masked2 = oralmodel.predict_disease(model2, final_image_448)

        # find labels
        predicted_disease = oralmodel.find_labels(masked1, dictionary_map.disease_maps())
        predicted_hygiene = oralmodel.find_labels(masked2, dictionary_map.hygiene_maps())
        predicted_labels = predicted_disease + predicted_hygiene

        # converting predicted mask to numpy array
        pred_argmax1 = masked1.numpy().astype(np.uint8)
        pred_argmax2 = masked2.numpy().astype(np.uint8)

        # adding colors to only labels found area and combining it with the original image
        image_overlapped_mask1, mask1 = oralmodel.draw_segmentation_masks(0.6, tensor_image, pred_argmax1, colors=oralmodel.disease_color_map())
        image_overlapped_mask2, mask2= oralmodel.draw_segmentation_masks(0.6,tensor_image_448, pred_argmax2, colors=oralmodel.hygiene_color_map())

        transform = T.ToPILImage()

        mask1 = transform(mask1)
        mask2= transform(mask2)


        contour_disease_mask = oralmodel.adding_contour(mask1)
        contour_hygiene_mask = oralmodel.adding_contour(mask2)

        # new saving method
        cv2.imwrite(mask_image_folder+'diseaseMask_'+fileName+'.png', contour_disease_mask)
        cv2.imwrite(mask_image_folder+'hygieneMask_'+fileName+'.png', contour_hygiene_mask)

        output = {
            "Success": True,
            "data": predicted_labels,
            "disease_mask":mask_image_folder+'diseaseMask_'+fileName+'.png',
            "hygiene_mask":mask_image_folder+'hygieneMask_'+fileName+'.png',
            "disease_original_image":resized_image_folder+'diseaseImage_'+fileName+'.png',
            "hygiene_original_image":resized_image_folder+'hygieneImage_'+fileName+'.png'
        }
    
        # return the labels
        return output


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0',port=8080)


