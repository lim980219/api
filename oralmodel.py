# Processing Input and Output Data
from PIL import Image, ImageColor
import torch
from torchvision import transforms as T
import numpy as np
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import os
import cv2



def get_image(arg_parser):
    '''Returns a Pillow Image given the uploaded image.'''
    args = arg_parser.parse_args()
    image_file = args.image 
    openImage = Image.open(image_file)
    fileName = os.path.splitext(image_file.filename)[0]
    return openImage.convert('RGB'), fileName

# disease label colors
def disease_color_map():
    # disease_label_colors = ([(0, 0, 0),  # 0=background
    #    # 1=C1, 2=C2, 3=C3, 4=GI1, 5=GI2
    #    (255, 255, 0), (0, 255, 255), (255, 0, 255), (102, 0, 0), (102, 0, 204),
    #    # 6=GI3, 7=PD1, 8=PD2, 9=PD3
    #    (255, 94, 0), (255, 0, 127), (102, 102, 255), (0,88,29)])

    disease_label_colors = ([(0, 0, 0),  # 0=background
       # 1=C1, 2=C2, 3=C3, 4=GI1, 5=GI2
       (153, 255, 153), (0, 255, 0), (0, 102, 0), (102, 102, 255), (0, 0, 255),
       # 6=GI3, 7=PD1, 8=PD2, 9=PD3
       (0, 0, 102), (255, 153, 204), (255, 0, 127), (102,0,51)])
    return disease_label_colors

# Hygiene label colors
def hygiene_color_map():
                        #0:'background',1:'am',2:'cecr',3:'gcr',4:'mcr',5:'ortho',6:'tar1',7:'tar2', 8:'tar3', 9:'zircr'
       hygiene_label_colors= [(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(255, 204, 153),(255,128, 0),(102,51, 0),(0,0,0)]
       return hygiene_label_colors


def preprocess_image(image):
    """Converts a PIL.Image into a Tensor of the 
    right dimensions 
    """
    # resizing input image to 224,224 and 448
    resized_input_image = image.resize((224, 224,), Image.NEAREST)
    resized_input_image_448 = image.resize((448, 448,), Image.NEAREST)

    # conver pillow image to tensor
    transform_to_tensor=T.Compose([T.PILToTensor(),T.Resize((224,224))])
    transform_to_tensor_448=T.Compose([T.PILToTensor(),T.Resize((448,448))])
    resized_tensor_image = transform_to_tensor(resized_input_image)
    resized_tensor_image_448 = transform_to_tensor_448(resized_input_image_448)

    return resized_input_image, resized_input_image_448, resized_tensor_image, resized_tensor_image_448


def predict_disease(model, image):

    # converting to tensor
    t = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    image = t(image)
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)

    return masked

def find_labels(predicted_mask, dictionary):

        pred_labelname=[]
        for i in dictionary:
                if dictionary[i] in predicted_mask.cpu().numpy():
                    pred_labelname.append(i)
        return pred_labelname


def draw_segmentation_masks(alpha,
    image: torch.Tensor,
    masks: torch.Tensor,
    colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    

    img_to_draw = Image.fromarray(masks)

    if colors is None:
        num_classes = 20
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1, 0])
        colors_t = torch.as_tensor([i for i in range(num_classes)])[:, None] * palette
        color_arr = (colors_t % 255).numpy().astype("uint8")
        color_arr[1:, 3] = 255
    else:
        color_list = []
        for color in colors:
            if isinstance(color, str):
                fill_color = ImageColor.getrgb(color) 
                color_list.append(fill_color)
            elif isinstance(color, tuple):
                color_list.append(color)

        color_arr = np.array(color_list).astype("uint8")
    img_to_draw.putpalette(color_arr, rawmode='RGB')
    img_to_draw = torch.from_numpy(np.array(img_to_draw.convert('RGBA')))
    img_to_draw = img_to_draw.permute((2, 0, 1))
    v= (torch.cat([image, torch.full(image.shape[1:], 255).unsqueeze(0)]).float()*alpha+img_to_draw.float()*(1.0-alpha)).to(dtype=torch.uint8)
    return v, img_to_draw


def adding_contour(mask_image):
    
    opencv_mask_image=cv2.cvtColor(np.array(mask_image), cv2.COLOR_RGB2BGR)
    gray_mask =  cv2.cvtColor(opencv_mask_image, cv2.COLOR_BGR2GRAY)
    filtered_mask = np.where(gray_mask > np.quantile(gray_mask, 0.2), gray_mask, 0)
    mask_threshold, thresh_image_mask = cv2.threshold(filtered_mask, 0, 255, cv2.THRESH_BINARY)

    # find contours
    contours1, hierarchy1 = cv2.findContours(image=thresh_image_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # draw contours
    mask1_copy = opencv_mask_image.copy()
    contour_mask = cv2.drawContours(image=mask1_copy, contours=contours1, contourIdx=-1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_8)

    return contour_mask
