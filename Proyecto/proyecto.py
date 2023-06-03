import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
#from mtcnn.mtcnn import MTCNN
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
 
 #Cargar imagenes
def cargar_imagen(directorio):
    #return cv2.cvtColor(cv2.imread(f'{directorio}/{nombre}'),cv2.COLOR_BGR2RGB)
    return cv2.imread(f'{directorio}',1)

 #----------------- Detectamos el rostro y exportamos los pixeles --------------------------
    
def reg_rostro(img, boxes):
    #data = pyplot.imread(img)
    #data = cargar_imagen("Conocidos", "varios.jpg")
    #cv2.imwrite("./Resultados/orig.jpg",data)
    solo_caras = []
    for box in boxes:
        cara_reg = img.crop((box))
        cara_reg = np.asarray(cara_reg)
        cara_reg = cv2.resize(cara_reg,dsize=(96,96)) #Guardamos la imagen con un tamaño de 96x96
        cv2.imwrite("./Resultados/"+str(len(solo_caras))+".jpg",cara_reg)
        solo_caras.append(cara_reg)
        
    return solo_caras

#help(MTCNN)
path = "./Conocidos/varios.jpg"
#pixeles = pyplot.imread(img)
img = cargar_imagen(path)
detector = MTCNN(keep_all=True)

# Detect face
boxes, probs = detector.detect(img)
frame = Image.fromarray(img)
caras = reg_rostro(frame, boxes)


# This return a pretrained model that is vggface2
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Read data from folder

dataset = datasets.ImageFolder('Conocidos') # photos folder path 
#dataset
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names
#print(idx_to_class)
def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    print(img.size)
    #Unlike other implementations, calling a facenet-pytorch MTCNN object directly with an image (i.e., using the forward method for those familiar with pytorch) will return torch tensors containing the detected face(s), rather than just the bounding boxes. This is to enable using the module easily as the first stage of a facial recognition pipeline, in which the faces are passed directly to an additional network or algorithm.
    face, prob = detector(img, return_prob=True)
    #print("face: ",face," prob:",prob)
    if face is not None and prob>0.92:
        print(face.squeeze(1).shape)
        #emb = resnet(face.unsqueeze(0))
        emb = resnet(face.squeeze(1)) 
        embedding_list.append(emb.detach()) 
        name_list.append(idx_to_class[idx])        

# save data
data = [embedding_list, name_list] 
torch.save(data, 'data.pt') # saving data.pt file



detector = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
# Using webcam recognize face

# loading data.pt file
load_data = torch.load('data.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 

cam = cv2.VideoCapture(0) 

while not cam.isOpened():
    cam = cv2.VideoCapture("./personas.mp4")
    cv2.waitKey(1000)
    print ("Wait for the header")

post_frame = cam.get(cv2.CAP_PROP_POS_FRAMES)

while True:
    ret, frame = cam.read()
    if not ret:
        print("fail to grab frame, try again")
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
        print("frame is not ready")
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)
        break
        
    img = Image.fromarray(frame)
    img_cropped_list, prob_list = detector(img, return_prob=True) 
    
    if img_cropped_list is not None:
        #return boxed faces
        boxes, _ = detector.detect(img)
                
        for i, prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                
                dist_list = [] # list of matched distances, minimum distance is used to identify the person
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list) # get minumum dist value
                min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                name = name_list[min_dist_idx] # get name corrosponding to minimum dist
                
                box = boxes[i]
                #print(type(box), box)
                original_frame = frame.copy() # storing copy of frame before drawing on it
                
                if min_dist<0.90:
                    #bgr
                    frame = cv2.putText(frame, str(name)+' '+str(min_dist), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (63, 0, 252),1, cv2.LINE_AA)
                    #frame = cv2.putText(frame, 'Hola ' +name, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (63, 0, 252),1, cv2.LINE_AA)

                #print(int(box[0]),int(box[1]),int(box[2]),int(box[3]))
                frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (13,214,53), 2)

    cv2.imshow("IMG", frame)

    if cv2.waitKey(10) == 27:
        break
    #if cam.get(cv2.CAP_PROP_POS_FRAMEScv2.CV_CAP_PRcv2.CAP_PROP_POS_FRAMESv2.CV_CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
#        # we stop
        break
        
    
    k = cv2.waitKey(1)
    if k%256==27: # ESC
        print('Esc pressed, closing...')
        break
        
    elif k%256==32: # space to save image
        print('Enter your name :')
        name = input()
        
        # create directory if not exists
        if not os.path.exists('data2/'+name):
            os.mkdir('data2/'+name)
            
        img_name = "data2/{}/{}.jpg".format(name, int(time.time()))
        cv2.imwrite(img_name, original_frame)
        print(" saved: {}".format(img_name))
        
        
cam.release()
cv2.destroyAllWindows()
