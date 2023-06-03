import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from mtcnn.mtcnn import MTCNN
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
#from tensorflow.keras.models import load_model
 
 #Cargar imagenes
def cargar_imagen(directorio):
    #return cv2.cvtColor(cv2.imread(f'{directorio}/{nombre}'),cv2.COLOR_BGR2RGB)
    return cv2.imread(f'{directorio}',1)

 #----------------- Detectamos el rostro y exportamos los pixeles --------------------------
    
def reg_rostro(img, lista_resultados):
    #data = pyplot.imread(img)
    #data = cargar_imagen("Conocidos", "varios.jpg")
    #cv2.imwrite("./Resultados/orig.jpg",data)
    solo_caras = []
    for i in range(len(lista_resultados)):
        x1,y1,ancho, alto = lista_resultados[0][i]
        x2,y2 = x1 + ancho, y1 + alto
        cara_reg = img[y1:y2, x1:x2]
        cara_reg = cv2.resize(cara_reg,dsize=(96,96)) #Guardamos la imagen con un tamaño de 150x200
        #cv2.imwrite("./Resultados/"+str(i)+".jpg",cara_reg)
        solo_caras.append(cara_reg)
        
    return solo_caras

#help(MTCNN)
path = "./Conocidos/varios.jpg"
#pixeles = pyplot.imread(img)
img = cargar_imagen(path)
detector = MTCNN(keep_all=True)
frame = Image.fromarray(img)
#caras_detecciones = detector.detect(frame)
#print(caras_detecciones)
#caras = reg_rostro(frame, caras_detecciones)

# Detect face
boxes, probs = detector.detect(frame)
print(boxes)
# Visualize
fig, ax = plt.subplots(figsize=(16, 12))
ax.imshow(frame)
ax.axis('off')

for box in boxes:
    ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
fig.show()
plt.waitforbuttonpress()

# load the model
#model = load_model('./Model/facenet_keras.h5')
# summarize input and output shape
#print(model.inputs)
#print(model.outputs)
#cv2.imshow('Imagen', caras[0])
#cv2.waitKey(0)