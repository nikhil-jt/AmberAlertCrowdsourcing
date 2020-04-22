import numpy as np
from PIL import Image
from PIL import ImageDraw
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import json
import pytesseract


interpreter = Interpreter(model_path="plate_model.tflite") #, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = input_details[0]['shape']
print(input_details)
print(output_details)

def json_to_numpy(path):
    '''
    Opens anchors.json and labels.json
    Returns numpy array.
    '''
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

ANCHOR_POINTS = json_to_numpy('anchors.json')

def detect_plate(car_img):
  img = car_img.resize((width, height))
  input_data = np.array(img.getdata()).reshape((1, height, width, 3)).astype(np.float32)/255
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))[:,0]
  scores = np.squeeze(interpreter.get_tensor(output_details[1]['index']))[:,1]
  boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))

  ty = boxes[:,0] / float(10)
  tx = boxes[:,1] / float(10)
  th = boxes[:,2] / float(5)
  tw = boxes[:,3] / float(5)

  yACtr = ANCHOR_POINTS[:,0]
  xACtr = ANCHOR_POINTS[:,1]
  ha    = ANCHOR_POINTS[:,2]
  wa    = ANCHOR_POINTS[:,3]

  w = np.exp(tw) * wa
  h = np.exp(th) * ha

  yCtr = ty * ha + yACtr
  xCtr = tx * wa + xACtr

  yMin = yCtr - h / float(2)
  xMin = xCtr - w / float(2)
  yMax = yCtr + h / float(2)
  xMax = xCtr + w / float(2)
    
  boxes_normalised = [yMin * height, xMin * width, yMax * height, xMax * width]
  print(np.max(scores), boxes[np.argmax(scores)])
  print("Highest Scoring Box: {}".format(np.transpose(np.squeeze(boxes_normalised))[np.argmax(scores)]) )
  draw = ImageDraw.Draw(car_img)
 # draw.rectangle(np.transpose(np.squeeze(boxes_normalised))[np.argmax(scores)])
  box_normalised = np.transpose(np.squeeze(boxes_normalised))[np.argmax(scores)]
  width_ratio = car_img.size[0]/300
  height_ratio = car_img.size[1]/300
  box_normalised[0] *= height_ratio
  box_normalised[1] *= width_ratio
  box_normalised[2] *= height_ratio
  box_normalised[3] *= width_ratio
  draw.line([(box_normalised[1], box_normalised[0]), (box_normalised[1], box_normalised[2]), (box_normalised[3], box_normalised[2]), (box_normalised[3], box_normalised[0]), (box_normalised[1], box_normalised[0])])
  print(box_normalised)
  car_img.save('plate.jpg')
  plate = car_img.crop((box_normalised[1], box_normalised[0], box_normalised[3], box_normalised[2]))
  plate.save('plate-trimmed.jpg')
  print(pytesseract.image_to_string(img))
  print(pytesseract.image_to_string(plate))
  testimg = Image.open("test-plate.jpg")
  print(pytesseract.image_to_string(testimg))

#detect_plate(Image.open("train_img.jpg"))
#detect_plate(Image.open("h_s_f.jpg"))
#detect_plate(Image.open("h_s_b.jpg"))
#detect_plate(Image.open("e_f.jpg"))
#detect_plate(Image.open("e_b.jpg"))
detect_plate(Image.open("test.jpg"))
#detect_plate(Image.open("testcar.jpg"))
