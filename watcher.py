from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import time
import subprocess
import cv2
import numpy as np
import av
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import sys

labels = dataset_utils.read_label_file("coco_labels.txt")
engine = DetectionEngine("mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")

interpreter = Interpreter(model_path="plate_model.tflite", experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
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

def recognize_plate(plate):
  plate = numpy.array(plate)
  gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
  gray = cv2.bilateralFilter(gray, 11, 17, 17)
  thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
  cv2.imwrite("gray.jpg", gray)
  cv2.imwrite("thresh.jpg", thresh[1])
  text = pytesseract.image_to_string(thresh[1], config='--psm 7')
  print("Detected Number is:", text)


def detect_plate(car_img):
  img = car_img.resize((width, height))
  input_data = np.array(img.getdata()).reshape((1, height, width, 3)).astype(np.float32) / 255
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))[:, 0]
  scores = np.squeeze(interpreter.get_tensor(output_details[1]['index']))[:, 1]
  boxes = np.squeeze(interpreter.get_tensor(output_details[0]['index']))

  ty = boxes[:, 0] / float(10)
  tx = boxes[:, 1] / float(10)
  th = boxes[:, 2] / float(5)
  tw = boxes[:, 3] / float(5)

  yACtr = ANCHOR_POINTS[:, 0]
  xACtr = ANCHOR_POINTS[:, 1]
  ha = ANCHOR_POINTS[:, 2]
  wa = ANCHOR_POINTS[:, 3]

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
  print("Highest Scoring Box: {}".format(np.transpose(np.squeeze(boxes_normalised))[np.argmax(scores)]))
  draw = ImageDraw.Draw(car_img)
  # draw.rectangle(np.transpose(np.squeeze(boxes_normalised))[np.argmax(scores)])
  box_normalised = np.transpose(np.squeeze(boxes_normalised))[np.argmax(scores)]
  width_ratio = car_img.size[0] / 300
  height_ratio = car_img.size[1] / 300
  box_normalised[0] *= height_ratio
  box_normalised[1] *= width_ratio
  box_normalised[2] *= height_ratio
  box_normalised[3] *= width_ratio
  draw.line([(box_normalised[1], box_normalised[0]), (box_normalised[1], box_normalised[2]),
             (box_normalised[3], box_normalised[2]), (box_normalised[3], box_normalised[0]),
             (box_normalised[1], box_normalised[0])])
  print(box_normalised)
  car_img.save('plate.jpg')
  plate = car_img.crop((box_normalised[1], box_normalised[0], box_normalised[3], box_normalised[2]))
  plate.save('plate-trimmed.jpg')
  recognize_plate(plate)



def detect_cars(img):
  ans = engine.detect_with_image(img, threshold=0.05, keep_aspect_ratio=True, relative_coord=False, top_k=1)
  if ans:
    draw = ImageDraw.Draw(img)
    for obj in ans:
       if labels[obj.label_id] in ['car', 'truck']:
         box = obj.bounding_box.flatten().tolist()
         print(box)
         draw.rectangle(box)
         car_img = img.crop(box)
         car_img.save('test.jpg')
         detect_plate(car_img)


folder='/mnt/TeslaCam/TeslaCam/RecentClips/'

def extract_key_frames(filename):
  container = av.open(filename)
  stream = container.streams.video[0]
  stream.codec_context.skip_frame = 'NONKEY'
  count = 0
  for frame in container.decode(stream):
#    print(count)
    img = frame.to_image()
    detect_cars(img)
    count += 1
    if count > 0:
      img.save('orig.jpg')
      return
  print(count)

extract_key_frames("./tesla.mp4")

class Handler(FileSystemEventHandler):
  @staticmethod
  def on_any_event(event):
#    print(event)
    if event.event_type == 'created':
      print(event.src_path)
      extract_key_frames(event.src_path)

#event_handler = Handler()
#observer = PollingObserver()
#observer.schedule(event_handler, folder, recursive=True)
#observer.start()
#while True:
#  time.sleep(5)


