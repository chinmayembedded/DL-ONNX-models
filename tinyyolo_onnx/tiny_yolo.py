import numpy
import onnxruntime
from PIL import Image,ImageDraw
import cv2
sess = onnxruntime.InferenceSession("./tiny_yolov2/model.onnx")

imgpath = './images/pp1.jpg'
for i in sess.get_inputs():
    print('Input:', i)
for o in sess.get_outputs():
    print('Output:', o)


from PIL import Image,ImageDraw, ImageFont
img = Image.open(imgpath)
img2 = img.resize((416, 416))

X = numpy.asarray(img2)
X = X.transpose(2,0,1)
X = X.reshape(1,3,416,416)

out = sess.run(None, {'image': X.astype(numpy.float32)})
out = out[0][0]

def display_yolo(img, seuil):
    import numpy as np
    numClasses = 20
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

    def sigmoid(x, derivative=False):
        return x*(1-x) if derivative else 1/(1+np.exp(-x))

    def softmax(x):
        scoreMatExp = np.exp(np.asarray(x))
        return scoreMatExp / scoreMatExp.sum(0)

    clut = [(0,0,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,128),
            (128,255,0),(128,128,0),(0,128,255),(128,0,128),
            (255,0,128),(128,0,255),(255,128,128),(128,255,128),(255,255,0),
            (255,128,128),(128,128,255),(255,128,128),(128,255,128),(128,255,128)]
    label = ["aeroplane","bicycle","bird","boat","bottle",
             "bus","car","cat","chair","cow","diningtable",
             "dog","horse","motorbike","person","pottedplant",
             "sheep","sofa","train","tvmonitor"]

    draw = ImageDraw.Draw(img)
    for cy in range(0,13):
        for cx in range(0,13):
            for b in range(0,5):
                channel = b*(numClasses+5)
                tx = out[channel  ][cy][cx]
                ty = out[channel+1][cy][cx]
                tw = out[channel+2][cy][cx]
                th = out[channel+3][cy][cx]
                tc = out[channel+4][cy][cx]

                x = (float(cx) + sigmoid(tx))*32
                y = (float(cy) + sigmoid(ty))*32

                w = np.exp(tw) * 32 * anchors[2*b  ]
                h = np.exp(th) * 32 * anchors[2*b+1]

                confidence = sigmoid(tc)

                classes = np.zeros(numClasses)
                for c in range(0,numClasses):
                    classes[c] = out[channel + 5 +c][cy][cx]
                    classes = softmax(classes)
                detectedClass = classes.argmax()
                
                if seuil < classes[detectedClass]*confidence:
                    color =clut[detectedClass]
                    print("@@@@@@@", label[detectedClass])
                    x = x - w/2
                    y = y - h/2
                    draw.line((x  ,y  ,x+w,y ),fill=color, width=3)
                    draw.line((x  ,y  ,x  ,y+h),fill=color, width=3)
                    draw.line((x+w,y  ,x+w,y+h),fill=color, width=3)
                    draw.line((x  ,y+h,x+w,y+h),fill=color, width=3)
                    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
                    draw.text((x+50,y), text=label[detectedClass], font=fnt, fill=color)
    return img

img2 = img.resize((416, 416))
result = display_yolo(img2, 0.038)
result.save("result.png")
print(result)
#cv2.imshow("result", result)