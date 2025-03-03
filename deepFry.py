from PIL import Image, ImageEnhance
import cv2

filename = "group"  # must be a png file in the input_images folder

img_path = "input_images/"+filename+".png"

face_cascade = cv2.CascadeClassifier("resources/face.xml")
eye_cascade = cv2.CascadeClassifier("resources/eye.xml")

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
bg_img = Image.open(img_path)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey+eh), (0, 255, 0), 2)

    for (ex, ey, ew, eh) in eyes:
        flare = cv2.imread("resources/flare.png", -1)
        flare = cv2.resize(flare, (ew*4, eh*4))

        xpos = int(x+ex-ew*1.5)
        ypos = int(y+ey-eh*1.5)

        flare[:, :, [0, 2]] = flare[:, :, [2, 0]]
        fg_img = Image.fromarray(flare)

        bg_img.paste(fg_img, (xpos, ypos), mask=fg_img)

converter = ImageEnhance.Color(bg_img)
img2 = converter.enhance(6.0)
converter2 = ImageEnhance.Sharpness(img2)
img2 = converter2.enhance(20.0)
converter3 = ImageEnhance.Contrast(img2)
img2 = converter3.enhance(2.0)
img2.save("final_images/"+filename+'_fried.png')