import argparse
import cv2,os,msvcrt

faceCascade = cv2.CascadeClassifier('xmls/haarcascade_frontalface_default.xml')
def detect_face(in_img):
    if in_img.ndim == 3:
        gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    elif in_img.ndim == 1:
        gray = in_img
    face_locations = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return face_locations
    
def main(config):

    use_camera = config.use_camera
    
    if use_camera:
        #real-time detecting face from computer's camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:            
            res, frame = cap.read()
            
            if not res:
                print('err reading camera.')
                break
                
            face_locations = detect_face(frame)
            
            for (x, y, w, h) in face_locations:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 255), 2)
                
            cv2.imshow('Real-Time Face Detect', frame)
            cv2.waitKey(1)
            if msvcrt.kbhit() and ord(msvcrt.getch()) == ord('q'):
                break                
        cap.release()
        cv2.destroyAllWindows()
    else:
        #detecting face from input image
        in_image = config.in_image
        draw_image = config.draw_image
        face_image = config.face_image
        
        if not os.path.exists(in_image):
            print('Cannot find image file %s, please check!' % in_image)
            return
            
        img = cv2.imread(in_image)
        face_locations = detect_face(img)        
        draw_img = img.copy()
        
        for (x, y, w, h) in face_locations:
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 225, 255), 2)
            face = img[y:y + h, x:x + w]
            cv2.imwrite(face_image, face)
            
        cv2.imwrite(draw_image, draw_img)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #usage:
    #face_detection.py --in_image example.jpg --draw_image example_out.jpg --face_image example_face.jpg
    #face_detection.py --use_camera True
    
    parser.add_argument('--in_image', type=str, default='example.jpg', help='source input image name')
    parser.add_argument('--draw_image', type=str, default='example_draw.jpg', help='draw face image name')
    parser.add_argument('--face_image', type=str, default='example_face.jpg', help='face image name (assume there is only one face in this image)')
    parser.add_argument('--use_camera', type=bool, default=False, help='detect faces from the default camera')
    
    config = parser.parse_args()
    
    print(config)
    main(config)