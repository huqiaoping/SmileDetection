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
    

smileCascade = cv2.CascadeClassifier('xmls/haarcascade_smile.xml')
def detect_smile_face(in_img):    
    if in_img.ndim == 3:
        gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    elif in_img.ndim == 1:
        gray = in_img
    smile_face_locations = smileCascade.detectMultiScale(
            gray,
            scaleFactor=1.16,
            minNeighbors=35,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    return smile_face_locations

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:            
        res, frame = cap.read()
        
        if not res:
            print('err reading camera.')
            break
            
        draw_img = frame.copy()
        face_locations = detect_face(frame)        
        for (x, y, w, h) in face_locations:
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 225, 255), 2)
            face = frame[y:y + h, x:x + w]
            smile_face_locations = detect_smile_face(face)
            if len(smile_face_locations) > 0:
                cv2.rectangle(draw_img, (x, y), (x + w, y+ h), (0, 0, 255), 3)
                cv2.putText(draw_img, 'Smiling', (x, y - 7), 3, 1.2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Real-Time Face Detect', draw_img)
        cv2.waitKey(1)
        if msvcrt.kbhit() and ord(msvcrt.getch()) == ord('q'):
            break                
    cap.release()
    cv2.destroyAllWindows()
        

if __name__ == '__main__':
    main()