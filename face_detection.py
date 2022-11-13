from mtcnn import MTCNN
import cv2
import os
import matplotlib.pyplot as plt
import time
import datetime
import json


IMAGES_PATH = os.path.join('data','images', 'maskoff')

def image_capture():
    """ Captura da imagem com Opencv """
    img = cv2.cvtColor(cv2.imread("./data/cam_images/captured_image.jpg"), cv2.COLOR_BGR2RGB)
    return detector_MTCNN(img)

def detector_MTCNN(img):
    """ Detecção da face com o método de  P-Net, R-Net, and O-Net - Cascaded Neural Networks """

    detector = MTCNN()
    faces = detector.detect_faces(img)
    return boxes_layer(img, faces)


def boxes_layer(image, faces):
    """ mostrar na imagem a localização das faces """

    for face in faces:
        # extraindo as coordenadas para o retângulo e keypoints e a prob.
        x, y, w, h = face['box']
        probs = face['confidence']
        kp = face['keypoints']

        # desenha o retângulo
        # rect = plt.Rectangle((x,y), width, height, color = 'green', fill = False)
        output_image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        # desenha keypoints
        cv2.circle(image, (kp['left_eye']), 1, (0, 155, 255), 2)
        cv2.circle(image, (kp['right_eye']), 1, (0, 155, 255), 2)
        cv2.circle(image, (kp['nose']), 1, (0, 155, 255), 2)
        cv2.circle(image, (kp['mouth_left']), 1, (0, 155, 255), 2)
        cv2.circle(image, (kp['mouth_right']), 1, (0, 155, 255), 2)

        # probabilidade
        # cv2.putText(image, str(
        #             probs), (x+w, y+w), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2, cv2.LINE_AA)
    plt.imshow(image)
    plt.savefig('./data/result/contagem.jpg')

    #passsagem dos valores para função que armazena a contagem
    dt_now = datetime.datetime.now()
    count = len(faces)
    return (dt_now,count)

def save_detection(dt_now,count):
    """ Salva a contagem de rostos com horario """

    new_data = {
       'date_time': dt_now,
       'Total_faces': count
    }
    print(new_data)
    # with open('./data/result/count_faces.json', 'r+') as file:
    #
    #     #dados existentes no json em um dict
    #     file_data = json.load(file)
    #     # junta o novo dado com o existente no arquivo
    #     file_data["emp_details"].append(new_data)
    #     # Sets file's current position at offset.
    #     file.seek(0)
    #     # convert back to json.
    #     json.dump(file_data, file, indent=4)





def job_capture_time():
    """ Função com o tempo para pausa e tempo de execução da capturas de pacotes """
    capture = True
    ini_exec = datetime.datetime.now()
    time_keep_capturing = datetime.datetime.now() + datetime.timedelta(seconds=200)
    pause_time_sec = 20
    file_number = 0
    while datetime.datetime.now() <= time_keep_capturing:
        try:
            file_number = file_number + 1
            capture_image_cam(capture)
            image_capture()
            time.sleep(pause_time_sec)
        except:
            break
        end_exec = datetime.datetime.now()

        execution_time = end_exec - ini_exec
        print(ini_exec, end_exec, time_keep_capturing, file_number)

def capture_image_cam(time):
    """ Captura imagem da webcam """
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            print(check) #prints true  enquanto a cam está aberta
            print(frame) #prints matrix values of each framecd
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            cv2.imwrite(filename='./data/cam_images/captured_image.jpg', img=frame)
            webcam.release()
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Resizing image to 416x416 scale...")
            img_ = cv2.resize(frame,(416,416))
            print("Resized...")
            img_resized = cv2.imwrite(filename='./data/cam_images/captured_image.jpg', img=img_)
            print("Image saved!")
            capture = False
            break
        except():
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
    return img_resized

if __name__ == '__main__':
    job_capture_time()
