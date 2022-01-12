import numpy as np

def calc_finger_cos(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point_3D = []
    # キーポイント
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = landmark.x
        landmark_y = landmark.y
        landmark_z = landmark.z
        landmark_point_3D.append([landmark_x, landmark_y,landmark_z])
    
    #listをarrayに変換
    landmark_point_3D_array = np.array(landmark_point_3D)

    #人差し指の指先と人差し指の第２関節
    v1 = landmark_point_3D_array[8] - landmark_point_3D_array[7]
    v2 = landmark_point_3D_array[5] - landmark_point_3D_array[7]
    
    finger_cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    #print("COS : " + str(finger_cos))
    return finger_cos