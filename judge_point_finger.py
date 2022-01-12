from calc_finger_cos import calc_finger_cos
from google.protobuf.json_format import MessageToDict
from model import KeyPointClassifier

# paramaters ---------------------------------------------------------------------
hard_cos_limit = -0.93
soft_cos_limit = -0.8


# 最初の人差し指の感知をsoftに、途中の人差し指の感知をhardに行う


# 「hand_sign_id=2 かつ cos<soft_limit」、またはcos<hard_limitでTrue。他はFalse
# soft_limitを入れたのは、手を閉じていてもid==2となってしまうため
# チョキとかを見分けられるわけではない
# 右手であることは前提
def judge_point_finger_soft(debug_image, hand_landmarks, pre_processed_landmark_list) : 

    # まずhand_sign_id を確認
    keypoint_classifier = KeyPointClassifier()
    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
    finger_cos = calc_finger_cos(debug_image, hand_landmarks)
    if hand_sign_id == 2 : 
        if finger_cos < soft_cos_limit :   # 折れてない
            return True
        else : 
            return False
    else :   # hand_sign_id != 2
        # 折れているかを判定
        if finger_cos > hard_cos_limit :   # この時折れている
            return False
        else : 
            return True


# hand_sign_id != 2 または 指が折れていたらFalse、他はTrue
def judge_point_finger_hard(debug_image, hand_landmarks, pre_processed_landmark_list) : 

    # まずhand_sign_id を確認
    keypoint_classifier = KeyPointClassifier()
    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
    finger_cos = calc_finger_cos(debug_image, hand_landmarks)
    #print("ID : " + str(hand_sign_id))
    if hand_sign_id != 2 : 
        return False
    else :   # hand_sign_id != 2
        # 折れているかを判定
        if finger_cos > hard_cos_limit :   # この時折れている
            return False
        else : 
            return True