from detect_point_finger import detect_point_finger
import matplotlib.pyplot as plt
from normalize import normalize
import cv2
import numpy as np
from tensorflow.python.keras.models import load_model
## 追加 ##
from pytorch_simple import LitSimplenet
import torch, torchvision
import torch.nn.functional as F
## 追加 ##


def main():    
    # paramaters
    filename = "image/image.jpg"
    limit_len_finger_list = 10
    num_reduce_finger_list = 3
    limit_max_prob = 0.50   # 事後確率の最大値がlimit_max_probを超えない場合、何もしない
    
    # 意味不明なのでtensorflowの警告文を出ない様にする。出力って結構時間かかるしね
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 指の軌跡をJPGに保存する
    finger_list = detect_point_finger()
    finger_x = []
    finger_y = []
    
    # finger_listが少なすぎる場合、ミスって途中でやめたor誤認識である確率が高い
    # finger_listが規定値以下の時は強制終了
    try : 
        if len(finger_list) < limit_len_finger_list : 
            print()
            print("--------- もう少しゆっくり書いて!!!!!! ----------")
            print()
            return 10
    except : 
        return 10
    
    # finger_listの最後の幾つかは、終了サインを出そうとしたときに記録してしまったものである可能性が高い
    # 幾つか削ってもちゃんと認識できるはずなので、最後の方を削ってしまう
    finger_list = finger_list[ : -1 * num_reduce_finger_list]
    
    size = normalize(finger_list)
    yoko = size[0]
    tate = size[1]
    
    for i in range(len(finger_list)) : 
        finger_x.append(finger_list[i][0])
        finger_y.append(-1 * finger_list[i][1])
    
    plt.figure(figsize=(4, 4), dpi = 7, facecolor = "black")
    plt.axis("off")
    plt.scatter(finger_x, finger_y)
    plt.plot(finger_x, finger_y, lw = 30, color = "w")
    plt.axis([yoko[0], yoko[1], -1 * tate[1], -1 * tate[0]])   #ここに超注意する
    plt.savefig(filename)
    # 指の軌跡保存終了
    
    # 保存した画像から数字を予測する
    # モデルの読み込み
    model = LitSimplenet(lr=0.05)
    model_path = 'classify_model/fine_tuning.pt'
    model.load_state_dict(torch.load(model_path))
    
    # 画像の読み込み
    image = cv2.imread(filename, 0)   #グレースケール画像として読み込む.
    image = np.asarray(image)
    image = image.reshape(28, 28, 1)
    
    # transform
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, ), (0.5, ))
    ])
    input = torch.unsqueeze(test_transforms(image), 0)
    
    # モデルに入力
    logits = model(input)
    output = model.model(input)
    preds = torch.argmax(logits, dim=1)  # 予測した数字
    
    max_prob = torch.amax(F.softmax(output, dim=1), dim=1).item() # 事後確率の最大値
    
    if max_prob > limit_max_prob:
        print(" ")
        print("書いた数字は " +  str(preds.item()) + ", 確率は " + str(max_prob))
        print(" ")
        return int(preds.item())
    else:
        print("")
        print("書いた数字は " +  str(preds.item()) + ", 確率は " + str(max_prob))
        print("事後確率の最大値が規定値以下なので何もしません")
        print("")
        return int(10)
    
    
if __name__ == "__main__":
    main()