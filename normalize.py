def normalize(place_list, ratio = 1.0) : 
    x_list = []
    y_list = []
    for place in place_list : 
        x_list.append(place[0])
        y_list.append(place[1])
    
    max_x = max(x_list)
    min_x = min(x_list)
    max_y = max(y_list)
    min_y = min(y_list)
    range_x = max_x - min_x
    range_y = max_y - min_y

    #上下左右に5%ずつ余裕を持たせる
    left = min_x - range_x * 0.05
    right = max_x + range_x * 0.05
    down = min_y - range_y * 0.05
    up = max_y + range_y * 0.05

    #中央を計算する
    yoko_middle = (left + right) / 2
    tate_middle = (down + up) / 2

    #縦 : 横 = ratio : 1 に直す
    width = right - left
    height = up - down

    if(height > width * ratio) :   #縦が予定より大きい時
        #横方向を修正する
        width = height / ratio   #widthを修正
        left = yoko_middle - width / 2
        right = yoko_middle + width / 2
    else : 
        #縦方向を修正する
        height = width * ratio
        down = tate_middle - height / 2
        up = tate_middle + height / 2

    yoko = (left, right)
    tate = (down, up)

    return (yoko, tate)



