from flask import Flask, request, render_template
import app
import webbrowser
import pyautogui

ui = Flask(__name__)

command_dict = {10 : "no_command"}

@ui.route("/",methods=['GET'])
def index():
    return render_template("index.html")

@ui.route('/run', methods=["GET", "POST"])
def run():
    # 現在のコマンドを取得
    for i in range(10) : 
        command_dict[i] = request.args.get("symbol_" + str(i), type = str)

    loop()
    return render_template("index.html")

@ui.route('/update', methods=["GET", "POST"])
def update() : 
    for i in range(10) : 
        command_dict[i] = request.args.get("symbol_" + str(i), type = str)

def loop() :
    continue_loop = True
    while continue_loop:
        result = app.main()
        print(result, command_dict[result])
        print()
        continue_loop = test(command_dict[result])
    
def test(application):
    if(application == "youtube"):
        webbrowser.open("https://www.youtube.com")
        return True
    elif(application == "google"):
        webbrowser.open("https://www.google.com")
        return True
    elif(application == "volumeup"):
        pyautogui.press("volumeup")
        return True
    elif(application == "volumedown"):
        pyautogui.press("volumedown")
        return True
    elif(application == "nextDesktop"):
        pyautogui.hotkey("winleft", "ctrl", "right")
        return True
    elif(application == "prevDesktop"):
        pyautogui.hotkey("winleft", "ctrl", "left")
        return True
    #スクショはMacの設定で許可しないとちゃんと映らない
    elif(application == "screenshot"):
        sc = pyautogui.screenshot()
        sc.save("screenshot.png")
        return True
    elif(application == "mouseup"):
        pyautogui.moveRel(0,-30)
        return True
    elif(application == "mousedown"):
        pyautogui.moveRel(0,30)
        return True
    elif(application == "mouseleft"):
        pyautogui.moveRel(-30,0)
        return True
    elif(application == "mousedown"):
        pyautogui.moveRel(30,0)
        return True
    elif(application == "mouseclick"):
        pyautogui.click()
        return True
    elif(application =="f11"):
        pyautogui.press("f11")
        return True
    elif(application =="spotlight"):
        pyautogui.hotkey("command","space")
        return True
    elif(application == "finish"):
        return False
    else:
        return True
        

if __name__ == "__main__":
    ui.run(debug=True, port=5000)
