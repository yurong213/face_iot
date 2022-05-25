import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()         # 啟用訓練人臉模型方法
recognizer.read('face.yml')                               # 讀取人臉模型檔
cascade_path = "haarcascade_frontalface_default.xml"  # 載入人臉追蹤模型
face_cascade = cv2.CascadeClassifier(cascade_path)        # 啟用人臉追蹤

cap = cv2.VideoCapture(2)                                 # 開啟攝影機
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    img = cv2.resize(img, (540, 300))              # 縮小尺寸，加快辨識效率
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換成黑白
    faces = face_cascade.detectMultiScale(gray)  # 追蹤人臉 ( 目的在於標記出外框 )

    # 建立姓名和 id 的對照表
    name = {'1': 'Alejandro_Toledo', '2': 'Alvaro_Uribe', '3': 'Amelie_Mauresmo', '4': 'Andre_Agassi', '5': 'Andy_Roddick', '6': 'Angelina_Jolie', '7': 'Ariel_Sharon', '8': 'Arnold_Schwarzenegger', '9': 'Atal_Bihari_Vajpayee', '10': 'David_Beckham', '11': 'Donald_Rumsfeld', '12': 'George_Robertson', '13': 'George_W_Bush', '14': 'Gerhard_Schroeder', '15': 'Gloria_Macapagal_Arroyo', '16': 'Gray_Davis', '17': 'Guillermo_Coria', '18': 'Hamid_Karzai', '19': 'Hans_Blix', '20': 'Hugo_Chavez', '21': 'Jack_Straw', '22': 'Jacques_Chirac', '23': 'Jean_Chretien', '24': 'Jennifer_Aniston', '25': 'Jennifer_Lopez', '26': 'Jeremy_Greenstock', '27': 'Jiang_Zemin', '28': 'John_Ashcroft', '29': 'John_Bolton', '30': 'John_Negroponte', '31': 'John_Snow', '32': 'Jose_Maria_Aznar', '33': 'Juan_Carlos_Ferrero', '34': 'Junichiro_Koizumi', '35': 'Kofi_Annan', '36': 'Lance_Armstrong', '37': 'Laura_Bush', '38': 'Lindsay_Davenport', '39': 'Lleyton_Hewitt', '40': 'Luiz_Inacio_Lula_da_Silva', '41': 'Mahmoud_Abbas', '42': 'man_1', '43': 'man_10', '44': 'man_11', '45': 'man_2', '46': 'man_3', '47': 'man_4', '48': 'man_5', '49': 'man_6', '50': 'man_7', '51': 'man_8', '52': 'man_9', '53': 'Megawati_Sukarnoputri', '54': 'Michael_Schumacher', '55': 'Naomi_Watts', '56': 'Nestor_Kirchner', '57': 'Nicole_Kidman', '58': 'Paul_Bremer', '59': 'Pervez_Musharraf', '60': 'Pete_Sampras', '61': 'Recep_Tayyip_Erdogan', '62': 'Renee_Zellweger', '63': 'Ricardo_Lagos', '64': 'Richard_Myers', '65': 'Roh_Moo-hyun', '66': 'Rudolph_Giuliani', '67': 'Silvio_Berlusconi', '68': 'Tiger_Woods', '69': 'Tom_Daschle', '70': 'Tom_Ridge', '71': 'Tony_Blair', '72': 'Vicente_Fox', '73': 'Vladimir_Putin', '74': 'woman_1', '75': 'woman_2', '76': 'woman_3', '77': 'woman_4', '78': 'woman_5', '79': 'woman_6', '80': 'woman_7', '81': 'woman_8', '82': 'oxxostudio'}

    # 依序判斷每張臉屬於哪個 id
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)            # 標記人臉外框
        idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])  # 取出 id 號碼以及信心指數 confidence
        if confidence < 60:
            text = name[str(idnum)]                               # 如果信心指數小於 60，取得對應的名字
        else:
            text = '???'                                          # 不然名字就是 ???
        # 在人臉外框旁加上名字
        cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('oxxostudio', img)
    if cv2.waitKey(5) == ord('q'):
        break    # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()