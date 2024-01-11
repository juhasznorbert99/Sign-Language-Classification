import cv2
import numpy as np
import mediapipe as mp
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM, GRU, SimpleRNN

actions = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y'])

model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(1,63)))
model.add(LSTM(64, return_sequences=True, activation='sigmoid'))
model.add(GRU(64, return_sequences=True, activation='selu'))
model.add(GRU(32, return_sequences=True, activation='selu'))
model.add(LSTM(32, return_sequences=True, activation='selu'))
model.add(SimpleRNN(32, return_sequences=False, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('finalActionsWithFunctions.h5')


#
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
#
# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.4

#
def convertImage(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

#
def desenarePunteManaDreapta(image, results):
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )


#
def extragerePuncteCheie(results):
    allLandmarks = []
    rightHand = []
    if results.right_hand_landmarks != None:
        for res in results.right_hand_landmarks.landmark:
            test = np.array([res.x, res.y, res.z, res.visibility])
            allLandmarks.append(test)
        for i in allLandmarks:
            rightHand.append(i[0])
            rightHand.append(i[1])
            rightHand.append(i[2])

        rightHand = np.array(rightHand)
    else:
        rightHand = np.zeros(63)
    rightHand


    return np.concatenate([rightHand])

frequencyArray = np.zeros(24)
silabisire = False
counter = 0
string = ""

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic() as holistic:
    while cap.isOpened():
        if silabisire == True:
            counter += 1
        ret, frame = cap.read()

        image, results = convertImage(frame, holistic)

        desenarePunteManaDreapta(image, results)

        keypoints = extragerePuncteCheie(results)

        sequence.insert(0, keypoints)
        sequence = sequence[:1]
        if len(sequence) == 1:
            predictResult = model.predict(np.expand_dims(sequence, axis=0))[0]
            value = actions[np.argmax(predictResult)]
            position = np.argmax(predictResult)
            if np.count_nonzero(keypoints == 0) != 63:
                if silabisire == False:
                    cv2.putText(image, str(value.upper()), (120, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                else:
                    if counter == 22:
                        string += actions[np.argmax(frequencyArray)]
                        frequencyArray = np.zeros(24)
                        counter = 0
                    else:
                        frequencyArray[position] += 1

                    cv2.putText(image, string.upper(), (120, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow('Licenta', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        else:
            if cv2.waitKey(10) & 0xFF == ord('s'):
                counter = 0
                string = ""
                silabisire = True
            if cv2.waitKey(10) & 0xFF == ord('l'):
                silabisire = False
    cap.release()
    cv2.destroyAllWindows()