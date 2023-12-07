"""Author: Eng.Rawan Khaled Mohamed"""

import pickle
import nltk
import speech_recognition as sr
from googletrans import Translator
import cv2
import numpy as np
import tensorflow.keras as keras
from matplotlib import pyplot


model2 = keras.models.load_model('model_saved3.h5')

mapper = {

    0: "positive",
    1: "negative",
    2: "neutral"
}

class recommend:
    def __init__(self):
        self.recomm = 0
        self.unrecomm = 0

    def set_recomm(self):
        self.recomm += 1
        print(f"Addition is successful. You added {1} to recomm")

    def set_unrecomm(self):
        self.unrecomm += 1
        print(f"Addition is successful. You added {1} to unrecomm")


# Create empty lists to store recommended and unrecommended products
lisrecomm = []
lisunrecomm = []

nltk.download('punkt')

# Load the saved model
with open('model_saved2 (1)', 'rb') as file:
    model = pickle.load(file)

# Load the saved vectorizer
with open('model_vector3', 'rb') as file:
    bow_counts = pickle.load(file)  # Load the saved vectorizer

recognizer = sr.Recognizer()

def translate_text(text, target_language='en'):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text


def Voice_Recognizer():
    global voice
    voice = ''
    try:
        with sr.Microphone() as source:
            # Start listening
            audio = recognizer.listen(source, timeout=2)
            # Recognize what you said using Google's speech recognition
            voice = recognizer.recognize_google(audio)
            # Print the recognized speech
            print("You said:", voice)

    except sr.WaitTimeoutError:
        print("Timeout error: No speech detected.")
        exit()
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        exit()
    except sr.UnknownValueError:
        print("Sorry, I could not understand what you said.")
        exit()

    return voice

def Voice_Recognizer2():
    global voice
    voice = ''
    try:
        with sr.Microphone() as source:
            # Start listening
            audio = recognizer.listen(source, timeout=2)
            text = recognizer.recognize_google(audio, language="ar")
            engtrans = translate_text(text, target_language='en')
            # Recognize what you said using Google's speech recognition
            # Print the recognized speech
            print("You said:", engtrans)

    except sr.WaitTimeoutError:
        print("Timeout error: No speech detected.")
        exit()
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        exit()
    except sr.UnknownValueError:
        print("Sorry, I could not understand what you said.")
        exit()

    return engtrans


def single_response(new_review, product, ID):
    # Fit the new review to the existing vocabulary before transforming
    new_review_bow = bow_counts.transform([new_review.lower()])  # Transform the new review
    predicted_sentiment = model.predict(new_review_bow)  # Make a prediction
    print("Predicted Sentiment:", predicted_sentiment)
    if predicted_sentiment[0] == "Positive":
        product.set_recomm()
        print("Thank you for your trust, your review sentiment will be added to the recommended section.")
        # Iterate through the products and call the compare function
        if product.recomm > product.unrecomm:
            if ID not in lisrecomm:
                lisrecomm.append(ID)
            if ID in lisunrecomm:
                lisunrecomm.remove(ID)

    elif predicted_sentiment[0] == "Negative":
        product.set_unrecomm()
        print(
            "Sorry to hear that, your review sentiment will be added to the critizing reviews section, we hope that your problem get solved soon.")
        if product.recomm < product.unrecomm:
            if ID not in lisunrecomm:
                lisunrecomm.append(ID)
            if ID in lisrecomm:
                lisrecomm.remove(ID)

def image_analysis(img):
    # Load the original image
    original_image = cv2.imread(img)

    # Display the original image
    fig, axes = pyplot.subplots(1, 2, figsize=(8, 4), dpi=100)
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Load and process the image for analysis
    input_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (48, 48))
    input_image = input_image.astype('float32') / 255.0
    input_image = input_image.reshape(48, 48, 1)

    # Make predictions for the input image
    prediction = model2.predict(input_image.reshape(1, 48, 48, 1))[0]
    predicted = np.argmax(prediction)

    # Display the processed image with prediction
    axes[1].imshow(input_image[:, :, 0], cmap='gray')
    axes[1].set_title(f'Prediction: {mapper[predicted]}')
    axes[1].axis('off')

    pyplot.show()

    print(f"Original Image: {img}")
    print(f"Prediction: {mapper[predicted]}")


def perform_video_analysis():
    # Load frontalface haar cascade
    facecascade = cv2.CascadeClassifier("raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_default.xml")

    def video_analysis(frame, frontalface):
        # Convert the frame to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        face_Detect = frontalface.detectMultiScale(grayscaled_frame, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in face_Detect:
            # Extract ROI for sentiment analysis
            face_roi = grayscaled_frame[y:y + h, x:x + w]

            if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                # Preprocess the face_roi for sentiment analysis
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi.astype('float32') / 255.0
                face_roi = face_roi.reshape(1, 48, 48, 1)

                # Make predictions for the face_roi
                prediction = model2.predict(face_roi)[0]
                predicted = np.argmax(prediction)

                # Determine frame color based on sentiment prediction
                if mapper[predicted] == "positive":
                    frame_color = (0, 255, 0)  # Green
                elif mapper[predicted] == "negative":
                    frame_color = (0, 0, 255)  # Red
                else:
                    frame_color = (255, 0, 0)  # Blue

                # Display the face with sentiment prediction and colored frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), frame_color, 2)
                cv2.putText(frame, f'Sentiment: {mapper[predicted]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, frame_color, 2)

        # Display the original frame
        cv2.imshow('Video for face detection and sentiment analysis', frame)

    # Start the video capture
    capture_on_video = cv2.VideoCapture(0)

    while True:
        _, frame = capture_on_video.read()

        # Perform face detection and sentiment analysis on the current frame
        video_analysis(frame, facecascade)

        # If the user presses q on the keyboard to quit the video, terminate
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    capture_on_video.release()
    cv2.destroyAllWindows()

def main(productt):
    global ID
    print("Welcome To sentiment assistant chatbot")

    while True:
        print("1.Written review")
        print("2.Recorded review")
        print("3.Recommended products")
        print("4.Unrecommended products")
        print("5.Image analysis")
        print("6.Real Time video analysis")
        print("7.Exit Chat bot")
        n = int(input("Please choose a choice: "))

        if n == 1:
            print("Please select a language")
            print("1.Arabic")
            print("2.English")
            m = int(input("Please select a number"))
            if m == 1:
                I = input("Please, Enter product name: ")
                ID = translate_text(I, target_language='en')
            elif m == 2:
                ID = input("Please, Enter product name: ")

            # Check if the ID already exists in the dictionary
            if ID not in productt:
                productt[ID] = {
                    "ATM": recommend()  # Create a new reccomend instance for each product
                }

            rec = productt[ID]["ATM"]  # Retrieve the user's reccomend_Function instance
            if m == 1:
                new_review = input("Please enter the written review: ")
                engtrans = translate_text(new_review, target_language='en')
                single_response(engtrans.lower(), rec, ID)
            elif m == 2:
                new_review = input("Please enter the written review: ")
                single_response(new_review.lower(), rec, ID)

        elif n == 2:
            print("Please select a language")
            print("1.Arabic")
            print("2.English")
            m = int(input("Please select a number: "))

            if m == 1:
                I = input("Please, Enter product name: ")
                ID = translate_text(I, target_language='en')
            elif m == 2:
                ID = input("Please, Enter product name: ")

            # Check if the ID already exists in the dictionary
            if ID not in productt:
                productt[ID] = {
                    "ATM": recommend()  # Create a new reccomend instance for each product
                }

            rec = productt[ID]["ATM"]  # Retrieve the user's reccomend_Function instance

            print("Sentiment Assistant is here. How can I help you?")
            print("Say something...")
            if m == 1:
                new_review = Voice_Recognizer2()
                voice_audio = new_review.lower()
                single_response(voice_audio, rec, ID)
            elif m == 2:
                new_review = Voice_Recognizer()
                voice_audio = new_review.lower()
                single_response(voice_audio, rec, ID)

        elif n == 3:
            print(lisrecomm)
        elif n == 4:
            print(lisunrecomm)
        elif n == 5:
            input_img_path = "Rwan.png"
            image_analysis(input_img_path)
        elif n == 6:
            # Call the function to perform video analysis
            perform_video_analysis()
        elif n == 7:
            print("Thank you for using sentiment assistant, See you Soon, Good Bye!")
            exit()
        else:
            print("Sorry you chose something out of range.")


if __name__ == "__main__":
    productt = {}  # Dictionary to store product and associated recomm and unrecomm
    while True:
        try:
            main(productt)
        except KeyboardInterrupt:
            print("\nExiting...!")
            break
