import speech_recognition as sr

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
            return audio

    def recognize(self, audio):
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return None

if __name__ == '__main__':
    assistant = VoiceAssistant()
    audio_data = assistant.listen()
    recognized_text = assistant.recognize(audio_data)
    if recognized_text:
        print("You said:", recognized_text)