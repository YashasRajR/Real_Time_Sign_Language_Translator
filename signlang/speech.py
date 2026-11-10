"""Offline text-to-speech (pyttsx3) that doesn't freeze the camera window."""
import threading


def speak(text):
    text = text.strip()
    if not text:
        return

    def run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:  # no speech engine installed
            print(f"Text-to-speech unavailable: {e}")

    threading.Thread(target=run, daemon=True).start()
