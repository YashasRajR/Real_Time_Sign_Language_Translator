"""Build readable text from recognised signs."""
from . import config


class Sentence:
    def __init__(self):
        self.text = ""

    def add(self, sign):
        if sign == "SPACE":
            self.space()
        elif sign == "DELETE":
            self.backspace()
        elif len(sign) == 1 and sign.isalpha():
            self.text += sign if not self.text.strip() else sign.lower()
        else:
            word = config.DISPLAY.get(sign, sign)
            if self.text and not self.text.endswith(" "):
                self.text += " "
            self.text += word + " "

    def space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def backspace(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""

    def __str__(self):
        return self.text
