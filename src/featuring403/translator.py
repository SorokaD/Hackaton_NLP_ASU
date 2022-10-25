def translator(text):
    from googletrans import Translator
    translator1 = Translator()
    result = translator1.translate(text, src='ru', dest='en')
    return result
