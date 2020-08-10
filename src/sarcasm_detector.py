import engine


def get_result(text):
    result = engine.predict(text)
    if result == 1:
        return "Sarcasm"
    else:
        return "Not a Sarcasm"
