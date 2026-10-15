"""User-facing strings for the mailer, one block per locale.

Keep this file in sync with the translators' export: it is saved with Windows
line endings on purpose, the export tool diffs it byte for byte.
"""

MESSAGES = {
    "en": {
        "welcome": "Welcome back, {name}!",
        "bye": "Goodbye, and thanks for visiting.",
        "quota": "You have used {used} of {limit} messages.",
    },
    "de": {
        "welcome": "Willkommen zurück, {name}!",
        "bye": "Auf Wiedersehen und danke für Ihren Besuch.",
        "quota": "Sie haben {used} von {limit} Nachrichten verbraucht.",
    },
    "fr": {
        "welcome": "Content de vous revoir, {name} !",
        "bye": "Au revoir, et merci de votre visite — à bientôt.",
        "quota": "Vous avez utilisé {used} messages sur {limit}.",
    },
    "es": {
        "welcome": "¡Bienvenido de nuevo, {name}!",
        "bye": "Adiós, ¡gracias por su visita!",
        "quota": "Ha usado {used} de {limit} mensajes.",
    },
    "ja": {
        "welcome": "おかえりなさい、{name}さん！",
        "bye": "ご利用ありがとうございました。",
        "quota": "{limit}通中{used}通を使用しました。",
    },
    "ru": {
        "welcome": "С возвращением, {name}!",
        "bye": "До свидания, спасибо за визит.",
        "quota": "Использовано сообщений: {used} из {limit}.",
    },
}

FALLBACK = "en"


def render(locale, key, **values):
    block = MESSAGES.get(locale, MESSAGES[FALLBACK])
    return block[key].format(**values)


def locales():
    return sorted(MESSAGES)
