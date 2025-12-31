"""
Text-to-Speech Manager for Blind Path Detection System
"""

import pyttsx3
from typing import Optional
from config import LANGUAGES


class TTSManager:
    """Text-to-Speech manager"""

    def __init__(self):
        self.engine = pyttsx3.init()
        self.set_default_settings()

        # Multi-language prompt texts
        self.prompts = {
            "en": {
                "clear": "Path is clear. Continue forward.",
                "partial": "Caution! Partially blocked ahead. Please adjust direction.",
                "full": "Danger! Path fully blocked. Stop immediately!",
                "uncertain": "Uncertain environment. Please proceed with caution.",
                "left": "Obstacle detected on right. Move to left.",
                "right": "Obstacle detected on left. Move to right.",
                "return_to_path": "Please return to the designated path.",
                "off_track": "You are off track. Please correct your direction."
            },
            "zh": {
                "clear": "道路通畅，请继续前行。",
                "partial": "警告！前方道路部分受阻，请调整方向。",
                "full": "危险！道路完全受阻，请立即停止！",
                "uncertain": "环境不确定，请谨慎前行。",
                "left": "右侧检测到障碍，请向左移动。",
                "right": "左侧检测到障碍，请向右移动。",
                "return_to_path": "请返回指定路径。",
                "off_track": "您已偏离路线，请调整方向。"
            },
            "ms": {
                "clear": "Laluan jelas. Teruskan perjalanan.",
                "partial": "Awas! Laluan separa tersekat. Sila laraskan arah.",
                "full": "Bahaya! Laluan tersekat sepenuhnya. Berhenti segera!",
                "uncertain": "Persekitaran tidak pasti. Sila berhati-hati.",
                "left": "Halangan dikesan di kanan. Bergerak ke kiri.",
                "right": "Halangan dikesan di kiri. Bergerak ke kanan."
            },
            "id": {
                "clear": "Jalan bersih. Lanjutkan perjalanan.",
                "partial": "Hati-hati! Jalan sebagian terhalang. Silakan sesuaikan arah.",
                "full": "Bahaya! Jalan sepenuhnya terhalang. Berhenti segera!",
                "uncertain": "Lingkungan tidak pasti. Harap berhati-hati.",
                "left": "Hambatan terdeteksi di kanan. Bergerak ke kiri.",
                "right": "Hambatan terdeteksi di kiri. Bergerak ke kanan."
            },
            "ar": {
                "clear": "الممر واضح. تابع التقدم.",
                "partial": "حذر! الممر مسدود جزئياً. يرجى تعديل الاتجاه.",
                "full": "خطر! الممر مسدود بالكامل. توقف فوراً!",
                "left": "تم الكشف عن عائق على اليمين. تحرك إلى اليسار.",
                "right": "تم الكشف عن عائق على اليسار. تحرك إلى اليمين."
            }
        }

    def set_default_settings(self):
        """Set default TTS parameters"""
        self.engine.setProperty('rate', 150)  # Speech rate
        self.engine.setProperty('volume', 0.9)  # Volume

    def set_language(self, language_code: str):
        """Set language"""
        # Try to find corresponding voice
        voices = self.engine.getProperty('voices')

        if language_code == "zh":
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    return
        elif language_code == "ms":
            # Malay usually uses English voice
            for voice in voices:
                if 'english' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    return
        elif language_code == "id":
            # Indonesian usually uses English voice
            for voice in voices:
                if 'english' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    return
        elif language_code == "ar":
            for voice in voices:
                if 'arabic' in voice.name.lower() or 'ar' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    return

        # Default to English voice
        for voice in voices:
            if 'english' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

    def speak(self, text: str, language: str = "en"):
        """Speak text"""
        try:
            self.set_language(language)
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

    def speak_prompt(self, prompt_key: str, language: str = "en"):
        """Speak predefined prompt"""
        if language in self.prompts and prompt_key in self.prompts[language]:
            self.speak(self.prompts[language][prompt_key], language)
        else:
            # Fallback to English
            if prompt_key in self.prompts["en"]:
                self.speak(self.prompts["en"][prompt_key], "en")