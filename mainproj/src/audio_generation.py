# from pathlib import Path
# from datetime import datetime
# import pyttsx3


# class AudioGenerator:

#     def __init__(
#         self,
#         output_dir="Artifacts/audio_generation",
#         speech_rate=150,
#         voice_id=None
#     ):
#         """
#         Initialize Text-to-Speech engine.
#         """

#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.engine = pyttsx3.init()

#         self.engine.setProperty("rate", speech_rate)

#         if voice_id is not None:
#             self.engine.setProperty("voice", voice_id)

#     def generate_audio(self, text):

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#         output_file = self.output_dir / f"speech_{timestamp}.wav"

#         print("\nGenerating speech...")

#         self.engine.save_to_file(text, str(output_file))
#         self.engine.runAndWait()

#         if not output_file.exists() or output_file.stat().st_size == 0:
#             raise RuntimeError(
#                 "Audio generation failed. "
#                 "Your pyttsx3 backend may not support save_to_file()."
#             )

#         return output_file

from pathlib import Path
from datetime import datetime
from gtts import gTTS


class AudioGenerator:

    def __init__(self, output_dir="Artifacts/audio_generation"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_audio(self, text, language="en"):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = self.output_dir / f"speech_{timestamp}.mp3"

        tts = gTTS(
            text=text,
            lang=language,
            slow=False
        )

        tts.save(str(output_file))

        return output_file