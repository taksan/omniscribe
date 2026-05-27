import whisper

model = whisper.load_model("base")
result = model.transcribe("meetings/meeting-20260527-103422.wav")
print(result["text"])

