from Preprocessing import *
import sys
sys.path.append('/home/hwang-gyuhan/Workspace/DenseNet/Testcode')


# Preprocessing 객체 생성
preprocessor = Preprocessing(sr=16000)

# MP3 파일 경로
mp3_file = "/home/hwang-gyuhan/Workspace/DenseNet/TestCode/song/Rock0a9SuLLrJg88CfDMlGse8v.mp3"

# 오디오 변환 처리
pcm_buffer = preprocessor.process_audio(mp3_file)

# PCM 데이터 읽기 및 확인
if pcm_buffer:
    pcm_data = pcm_buffer.read()
    print(f"PCM Data Size: {len(pcm_data)} bytes")