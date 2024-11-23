import os
import soundfile as sf

# FMA 데이터셋 경로 및 장르 설정
base_dir = "/home/hwang-gyuhan/Workspace/dataset"
input_dir = os.path.join(base_dir, "fma_wav")
output_dir = os.path.join(base_dir, "fma_wav_pcm")

genres = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

# 변환 작업 시작
i = 0
total_files = sum(len(os.listdir(os.path.join(input_dir, g))) for g in genres)  # 전체 파일 개수 계산

for genre in genres:
    genre_input_dir = os.path.join(input_dir, genre)
    genre_output_dir = os.path.join(output_dir, genre)
    
    # 출력 폴더가 없으면 생성
    if not os.path.exists(genre_output_dir):
        os.makedirs(genre_output_dir)
    
    # 장르별 파일 변환
    for filename in os.listdir(genre_input_dir):
        # 입력 파일 경로 및 출력 파일 경로 생성
        input_file = os.path.join(genre_input_dir, filename)
        output_file = os.path.join(genre_output_dir, filename.split(".")[0] + "_PCM16.wav")
        
        # 오디오 파일 읽기 및 변환
        try:
            data, samplerate = sf.read(input_file)
            sf.write(output_file, data, samplerate, subtype='PCM_16')
            
            # 진행 상황 출력
            print(f"[{genre}] {filename} 변환 완료 ({i+1}/{total_files})")
            i += 1
        except Exception as e:
            print(f"파일 변환 실패: {input_file}, 오류: {e}")

print("\n모든 파일 변환 완료!")