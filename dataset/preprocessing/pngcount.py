import os

def count_png_files(directory):
    png_count = 0
    
    # 디렉토리 내의 모든 파일을 탐색
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                png_count += 1
    
    return png_count

# 여기에서 경로를 지정하세요
directory_path = '/home/hwang-gyuhan/Workspace/dataset/feature_image_all/chroma_cens/'  # PNG 파일이 있는 디렉토리 경로

png_file_count = count_png_files(directory_path)

print(f"'{directory_path}' 디렉토리에 있는 PNG 파일 개수: {png_file_count}")