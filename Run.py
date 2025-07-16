#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run.py - 사용자가 실행하는 메인 파일
허깅페이스 Transformers를 이용한 로컬 AI 웹 인터페이스
"""

import os
import sys
from Web import create_app

def main():
    """메인 함수: 웹 애플리케이션 실행"""
    print("허깅페이스 Transformers 로컬 AI 웹 인터페이스를 시작합니다...")
    
    # 필요한 폴더 확인 및 생성
    for folder in ['Chat', 'Setting-ai', 'models']:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"{folder} 폴더를 생성했습니다.")
    
    # GPU 사용 가능 여부 확인
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB 단위로 변환
            print(f"GPU 감지됨: {device_name} (메모리: {gpu_memory:.2f} GB)")
        else:
            print("GPU가 감지되지 않았습니다. CPU 모드로 실행됩니다.")
    except Exception as e:
        print(f"GPU 확인 중 오류 발생: {e}")
        print("CPU 모드로 실행됩니다.")
    
    # 웹 애플리케이션 실행
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()