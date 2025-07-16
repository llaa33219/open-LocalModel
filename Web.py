#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Web.py - 웹 관리 및 실행 파일
open-LocalModel: 허깅페이스 Transformers를 이용한 로컬 AI 모델 인터페이스
"""

import os
import json
from typing import Dict, List, Optional, Any
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, Response, stream_with_context
from AI_model import AIModelManager
from Chat import ChatManager

# 설정 관리 클래스
class SettingsManager:
    """설정 관리 클래스"""
    
    def __init__(self, settings_dir: str = "Setting-ai"):
        """
        초기화 함수
        
        Args:
            settings_dir: 설정 저장 디렉토리
        """
        self.settings_dir = settings_dir
        self.current_settings = self._get_default_settings()
        
        # 설정 디렉토리 생성
        if not os.path.exists(settings_dir):
            os.makedirs(settings_dir)
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """
        기본 설정 가져오기
        
        Returns:
            기본 설정
        """
        return {
            "max_input_tokens": 4096,
            "max_output_tokens": 2048,
            "max_total_tokens": 8192,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "show_thinking": False,
            "device_type": "gpu",  # 기본값은 GPU
            "use_quantization": True,  # 기본적으로 양자화 사용
            "quantization_bits": 8  # 기본 8비트 양자화
        }
    
    def save_settings(self, name: str, settings: Dict[str, Any]) -> bool:
        """
        설정 저장
        
        Args:
            name: 설정 이름
            settings: 설정 내용
            
        Returns:
            저장 성공 여부
        """
        try:
            settings_path = os.path.join(self.settings_dir, f"{name}.json")
            
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"설정 저장 중 오류 발생: {e}")
            return False
    
    def load_settings(self, name: str) -> Optional[Dict[str, Any]]:
        """
        설정 로드
        
        Args:
            name: 설정 이름
            
        Returns:
            설정 내용 또는 None
        """
        try:
            settings_path = os.path.join(self.settings_dir, f"{name}.json")
            
            if not os.path.exists(settings_path):
                return None
            
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            self.current_settings = settings
            return settings
        except Exception as e:
            print(f"설정 로드 중 오류 발생: {e}")
            return None
    
    def get_settings_list(self) -> List[str]:
        """
        설정 목록 가져오기
        
        Returns:
            설정 이름 목록
        """
        try:
            settings_files = [f.replace('.json', '') for f in os.listdir(self.settings_dir) if f.endswith('.json')]
            return settings_files
        except Exception as e:
            print(f"설정 목록 가져오기 중 오류 발생: {e}")
            return []
    
    def get_current_settings(self) -> Dict[str, Any]:
        """
        현재 설정 가져오기
        
        Returns:
            현재 설정
        """
        return self.current_settings
    
    def update_current_settings(self, settings: Dict[str, Any]) -> None:
        """
        현재 설정 업데이트
        
        Args:
            settings: 업데이트할 설정
        """
        self.current_settings.update(settings)

# Flask 애플리케이션 생성 함수
def create_app():
    """
    Flask 애플리케이션 생성
    
    Returns:
        Flask 애플리케이션
    """
    app = Flask(__name__, static_url_path='')
    
    # 인스턴스 생성
    model_manager = AIModelManager()
    chat_manager = ChatManager()
    settings_manager = SettingsManager()
    
    # 라우트 정의
    @app.route('/')
    def index():
        return send_file('index.html')
    
    @app.route('/style.css')
    def serve_css():
        return send_file('style.css')
    
    @app.route('/script.js')
    def serve_js():
        return send_file('script.js')
    
    @app.route('/api/models/available', methods=['GET'])
    def get_available_models():
        filter_text = request.args.get('filter', '')
        limit = int(request.args.get('limit', 50))
        author = request.args.get('author', '')
        sort_by = request.args.get('sort_by', 'downloads')
        
        models = model_manager.get_available_models(
            filter_text=filter_text,
            limit=limit,
            author=author,
            sort_by=sort_by
        )
        return jsonify(models)
    
    @app.route('/api/models/installed', methods=['GET'])
    def get_installed_models():
        models = model_manager.get_installed_models()
        return jsonify(models)
    
    @app.route('/api/models/install', methods=['POST'])
    def install_model():
        data = request.json
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({"success": False, "error": "모델 ID가 필요합니다."}), 400
        
        success = model_manager.install_model(model_id)
        return jsonify({"success": success})
    
    @app.route('/api/models/remove', methods=['POST'])
    def remove_model():
        data = request.json
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({"success": False, "error": "모델 ID가 필요합니다."}), 400
        
        success = model_manager.remove_model(model_id)
        return jsonify({"success": success})
    
    @app.route('/api/models/settings', methods=['GET'])
    def get_model_settings():
        model_id = request.args.get('model_id')
        
        if not model_id:
            return jsonify({"success": False, "error": "모델 ID가 필요합니다."}), 400
        
        settings = model_manager.get_model_settings(model_id)
        return jsonify({"success": True, "settings": settings})
    
    @app.route('/api/models/load', methods=['POST'])
    def load_model():
        data = request.json
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({"success": False, "error": "모델 ID가 필요합니다."}), 400
        
        try:
            # 현재 설정을 가져와서 디바이스 정보 전달
            current_settings = settings_manager.get_current_settings()
            
            # model_manager.load_model 호출 시 설정 추가
            result = model_manager.load_model(model_id, current_settings)
            
            # result가 튜플인 경우 (오류 메시지가 있는 경우)
            if isinstance(result, tuple):
                success, error_message = result
                
                # GPU 메모리 부족 오류의 경우 클라이언트에게 특별 플래그 전달
                if success and "GPU 메모리 부족" in error_message:
                    # CPU로 전환되었다는 메시지가 있으면, 설정도 업데이트
                    current_settings["device_type"] = "cpu"
                    settings_manager.update_current_settings(current_settings)
                    return jsonify({
                        "success": True, 
                        "settings": current_settings,
                        "gpu_memory_error": True,
                        "message": error_message
                    })
                elif not success and "GPU 메모리 부족" in error_message:
                    return jsonify({
                        "success": False, 
                        "error": error_message,
                        "gpu_memory_error": True
                    })
                else:
                    return jsonify({"success": success, "error": error_message})
            
            success = result
            
            # 모델 로드 성공 시 모델의 기본 설정을 현재 설정으로 설정
            if success:
                model_settings = model_manager.get_model_settings(model_id)
                
                # 현재 디바이스 설정 유지
                model_settings["device_type"] = current_settings.get("device_type", "gpu")
                model_settings["use_quantization"] = current_settings.get("use_quantization", True)
                model_settings["quantization_bits"] = current_settings.get("quantization_bits", 8)
                
                settings_manager.update_current_settings(model_settings)
                return jsonify({"success": success, "settings": model_settings})
            else:
                return jsonify({"success": False, "error": "모델 로드에 실패했습니다."})
                
        except Exception as e:
            error_message = str(e)
            # Transformers 버전 관련 오류 메시지 처리
            if "does not recognize this architecture" in error_message or "Transformers" in error_message:
                return jsonify({
                    "success": False, 
                    "error": "모델 아키텍처가 현재 Transformers 버전에서 지원되지 않습니다. 다음 명령어로 업데이트하세요: pip install --upgrade transformers"
                })
            # GPU 메모리 부족 오류 처리
            elif "CUDA out of memory" in error_message or "GPU 메모리 부족" in error_message:
                return jsonify({
                    "success": False, 
                    "error": "GPU 메모리 부족: CPU 모드로 전환하거나 더 작은 모델을 사용하세요.",
                    "gpu_memory_error": True
                })
            return jsonify({"success": False, "error": f"모델 로드 중 오류: {error_message}"})
    
    @app.route('/api/chat/new', methods=['POST'])
    def create_new_chat():
        chat_id = chat_manager.create_new_chat()
        return jsonify({"chat_id": chat_id})
    
    @app.route('/api/chat/history', methods=['GET'])
    def get_chat_history():
        history = chat_manager.get_chat_history()
        return jsonify(history)
    
    @app.route('/api/chat/load', methods=['POST'])
    def load_chat():
        data = request.json
        chat_id = data.get('chat_id')
        
        if not chat_id:
            return jsonify({"success": False, "error": "채팅 ID가 필요합니다."}), 400
        
        success = chat_manager.load_chat(chat_id)
        
        if success:
            return jsonify({"success": True, "messages": chat_manager.get_current_chat()})
        else:
            return jsonify({"success": False, "error": "채팅을 로드할 수 없습니다."}), 400
    
    @app.route('/api/chat/delete', methods=['POST'])
    def delete_chat():
        data = request.json
        chat_id = data.get('chat_id')
        
        if not chat_id:
            return jsonify({"success": False, "error": "채팅 ID가 필요합니다."}), 400
        
        success = chat_manager.delete_chat(chat_id)
        return jsonify({"success": success})
    
    @app.route('/api/chat/message', methods=['POST'])
    def send_message():
        data = request.json
        message = data.get('message')
        
        if not message:
            return jsonify({"success": False, "error": "메시지가 필요합니다."}), 400
        
        # 사용자 메시지 추가
        chat_manager.add_message("user", message)
        
        # 스트리밍 응답을 위한 제너레이터 함수
        def generate():
            # 응답 시작을 알림
            yield json.dumps({"type": "start"}) + "\n"
            
            # AI 응답 생성
            settings = settings_manager.get_current_settings()
            
            # 스트리밍 모드 활성화
            settings["streaming"] = True
            
            for token in model_manager.generate_text_streaming(message, settings):
                # 토큰이 딕셔너리인 경우 (thinking 시작/종료 등 특수 이벤트)
                if isinstance(token, dict):
                    yield json.dumps({"type": "event", "data": token}) + "\n"
                else:
                    # 일반 텍스트 토큰
                    yield json.dumps({"type": "token", "data": token}) + "\n"
            
            # 전체 응답 가져오기 (채팅 기록에 저장하기 위함)
            complete_response = model_manager.get_last_complete_response()
            
            # thinking 부분이 있는지 확인
            if isinstance(complete_response, dict) and "thinking" in complete_response and "response" in complete_response:
                thinking_part = complete_response["thinking"]
                response_part = complete_response["response"]
                
                # 채팅에 추가
                chat_manager.add_message("assistant", response_part, thinking_part)
                
                # 응답 완료 이벤트 전송
                yield json.dumps({
                    "type": "complete", 
                    "full_response": response_part,
                    "thinking": thinking_part
                }) + "\n"
            else:
                # 추론 과정이 없는 경우
                response_text = complete_response if isinstance(complete_response, str) else complete_response.get("response", "")
                chat_manager.add_message("assistant", response_text)
                
                # 응답 완료 이벤트 전송
                yield json.dumps({
                    "type": "complete", 
                    "full_response": response_text
                }) + "\n"
        
        # 스트리밍 응답 반환
        return Response(stream_with_context(generate()), 
                      mimetype='text/event-stream',
                      headers={'Cache-Control': 'no-cache', 
                              'X-Accel-Buffering': 'no'})
    
    @app.route('/api/settings/list', methods=['GET'])
    def get_settings_list():
        settings_list = settings_manager.get_settings_list()
        return jsonify(settings_list)
    
    @app.route('/api/settings/current', methods=['GET'])
    def get_current_settings():
        settings = settings_manager.get_current_settings()
        return jsonify(settings)
    
    @app.route('/api/settings/save', methods=['POST'])
    def save_settings():
        data = request.json
        name = data.get('name')
        settings = data.get('settings')
        
        if not name or not settings:
            return jsonify({"success": False, "error": "이름과 설정이 필요합니다."}), 400
        
        success = settings_manager.save_settings(name, settings)
        return jsonify({"success": success})
    
    @app.route('/api/settings/load', methods=['POST'])
    def load_settings():
        data = request.json
        name = data.get('name')
        
        if not name:
            return jsonify({"success": False, "error": "설정 이름이 필요합니다."}), 400
        
        settings = settings_manager.load_settings(name)
        
        if settings:
            return jsonify({"success": True, "settings": settings})
        else:
            return jsonify({"success": False, "error": "설정을 로드할 수 없습니다."}), 400
    
    @app.route('/api/settings/update', methods=['POST'])
    def update_settings():
        data = request.json
        settings = data.get('settings')
        
        if not settings:
            return jsonify({"success": False, "error": "설정이 필요합니다."}), 400
        
        settings_manager.update_current_settings(settings)
        return jsonify({"success": True})
    
    return app