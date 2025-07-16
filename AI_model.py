#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI_model.py - AI 모델 설치 및 실행, 관리 파일
허깅페이스 Transformers를 이용한 로컬 AI 웹 인터페이스
"""

import os
import json
import shutil
from typing import Dict, List, Optional, Union, Any, Tuple
from huggingface_hub import HfApi, list_models, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

class AIModelManager:
    """AI 모델 관리 클래스"""
    
    def __init__(self, models_dir: str = "models"):
        """
        초기화 함수
        
        Args:
            models_dir: 모델 저장 디렉토리
        """
        self.models_dir = models_dir
        self.installed_models = {}
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_id = None
        self.hf_api = HfApi()
        
        # 모델 디렉토리 생성
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        # 설치된 모델 정보 로드
        self._load_installed_models()
    
    def _load_installed_models(self) -> None:
        """설치된 모델 정보 로드"""
        models_info_path = os.path.join(self.models_dir, "models_info.json")
        if os.path.exists(models_info_path):
            try:
                with open(models_info_path, 'r', encoding='utf-8') as f:
                    self.installed_models = json.load(f)
            except Exception as e:
                print(f"모델 정보 로드 중 오류 발생: {e}")
                self.installed_models = {}
    
    def _save_installed_models(self) -> None:
        """설치된 모델 정보 저장"""
        models_info_path = os.path.join(self.models_dir, "models_info.json")
        try:
            with open(models_info_path, 'w', encoding='utf-8') as f:
                json.dump(self.installed_models, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"모델 정보 저장 중 오류 발생: {e}")
    
    def get_available_models(self, filter_text: str = "", limit: int = 50, author: str = "", sort_by: str = "downloads") -> List[Dict[str, Any]]:
        """
        허깅페이스에서 사용 가능한 모델 목록 가져오기
        
        Args:
            filter_text: 필터링할 텍스트
            limit: 반환할 최대 모델 수
            author: 특정 제작자(작성자)로 필터링
            sort_by: 정렬 기준 ("downloads", "lastModified", "createdAt")
            
        Returns:
            모델 정보 목록
        """
        try:
            # 유효한 정렬 옵션 체크
            valid_sort_options = ["downloads", "lastModified", "createdAt"]
            if sort_by not in valid_sort_options:
                sort_by = "downloads"  # 기본값으로 설정

            # 지원하지 않는 아키텍처 목록
            unsupported_architectures = ["phi", "phi-msft", "mamba", "jamba", "persimmon"]
            
            # 필터 텍스트에 제작자 포함 여부 결정
            search_filter = filter_text
            
            # 제작자 필터가 있는 경우, 더 많은 모델을 가져와 필터링을 위한 여유 확보
            fetch_limit = limit * 3 if author else limit * 2
            
            # 텍스트 생성 모델만 필터링
            models = list_models(
                filter=search_filter,
                task="text-generation",
                limit=fetch_limit,
                sort=sort_by,
                direction=-1
            )
            
            filtered_models = []
            count = 0
            
            for model in models:
                # 지원하지 않는 아키텍처 제외
                if any(unsupp in model.id.lower() for unsupp in unsupported_architectures):
                    continue
                    
                # 모델 태그 체크 - 지원하지 않는 태그가 있으면 제외
                if model.tags and any(unsupp in model.tags for unsupp in unsupported_architectures):
                    continue
                
                # 모델 ID에서 제작자 추출
                model_author = model.id.split("/")[0] if "/" in model.id else "Unknown"
                
                # 제작자 필터링
                if author and not model_author.lower().startswith(author.lower()):
                    continue
                
                filtered_models.append({
                    "id": model.id,
                    "name": model.id.split("/")[-1],
                    "author": model_author,
                    "downloads": model.downloads,
                    "tags": model.tags,
                    "pipeline_tag": model.pipeline_tag
                })
                
                count += 1
                if count >= limit:
                    break
                    
            return filtered_models
        except Exception as e:
            print(f"모델 목록 가져오기 중 오류 발생: {e}")
            return []
    
    def install_model(self, model_id: str) -> bool:
        """
        모델 설치
        
        Args:
            model_id: 설치할 모델 ID
            
        Returns:
            설치 성공 여부
        """
        try:
            # 모델 디렉토리 경로
            model_dir = os.path.join(self.models_dir, model_id.replace("/", "_"))
            
            # 모델 다운로드
            snapshot_download(
                repo_id=model_id,
                local_dir=model_dir,
                ignore_patterns=["*.safetensors", "*.bin"] if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")) else None
            )
            
            # 추론 모델 여부 판단 키워드 확장
            instruction_keywords = ["instruct", "chat", "assistant", "dialog", "dialogue", "conversational", "supervised", "sft"]
            is_inference_model = any(keyword in model_id.lower() for keyword in instruction_keywords)
            
            # 모델 정보 저장
            self.installed_models[model_id] = {
                "path": model_dir,
                "name": model_id.split("/")[-1],
                "author": model_id.split("/")[0] if "/" in model_id else "Unknown",
                "is_inference_model": is_inference_model
            }
            
            # 기본 설정 가져오기 (모델을 실제로 로드하여 최대 토큰 수 확인)
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                max_model_tokens = tokenizer.model_max_length
                
                # 적절한 기본값 설정
                self.installed_models[model_id]["model_settings"] = {
                    "max_input_tokens": min(max_model_tokens // 2, 4096),  # 모델 최대 토큰의 절반이나 4096 중 작은 값
                    "max_output_tokens": min(max_model_tokens // 4, 2048),  # 모델 최대 토큰의 1/4이나 2048 중 작은 값
                    "max_total_tokens": max_model_tokens,  # 모델 최대 토큰
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
            except Exception as e:
                print(f"모델 토큰 제한 확인 중 오류: {e}, 기본값 사용")
                # 토큰 제한을 가져올 수 없는 경우 기본값 사용
                self.installed_models[model_id]["model_settings"] = self._get_default_settings()
            
            self._save_installed_models()
            return True
        except Exception as e:
            print(f"모델 설치 중 오류 발생: {e}")
            return False
    
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
    
    def get_model_settings(self, model_id: str) -> Dict[str, Any]:
        """
        특정 모델의 설정 가져오기
        
        Args:
            model_id: 모델 ID
            
        Returns:
            모델 설정
        """
        if model_id in self.installed_models:
            if "model_settings" not in self.installed_models[model_id]:
                # 이전에 설치된 모델에 설정이 없는 경우
                try:
                    # 모델을 잠시 로드하여 최대 토큰 수 확인
                    model_dir = self.installed_models[model_id]["path"]
                    tokenizer = AutoTokenizer.from_pretrained(model_dir)
                    max_model_tokens = tokenizer.model_max_length
                    
                    # 토큰 제한 설정
                    self.installed_models[model_id]["model_settings"] = {
                        "max_input_tokens": min(max_model_tokens // 2, 4096),
                        "max_output_tokens": min(max_model_tokens // 4, 2048),
                        "max_total_tokens": max_model_tokens,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 50,
                        "repetition_penalty": 1.1,
                        "do_sample": True,
                        "show_thinking": False,
                        "device_type": "gpu",
                        "use_quantization": True,
                        "quantization_bits": 8
                    }
                except Exception:
                    # 가져올 수 없으면 기본값 사용
                    self.installed_models[model_id]["model_settings"] = self._get_default_settings()
                
                self._save_installed_models()
            
            return self.installed_models[model_id]["model_settings"]
        
        # 모델이 설치되지 않은 경우 기본값 반환
        return self._get_default_settings()
    
    def update_model_settings(self, model_id: str, settings: Dict[str, Any]) -> bool:
        """
        모델 설정 업데이트
        
        Args:
            model_id: 모델 ID
            settings: 업데이트할 설정
            
        Returns:
            업데이트 성공 여부
        """
        if model_id in self.installed_models:
            if "model_settings" not in self.installed_models[model_id]:
                self.installed_models[model_id]["model_settings"] = self._get_default_settings()
            
            self.installed_models[model_id]["model_settings"].update(settings)
            self._save_installed_models()
            return True
        return False
    
    def remove_model(self, model_id: str) -> bool:
        """
        모델 제거
        
        Args:
            model_id: 제거할 모델 ID
            
        Returns:
            제거 성공 여부
        """
        try:
            if model_id in self.installed_models:
                model_dir = self.installed_models[model_id]["path"]
                
                # 모델 디렉토리 삭제
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                
                # 모델 정보에서 제거
                del self.installed_models[model_id]
                self._save_installed_models()
                
                # 현재 모델이 제거된 모델인 경우 초기화
                if self.current_model_id and model_id == self.current_model_id:
                    self.current_model = None
                    self.current_tokenizer = None
                    self.current_model_id = None
                
                return True
            return False
        except Exception as e:
            print(f"모델 제거 중 오류 발생: {e}")
            return False
    
    def get_installed_models(self) -> Dict[str, Dict[str, Any]]:
        """
        설치된 모델 목록 가져오기
        
        Returns:
            설치된 모델 정보
        """
        return self.installed_models
    
    def load_model(self, model_id: str, settings: Optional[Dict[str, Any]] = None) -> Union[bool, Tuple[bool, str]]:
        """
        모델 로드
        
        Args:
            model_id: 로드할 모델 ID
            settings: 모델 로드 설정 (선택 사항)
            
        Returns:
            로드 성공 여부 또는 (성공 여부, 오류 메시지) 튜플
        """
        try:
            if model_id in self.installed_models:
                model_path = self.installed_models[model_id]["path"]
                
                # 설정이 없는 경우 기본 설정 사용
                if settings is None:
                    settings = self.get_model_settings(model_id)
                
                # 디바이스 설정 가져오기
                device_type = settings.get("device_type", "gpu")
                use_quantization = settings.get("use_quantization", True)
                quantization_bits = settings.get("quantization_bits", 8)
                
                try:
                    # 먼저 토크나이저 로드 시도
                    self.current_tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
                    # 모델 아키텍처 감지 시도
                    try:
                        # config.json 파일에서 모델 아키텍처 확인
                        import os
                        import json
                        config_path = os.path.join(model_path, "config.json")
                        
                        if os.path.exists(config_path):
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                                model_type = config.get("model_type", "")
                                
                                # 특정 모델 타입 처리
                                if model_type == "phi-msft":
                                    error_msg = (
                                        f"'{model_type}' 모델 아키텍처는 현재 Transformers 버전에서 지원되지 않습니다.\n"
                                        "다음 명령어로 Transformers를 업데이트하세요:\n\n"
                                        "pip install --upgrade transformers\n\n"
                                        "또는 최신 개발 버전을 설치하세요:\n\n"
                                        "pip install git+https://github.com/huggingface/transformers.git\n\n"
                                        "업데이트 후 애플리케이션을 재시작하세요."
                                    )
                                    raise ValueError(error_msg)
                    except Exception as e:
                        print(f"모델 아키텍처 확인 중 오류: {e}")
                    
                    # 장치 설정 및 양자화 설정에 따른 모델 로드
                    import torch
                    
                    # 디바이스 설정
                    if device_type == "gpu" and torch.cuda.is_available():
                        device_map = "auto"
                        
                        # GPU 사용 시 dtype 설정
                        if torch.cuda.is_bf16_supported():
                            dtype = torch.bfloat16
                        else:
                            dtype = torch.float16
                            
                        # 양자화 설정 (GPU에서만 가능)
                        if use_quantization:
                            if quantization_bits == 4:
                                print("4비트 양자화로 모델 로드 시도...")
                                from transformers import BitsAndBytesConfig
                                quantization_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_use_double_quant=True
                                )
                            elif quantization_bits == 8:
                                print("8비트 양자화로 모델 로드 시도...")
                                from transformers import BitsAndBytesConfig
                                quantization_config = BitsAndBytesConfig(
                                    load_in_8bit=True,
                                    llm_int8_enable_fp32_cpu_offload=True
                                )
                            else:
                                quantization_config = None
                        else:
                            quantization_config = None
                    else:
                        # CPU 사용 시 설정
                        device_map = "cpu"
                        dtype = torch.float32
                        quantization_config = None
                        print("CPU로 모델 로드 시도...")
                    
                    # 모델 로드 시도
                    try:
                        # 양자화 설정이 있는 경우 사용
                        if quantization_config is not None:
                            self.current_model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                device_map=device_map,
                                torch_dtype=dtype,
                                quantization_config=quantization_config,
                                low_cpu_mem_usage=True
                            )
                        else:
                            # 양자화 없이 로드
                            self.current_model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                device_map=device_map,
                                torch_dtype=dtype,
                                low_cpu_mem_usage=True
                            )
                        
                        print(f"모델이 성공적으로 로드되었습니다. 장치: {device_type}, 양자화: {use_quantization}")
                        
                    except Exception as e:
                        print(f"첫 번째 로드 시도 실패: {e}")
                        
                        # GPU 메모리 부족 오류인 경우 CPU로 자동 전환
                        if "CUDA out of memory" in str(e) and device_type == "gpu":
                            print("GPU 메모리 부족, CPU로 전환 시도...")
                            
                            # CPU로 로드 시도
                            self.current_model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                device_map="cpu",
                                torch_dtype=torch.float32,
                                low_cpu_mem_usage=True
                            )
                            
                            return True, "GPU 메모리 부족으로 CPU로 전환되었습니다."
                        
                        # 다른 오류인 경우 가장 기본적인 설정으로 시도
                        try:
                            print("가장 기본적인 설정으로 로드 시도...")
                            self.current_model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                device_map="auto"
                            )
                        except Exception as e2:
                            # 아키텍처 관련 오류 확인
                            if "does not recognize this architecture" in str(e2):
                                error_msg = (
                                    f"모델 아키텍처가 현재 Transformers 버전에서 지원되지 않습니다.\n"
                                    "다음 명령어로 Transformers를 업데이트하세요:\n\n"
                                    "pip install --upgrade transformers\n\n"
                                    "또는 최신 개발 버전을 설치하세요:\n\n"
                                    "pip install git+https://github.com/huggingface/transformers.git\n\n"
                                    "업데이트 후 애플리케이션을 재시작하세요."
                                )
                                raise ValueError(error_msg)
                            else:
                                raise e2
                    
                    # 현재 로드된 모델 ID 저장
                    self.current_model_id = model_id
                    
                    # 모델의 최대 토큰 수 확인 및 업데이트
                    try:
                        max_model_tokens = self.current_tokenizer.model_max_length
                        
                        # 모델 설정에 최대 토큰 수 업데이트
                        if "model_settings" not in self.installed_models[model_id]:
                            self.installed_models[model_id]["model_settings"] = self._get_default_settings()
                        
                        self.installed_models[model_id]["model_settings"]["max_total_tokens"] = max_model_tokens
                        self.installed_models[model_id]["model_settings"]["max_input_tokens"] = min(max_model_tokens // 2, 4096)
                        self.installed_models[model_id]["model_settings"]["max_output_tokens"] = min(max_model_tokens // 4, 2048)
                        
                        # 디바이스 설정 저장
                        self.installed_models[model_id]["model_settings"]["device_type"] = device_type
                        self.installed_models[model_id]["model_settings"]["use_quantization"] = use_quantization
                        self.installed_models[model_id]["model_settings"]["quantization_bits"] = quantization_bits
                        
                        self._save_installed_models()
                    except Exception as e:
                        print(f"모델 토큰 제한 업데이트 중 오류: {e}")
                    
                    return True
                        
                except ValueError as e:
                    # 명시적으로 발생시킨 ValueError는 그대로 전달
                    raise e
                except Exception as e:
                    print(f"모델 로드 중 일반 오류: {e}")
                    raise e
            
            return False
        except Exception as e:
            error_message = str(e)
            print(f"모델 로드 중 오류 발생: {error_message}")
            
            # 특정 오류 메시지에 대해 더 명확한 사용자 안내 메시지 반환
            if "does not recognize this architecture" in error_message:
                self.current_model = None
                self.current_tokenizer = None
                self.current_model_id = None
                return False, "모델 아키텍처가 현재 Transformers 버전에서 지원되지 않습니다. Transformers 라이브러리를 업데이트해야 합니다."
            
            # GPU 메모리 부족 오류 처리
            if "CUDA out of memory" in error_message:
                self.current_model = None
                self.current_tokenizer = None
                self.current_model_id = None
                return False, "GPU 메모리 부족: CPU 모드로 전환하거나 더 작은 모델을 사용하세요."
            
            # 기타 오류
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_id = None
            return False, f"모델 로드 오류: {error_message}"
    
    def force_thinking_generation(self, prompt, settings):
        """
        추론 과정을 강제로 생성하는 함수
        
        Args:
            prompt: 입력 프롬프트
            settings: 생성 설정
            
        Returns:
            추론 과정
        """
        thinking_prompt = f"{prompt}\n\n<thinking>"
        
        try:
            thinking_inputs = self.current_tokenizer(thinking_prompt, return_tensors="pt")
            # 모델 장치로 텐서 이동
            for key in thinking_inputs:
                if hasattr(thinking_inputs[key], "to"):
                    thinking_inputs[key] = thinking_inputs[key].to(self.current_model.device)
            
            # 추론 과정 생성
            thinking_outputs = self.current_model.generate(
                **thinking_inputs,
                max_new_tokens=max(settings.get("max_output_tokens", 2048) // 2, 512),
                temperature=settings.get("temperature", 0.7),
                top_p=settings.get("top_p", 0.9),
                top_k=settings.get("top_k", 50),
                repetition_penalty=settings.get("repetition_penalty", 1.1),
                do_sample=settings.get("do_sample", True),
                pad_token_id=self.current_tokenizer.eos_token_id
            )
            thinking = self.current_tokenizer.decode(thinking_outputs[0], skip_special_tokens=True)
            thinking = thinking.replace(thinking_prompt, "").strip()
            
            # 종료 태그가 있으면 제거
            end_tags = ["</think>", "</thinking>", "</thoughts>", "</reasoning>", "</THINK>", "</THINKING>"]
            for end_tag in end_tags:
                if end_tag in thinking:
                    thinking = thinking.split(end_tag)[0].strip()
                    break
            
            return thinking
        except Exception as e:
            print(f"추론 과정 강제 생성 중 오류: {e}")
            return f"추론 과정 생성 실패: {str(e)}"


    def improve_thinking_detection(self, response, prompt, settings):
        """
        응답에서 추론 과정을 더 잘 감지하기 위한 함수
        
        Args:
            response: 모델 응답
            prompt: 원본 프롬프트
            settings: 생성 설정
            
        Returns:
            딕셔너리 {thinking, response} 또는 원본 응답
        """
        print("추론 과정 감지 시도 중...")
        
        # XML 태그 처리를 위한 변수들
        thinking_part = None
        response_part = None
        
        # 태그 쌍 정의
        tag_pairs = [
            ("<think>", "</think>"),
            ("<thinking>", "</thinking>"),
            ("<thoughts>", "</thoughts>"),
            ("<reasoning>", "</reasoning>"),
            ("<THINK>", "</THINK>"),
            ("<THINKING>", "</THINKING>")
        ]
        
        # 태그 쌍으로 분리 시도
        for start_tag, end_tag in tag_pairs:
            if start_tag in response and end_tag in response:
                print(f"태그 쌍 '{start_tag}'-'{end_tag}' 감지됨")
                try:
                    thinking_start = response.index(start_tag) + len(start_tag)
                    thinking_end = response.index(end_tag, thinking_start)
                    
                    thinking_part = response[thinking_start:thinking_end].strip()
                    
                    # 전체 태그 블록 찾기 및 제거
                    full_tag_block = response[response.index(start_tag):response.index(end_tag) + len(end_tag)]
                    response_part = response.replace(full_tag_block, "").strip()
                    
                    print(f"태그 쌍 감지 성공: {thinking_part[:20]}... -> {response_part[:20]}...")
                    break
                except Exception as e:
                    print(f"태그 쌍 '{start_tag}'-'{end_tag}' 처리 중 오류: {e}")
        
        # 종료 태그만으로 분리 시도
        if not thinking_part:
            end_tags = ["</think>", "</thinking>", "</thoughts>", "</reasoning>", "</THINK>", "</THINKING>"]
            for end_tag in end_tags:
                if end_tag in response:
                    print(f"종료 태그 '{end_tag}' 감지됨")
                    try:
                        parts = response.split(end_tag, 1)
                        
                        if len(parts) == 2:
                            thinking_part = parts[0].strip()
                            response_part = parts[1].strip()
                            
                            print(f"종료 태그 감지 성공: {thinking_part[:20]}... -> {response_part[:20]}...")
                            break
                    except Exception as e:
                        print(f"종료 태그 '{end_tag}' 처리 중 오류: {e}")
        
        # 감지에 실패했지만 추론 과정이 필요한 경우 강제로 생성
        if not thinking_part and settings.get("show_thinking", False):
            print("태그 감지 실패, 추론 과정 강제 생성 시도...")
            thinking_part = self.force_thinking_generation(prompt, settings)
            response_part = response
        
        # 추론 과정이 감지된 경우 딕셔너리 반환
        if thinking_part and response_part:
            return {
                "thinking": thinking_part,
                "response": response_part
            }
        
        # 감지 실패 시 원본 응답 반환
        return response

    def generate_text_streaming(self, prompt: str, settings: Dict[str, Any]):
        """
        텍스트를 스트리밍 방식으로 생성
        
        Args:
            prompt: 입력 프롬프트
            settings: 생성 설정
            
        Yields:
            생성된 토큰 또는 이벤트
        """
        if not self.current_model or not self.current_tokenizer:
            yield "모델이 로드되지 않았습니다."
            return
        
        # 마지막 완전한 응답을 저장하기 위한 변수 초기화
        self._last_complete_response = ""
        self._thinking_part = None
        
        try:
            # 설정 파라미터 추출
            max_input_tokens = settings.get("max_input_tokens", 4096)
            max_output_tokens = settings.get("max_output_tokens", 2048)
            temperature = settings.get("temperature", 0.7)
            top_p = settings.get("top_p", 0.9)
            top_k = settings.get("top_k", 50)
            repetition_penalty = settings.get("repetition_penalty", 1.1)
            do_sample = settings.get("do_sample", True)
            show_thinking = settings.get("show_thinking", False)
            
            # 입력 토큰 제한
            input_tokens = self.current_tokenizer.encode(prompt, truncation=True, max_length=max_input_tokens)
            if len(input_tokens) > max_input_tokens:
                # 입력이 너무 길면 잘라냄
                input_tokens = input_tokens[:max_input_tokens]
                prompt = self.current_tokenizer.decode(input_tokens)
            
            # 추론 모델 여부 확인
            model_id = self.current_model_id
            is_inference_model = model_id and self.installed_models[model_id].get("is_inference_model", False)
            
            # 추론 과정 (thinking) 표시 처리
            if show_thinking and is_inference_model:
                # '생각 중' 이벤트 전송
                yield {"event": "thinking_start"}
                
                # 추론 모델에 thinking 태그 포함 프롬프트 전달
                modified_prompt = f"{prompt}\n\n아래 형식으로 답변해주세요:\n<thinking>\n[문제를 푸는 과정, 추론 과정을 상세히 작성]\n</thinking>\n\n[최종 답변]"
                
                try:
                    inputs = self.current_tokenizer(modified_prompt, return_tensors="pt")
                    for key in inputs:
                        if hasattr(inputs[key], "to"):
                            inputs[key] = inputs[key].to(self.current_model.device)
                    
                    # Streamer 설정
                    from transformers import TextIteratorStreamer
                    import threading
                    
                    streamer = TextIteratorStreamer(
                        self.current_tokenizer, 
                        skip_prompt=True,
                        timeout=60.0
                    )
                    
                    # 생성 중 다른 처리를 위해 스레드 사용
                    generation_kwargs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs.get("attention_mask", None),
                        "max_new_tokens": max_output_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "do_sample": do_sample,
                        "pad_token_id": self.current_tokenizer.eos_token_id,
                        "streamer": streamer
                    }
                    
                    thread = threading.Thread(target=self.current_model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    # 누적 응답
                    accumulated_text = ""
                    thinking_mode = False
                    thinking_text = ""
                    response_text = ""
                    
                    # 토큰 스트리밍
                    for new_text in streamer:
                        accumulated_text += new_text
                        
                        # thinking 태그 처리
                        if "<thinking>" in accumulated_text and not thinking_mode:
                            thinking_mode = True
                            parts = accumulated_text.split("<thinking>", 1)
                            if len(parts) > 1:
                                response_text += parts[0]
                                thinking_text = ""  # thinking 시작
                                # thinking 시작 이벤트 전송
                                yield {"event": "thinking_content_start"}
                            continue
                        
                        if thinking_mode and "</thinking>" in accumulated_text:
                            thinking_mode = False
                            parts = accumulated_text.split("</thinking>", 1)
                            if len(parts) > 1:
                                thinking_text += parts[0]
                                response_text = parts[1]  # thinking 이후 응답
                                self._thinking_part = thinking_text.strip()  # 저장
                                # thinking 종료 및 응답 시작 이벤트 전송
                                yield {"event": "thinking_content_end"}
                                yield {"event": "response_start"}
                                # 응답 첫 부분 전송
                                if response_text:
                                    yield response_text
                            continue
                        
                        # 현재 모드에 따라 텍스트 누적
                        if thinking_mode:
                            thinking_text += new_text
                            # thinking 내용 스트리밍
                            yield {"event": "thinking_token", "token": new_text}
                        else:
                            if "</thinking>" in accumulated_text:
                                # 이미 thinking이 끝난 상태
                                response_text += new_text
                                yield new_text
                            else:
                                # thinking 태그가 아직 없는 경우 일반 텍스트로 취급
                                accumulated_text += new_text
                                yield new_text
                    
                    # 저장할 전체 응답 구성
                    if self._thinking_part:
                        self._last_complete_response = {
                            "thinking": self._thinking_part,
                            "response": response_text.strip()
                        }
                    else:
                        self._last_complete_response = accumulated_text.strip()
                    
                except Exception as e:
                    print(f"스트리밍 생성 중 오류: {e}")
                    yield f"오류 발생: {str(e)}"
                    self._last_complete_response = f"오류 발생: {str(e)}"
            
            else:
                # 일반 생성 (thinking 없음)
                try:
                    inputs = self.current_tokenizer(prompt, return_tensors="pt")
                    for key in inputs:
                        if hasattr(inputs[key], "to"):
                            inputs[key] = inputs[key].to(self.current_model.device)
                    
                    # Streamer 설정
                    from transformers import TextIteratorStreamer
                    import threading
                    
                    streamer = TextIteratorStreamer(
                        self.current_tokenizer, 
                        skip_prompt=True,
                        timeout=60.0
                    )
                    
                    # 생성 설정
                    generation_kwargs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs.get("attention_mask", None),
                        "max_new_tokens": max_output_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "do_sample": do_sample,
                        "pad_token_id": self.current_tokenizer.eos_token_id,
                        "streamer": streamer
                    }
                    
                    # 이벤트: 응답 시작
                    yield {"event": "response_start"}
                    
                    thread = threading.Thread(target=self.current_model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    # 응답 누적
                    full_response = ""
                    
                    # 토큰 스트리밍
                    for new_text in streamer:
                        full_response += new_text
                        yield new_text
                    
                    # 전체 응답 저장
                    self._last_complete_response = full_response.strip()
                    
                except Exception as e:
                    print(f"스트리밍 생성 중 오류: {e}")
                    yield f"오류 발생: {str(e)}"
                    self._last_complete_response = f"오류 발생: {str(e)}"
        
        except Exception as e:
            print(f"스트리밍 텍스트 생성 중 오류 발생: {e}")
            yield f"오류 발생: {str(e)}"
            self._last_complete_response = f"오류 발생: {str(e)}"

    def get_last_complete_response(self):
        """
        마지막으로 생성된 완전한 응답 반환
        
        Returns:
            마지막 완전한 응답
        """
        return getattr(self, "_last_complete_response", "")
    
    def generate_text(self, prompt: str, settings: Dict[str, Any]) -> Union[str, Dict[str, str]]:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            settings: 생성 설정
            
        Returns:
            생성된 텍스트 또는 추론 과정을 포함한 딕셔너리
        """
        if not self.current_model or not self.current_tokenizer:
            return "모델이 로드되지 않았습니다."
        
        try:
            # 설정 파라미터 추출
            max_input_tokens = settings.get("max_input_tokens", 4096)
            max_output_tokens = settings.get("max_output_tokens", 2048)
            temperature = settings.get("temperature", 0.7)
            top_p = settings.get("top_p", 0.9)
            top_k = settings.get("top_k", 50)
            repetition_penalty = settings.get("repetition_penalty", 1.1)
            do_sample = settings.get("do_sample", True)
            show_thinking = settings.get("show_thinking", False)
            
            # 입력 토큰 제한
            input_tokens = self.current_tokenizer.encode(prompt, truncation=True, max_length=max_input_tokens)
            if len(input_tokens) > max_input_tokens:
                # 입력이 너무 길면 잘라냄
                input_tokens = input_tokens[:max_input_tokens]
                prompt = self.current_tokenizer.decode(input_tokens)
            
            # 추론 모델 여부 확인
            model_id = self.current_model_id
            is_inference_model = model_id and self.installed_models[model_id].get("is_inference_model", False)
            
            # 추론 모델 확인 및 show_thinking이 활성화된 경우
            if show_thinking and is_inference_model:
                # 추론 모델에 직접 thinking 태그 포함 프롬프트 전달
                modified_prompt = f"{prompt}\n\n아래 형식으로 답변해주세요:\n<thinking>\n[문제를 푸는 과정, 추론 과정을 상세히 작성]\n</thinking>\n\n[최종 답변]"
                
                try:
                    inputs = self.current_tokenizer(modified_prompt, return_tensors="pt")
                    for key in inputs:
                        if hasattr(inputs[key], "to"):
                            inputs[key] = inputs[key].to(self.current_model.device)
                    
                    # 텍스트 생성
                    outputs = self.current_model.generate(
                        **inputs,
                        max_new_tokens=max_output_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        do_sample=do_sample,
                        pad_token_id=self.current_tokenizer.eos_token_id
                    )
                    
                    response = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response.replace(modified_prompt, "").strip()
                    
                    # 추론 과정 감지 함수 호출
                    return self.improve_thinking_detection(response, prompt, settings)
                    
                except Exception as e:
                    print(f"추론 모델 처리 중 오류: {e}")
                    # 오류 발생 시 일반 방식으로 재시도
            
            # 일반 모델 처리 또는 추론 모델 실패 시 백업 처리
            if show_thinking and not is_inference_model:
                thinking_prompt = f"{prompt}\n\n<thinking>"
                
                # 추론 과정 생성
                try:
                    thinking_inputs = self.current_tokenizer(thinking_prompt, return_tensors="pt")
                    # 모델 장치로 텐서 이동
                    for key in thinking_inputs:
                        if hasattr(thinking_inputs[key], "to"):
                            thinking_inputs[key] = thinking_inputs[key].to(self.current_model.device)
                    
                    # 추론 과정 생성
                    thinking_outputs = self.current_model.generate(
                        **thinking_inputs,
                        max_new_tokens=max_output_tokens // 2,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        do_sample=do_sample,
                        pad_token_id=self.current_tokenizer.eos_token_id
                    )
                    thinking = self.current_tokenizer.decode(thinking_outputs[0], skip_special_tokens=True)
                    thinking = thinking.replace(thinking_prompt, "")
                    
                    # 종료 태그가 있으면 제거
                    if "</thinking>" in thinking:
                        thinking = thinking.split("</thinking>")[0].strip()
                    
                    # 최종 응답 생성
                    final_prompt = f"{prompt}"
                    final_inputs = self.current_tokenizer(final_prompt, return_tensors="pt")
                    # 모델 장치로 텐서 이동
                    for key in final_inputs:
                        if hasattr(final_inputs[key], "to"):
                            final_inputs[key] = final_inputs[key].to(self.current_model.device)
                    
                    final_outputs = self.current_model.generate(
                        **final_inputs,
                        max_new_tokens=max_output_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        do_sample=do_sample,
                        pad_token_id=self.current_tokenizer.eos_token_id
                    )
                    response = self.current_tokenizer.decode(final_outputs[0], skip_special_tokens=True)
                    response = response.replace(final_prompt, "").strip()
                    
                    return {
                        "thinking": thinking,
                        "response": response
                    }
                except Exception as thinking_error:
                    print(f"Thinking 생성 중 오류 발생, 일반 생성으로 시도: {thinking_error}")
                    # 추론 과정 생성 실패 시 일반 생성으로 진행
            
            # 일반 생성 시도
            try:
                # 입력 토큰화 및 모델 장치로 이동
                inputs = self.current_tokenizer(prompt, return_tensors="pt")
                for key in inputs:
                    if hasattr(inputs[key], "to"):
                        inputs[key] = inputs[key].to(self.current_model.device)
                
                # 텍스트 생성
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=max_output_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=self.current_tokenizer.eos_token_id
                )
                
                response = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.replace(prompt, "").strip()
                
                # show_thinking이 활성화된 경우 추론 과정 감지 시도
                if show_thinking:
                    # 추론 과정 감지 함수 호출
                    return self.improve_thinking_detection(response, prompt, settings)
                
                return response
                
            except Exception as e:
                print(f"일반 생성 중 오류 발생: {e}")
                return f"텍스트 생성 중 오류가 발생했습니다: {str(e)}"
            
        except Exception as e:
            print(f"텍스트 생성 중 오류 발생: {e}")
            return f"텍스트 생성 중 오류가 발생했습니다: {str(e)}"