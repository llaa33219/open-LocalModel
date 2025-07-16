#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chat.py - 채팅 기능 파일
open-LocalModel: 허깅페이스 Transformers를 이용한 로컬 AI 모델 인터페이스
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

class ChatManager:
    """채팅 관리 클래스"""
    
    def __init__(self, chat_dir: str = "Chat"):
        """
        초기화 함수
        
        Args:
            chat_dir: 채팅 저장 디렉토리
        """
        self.chat_dir = chat_dir
        self.current_chat_id = None
        self.current_chat = []
        
        # 채팅 디렉토리 생성
        if not os.path.exists(chat_dir):
            os.makedirs(chat_dir)
    
    def create_new_chat(self) -> str:
        """
        새 채팅 생성
        
        Returns:
            새 채팅 ID
        """
        # 현재 채팅 저장
        if self.current_chat_id and self.current_chat:
            self.save_chat()
        
        # 새 채팅 ID 생성 (타임스탬프 기반)
        self.current_chat_id = f"chat_{int(time.time())}"
        self.current_chat = []
        
        return self.current_chat_id
    
    def save_chat(self) -> bool:
        """
        현재 채팅 저장
        
        Returns:
            저장 성공 여부
        """
        if not self.current_chat_id or not self.current_chat:
            return False
        
        try:
            chat_path = os.path.join(self.chat_dir, f"{self.current_chat_id}.json")
            
            # 채팅 메타데이터 추가
            chat_data = {
                "id": self.current_chat_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "messages": self.current_chat
            }
            
            with open(chat_path, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"채팅 저장 중 오류 발생: {e}")
            return False
    
    def load_chat(self, chat_id: str) -> bool:
        """
        채팅 로드
        
        Args:
            chat_id: 로드할 채팅 ID
            
        Returns:
            로드 성공 여부
        """
        try:
            chat_path = os.path.join(self.chat_dir, f"{chat_id}.json")
            
            if not os.path.exists(chat_path):
                return False
            
            with open(chat_path, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
            
            self.current_chat_id = chat_id
            self.current_chat = chat_data.get("messages", [])
            
            return True
        except Exception as e:
            print(f"채팅 로드 중 오류 발생: {e}")
            return False
    
    def add_message(self, role: str, content: str, thinking: Optional[str] = None) -> None:
        """
        메시지 추가
        
        Args:
            role: 메시지 역할 (user 또는 assistant)
            content: 메시지 내용
            thinking: 추론 과정 (선택 사항)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        # 추론 과정이 있는 경우 추가
        if thinking:
            message["thinking"] = thinking
        
        self.current_chat.append(message)
        self.save_chat()
    
    def get_current_chat(self) -> List[Dict[str, Any]]:
        """
        현재 채팅 가져오기
        
        Returns:
            현재 채팅 메시지 목록
        """
        return self.current_chat
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        채팅 기록 가져오기
        
        Returns:
            채팅 기록 목록
        """
        chat_files = [f for f in os.listdir(self.chat_dir) if f.endswith('.json')]
        chat_history = []
        
        for chat_file in chat_files:
            try:
                with open(os.path.join(self.chat_dir, chat_file), 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                
                # 채팅 요약 정보 추출
                chat_summary = {
                    "id": chat_data.get("id", chat_file.replace('.json', '')),
                    "created_at": chat_data.get("created_at", ""),
                    "updated_at": chat_data.get("updated_at", ""),
                    "message_count": len(chat_data.get("messages", [])),
                    "preview": chat_data.get("messages", [{}])[0].get("content", "")[:50] + "..." if chat_data.get("messages") else ""
                }
                
                chat_history.append(chat_summary)
            except Exception as e:
                print(f"채팅 기록 로드 중 오류 발생: {chat_file}, {e}")
        
        # 최신 채팅이 먼저 오도록 정렬
        chat_history.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return chat_history
    
    def delete_chat(self, chat_id: str) -> bool:
        """
        채팅 삭제
        
        Args:
            chat_id: 삭제할 채팅 ID
            
        Returns:
            삭제 성공 여부
        """
        try:
            chat_path = os.path.join(self.chat_dir, f"{chat_id}.json")
            
            if not os.path.exists(chat_path):
                return False
            
            os.remove(chat_path)
            
            # 현재 채팅이 삭제된 채팅인 경우 초기화
            if self.current_chat_id == chat_id:
                self.current_chat_id = None
                self.current_chat = []
            
            return True
        except Exception as e:
            print(f"채팅 삭제 중 오류 발생: {e}")
            return False