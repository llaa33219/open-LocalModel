// DOM 요소
const appHeader = document.querySelector('.app-header');
const chatSection = document.getElementById('chatSection');
const settingsSection = document.getElementById('settingsSection');
const toggleChatBtn = document.getElementById('toggleChatBtn');
const settingsBtn = document.getElementById('settingsBtn');
const closeSettingsBtn = document.getElementById('closeSettingsBtn');
const chatHistory = document.getElementById('chatHistory');
const messages = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const deleteChatBtn = document.getElementById('deleteChatBtn');
const thinkingIndicator = document.getElementById('thinkingIndicator');
const modelSelect = document.getElementById('modelSelect');
const removeModelBtn = document.getElementById('removeModelBtn');
const installModelBtn = document.getElementById('installModelBtn');
const installModelModal = document.getElementById('installModelModal');
const closeModelModal = document.getElementById('closeModelModal');
const modelSearchInput = document.getElementById('modelSearchInput');
const searchModelBtn = document.getElementById('searchModelBtn');
const availableModelList = document.getElementById('availableModelList');
const templateSelect = document.getElementById('templateSelect');
const templateName = document.getElementById('templateName');
const saveTemplateBtn = document.getElementById('saveTemplateBtn');
const loadTemplateBtn = document.getElementById('loadTemplateBtn');
const resetSettingsBtn = document.getElementById('resetSettingsBtn');
const thinkBtn = document.getElementById('thinkBtn');

// 설정 파라미터 요소
const maxInputTokens = document.getElementById('maxInputTokens');
const maxOutputTokens = document.getElementById('maxOutputTokens');
const maxTotalTokens = document.getElementById('maxTotalTokens');
const temperature = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperatureValue');
const topP = document.getElementById('topP');
const topPValue = document.getElementById('topPValue');
const topK = document.getElementById('topK');
const repetitionPenalty = document.getElementById('repetitionPenalty');
const repetitionPenaltyValue = document.getElementById('repetitionPenaltyValue');
const doSample = document.getElementById('doSample');
const showThinking = document.getElementById('showThinking');
const deviceGPU = document.getElementById('deviceGPU');
const deviceCPU = document.getElementById('deviceCPU');
const useQuantization = document.getElementById('useQuantization');
const quantizationBits = document.getElementById('quantizationBits');
const quantizationOptionsDiv = document.getElementById('quantizationOptionsDiv');

// 현재 채팅 ID
let currentChatId = null;

// 현재 선택된 모델 ID
let currentModelId = null;

// 현재 선택된 모델 이름
let currentModelName = "AI";

// 응답 중단을 위한 컨트롤러
let currentResponseController = null;

// 응답 진행 중 상태
let isResponding = false;

// 마크다운 설정 초기화
function initializeMarkdown() {
    // 마크다운 파서 설정
    marked.setOptions({
        renderer: new marked.Renderer(),
        highlight: function(code, lang) {
            const language = hljs.getLanguage(lang) ? lang : 'plaintext';
            return hljs.highlight(code, { language }).value;
        },
        langPrefix: 'hljs language-', // highlight.js css에서 사용할 클래스 프리픽스
        pedantic: false,
        gfm: true,
        breaks: true,
        sanitize: false,
        smartypants: false,
        xhtml: false
    });
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    // 마크다운 초기화
    initializeMarkdown();
    
    // 새 채팅 생성
    createNewChat();
    
    // 설치된 모델 로드
    loadInstalledModels();
    
    // 설정 템플릿 로드
    loadSettingsTemplates();
    
    // 채팅 기록 로드
    loadChatHistory();
    
    // 현재 설정 로드
    loadCurrentSettings();
    
    // 양자화 옵션 초기화
    quantizationOptionsDiv.style.display = useQuantization.checked && deviceGPU.checked ? 'block' : 'none';
    
    // Think 버튼 상태 초기화
    updateThinkButtonState();
    
    // 토글 스위치에 클릭 이벤트 추가
    document.querySelectorAll('.toggle-switch').forEach(toggleSwitch => {
        const toggleId = toggleSwitch.querySelector('input[type="checkbox"]').id;
        const toggleInput = document.getElementById(toggleId);
        
        if (toggleInput) {
            toggleSwitch.addEventListener('click', (e) => {
                // 체크박스 상태 토글
                toggleInput.checked = !toggleInput.checked;
                
                // 이벤트 트리거
                const event = new Event('change');
                toggleInput.dispatchEvent(event);
                
                // 토글 애니메이션
                toggleSwitch.classList.add('toggled');
                setTimeout(() => {
                    toggleSwitch.classList.remove('toggled');
                }, 300);
            });
        }
    });
    
    // 툴팁 초기화
    initializeTooltips();
    
    // 텍스트 영역 자동 크기 조절
    initializeAutoResizeTextarea();
});

// 툴팁 초기화 함수
function initializeTooltips() {
    const tooltips = document.querySelectorAll('[title]');
    tooltips.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

// 툴팁 표시 함수
function showTooltip(e) {
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip-popup';
    tooltip.textContent = this.getAttribute('title');
    
    document.body.appendChild(tooltip);
    
    const rect = this.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    
    tooltip.style.top = `${rect.top - tooltipRect.height - 10}px`;
    tooltip.style.left = `${rect.left + (rect.width / 2) - (tooltipRect.width / 2)}px`;
    
    // 애니메이션 효과
    setTimeout(() => {
        tooltip.style.opacity = '1';
        tooltip.style.transform = 'translateY(0)';
    }, 10);
    
    this.tooltip = tooltip;
}

// 툴팁 숨김 함수
function hideTooltip() {
    if (this.tooltip) {
        this.tooltip.style.opacity = '0';
        this.tooltip.style.transform = 'translateY(10px)';
        
        setTimeout(() => {
            if (this.tooltip && this.tooltip.parentNode) {
                this.tooltip.parentNode.removeChild(this.tooltip);
                this.tooltip = null;
            }
        }, 300);
    }
}

// 텍스트 영역 자동 크기 조절 초기화
function initializeAutoResizeTextarea() {
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        const newHeight = Math.min(this.scrollHeight, 200);
        this.style.height = `${newHeight}px`;
    });
}

// 토글 버튼 이벤트
toggleChatBtn.addEventListener('click', () => {
    chatSection.classList.toggle('collapsed');
    
    // 아이콘 변경
    if (chatSection.classList.contains('collapsed')) {
        toggleChatBtn.innerHTML = '<i class="fas fa-chevron-right"></i>';
    } else {
        toggleChatBtn.innerHTML = '<i class="fas fa-chevron-left"></i>';
    }
    
    // 애니메이션 효과
    toggleChatBtn.classList.add('clicked');
    setTimeout(() => {
        toggleChatBtn.classList.remove('clicked');
    }, 300);
});

// 설정 버튼 이벤트
settingsBtn.addEventListener('click', () => {
    toggleSettingsPanel();
    
    // 버튼 애니메이션
    settingsBtn.classList.add('clicked');
    setTimeout(() => {
        settingsBtn.classList.remove('clicked');
    }, 300);
});

closeSettingsBtn.addEventListener('click', () => {
    hideSettingsPanel();
    
    // 버튼 애니메이션
    closeSettingsBtn.classList.add('clicked');
    setTimeout(() => {
        closeSettingsBtn.classList.remove('clicked');
    }, 300);
});

// 설정 패널 토글 함수
function toggleSettingsPanel() {
    const isVisible = settingsSection.classList.contains('visible');
    
    if (isVisible) {
        hideSettingsPanel();
    } else {
        showSettingsPanel();
    }
}

// 설정 패널 표시 함수
function showSettingsPanel() {
    settingsSection.classList.add('visible');
    settingsBtn.classList.add('active');
    
    // 배경 오버레이 추가
    const overlay = document.createElement('div');
    overlay.className = 'settings-overlay';
    overlay.addEventListener('click', hideSettingsPanel);
    document.body.appendChild(overlay);
    
    // 애니메이션 효과
    setTimeout(() => {
        overlay.style.opacity = '1';
    }, 10);
}

// 설정 패널 숨김 함수
function hideSettingsPanel() {
    settingsSection.classList.remove('visible');
    settingsBtn.classList.remove('active');
    
    // 배경 오버레이 제거
    const overlay = document.querySelector('.settings-overlay');
    if (overlay) {
        overlay.style.opacity = '0';
        setTimeout(() => {
            if (overlay.parentNode) {
                overlay.parentNode.removeChild(overlay);
            }
        }, 300);
    }
}

// Think 버튼 이벤트
thinkBtn.addEventListener('click', () => {
    showThinking.checked = !showThinking.checked;
    updateThinkButtonState();
    updateSettings();
    
    // 버튼 애니메이션
    thinkBtn.classList.add('clicked');
    setTimeout(() => {
        thinkBtn.classList.remove('clicked');
    }, 300);
});

// Think 버튼 상태 업데이트 함수
function updateThinkButtonState() {
    if (showThinking.checked) {
        thinkBtn.classList.add('active');
    } else {
        thinkBtn.classList.remove('active');
    }
}

// 메시지 전송 이벤트
sendBtn.addEventListener('click', handleSendButtonClick);
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendButtonClick();
    }
});

// 보내기/정지 버튼 클릭 처리 함수
function handleSendButtonClick() {
    if (isResponding) {
        // 응답 중인 경우 중지
        stopResponse();
    } else {
        // 응답 중이 아닌 경우 메시지 전송
        sendMessage();
    }
}

// 응답 중지 함수
function stopResponse() {
    if (currentResponseController) {
        currentResponseController.abort();
        currentResponseController = null;
    }
    
    // 버튼 상태 업데이트
    updateSendButtonState(false);
    
    // 상태 업데이트
    isResponding = false;
    
    // 알림 표시
    showNotification('AI 응답이 중지되었습니다.', 'info');
}

// 보내기 버튼 상태 업데이트 함수
function updateSendButtonState(responding) {
    if (responding) {
        // 응답 중 - 정지 버튼으로 변경
        sendBtn.innerHTML = '<i class="fas fa-stop"></i>';
        sendBtn.classList.add('stop-button');
        sendBtn.title = '응답 중지';
    } else {
        // 대기 중 - 보내기 버튼으로 변경
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        sendBtn.classList.remove('stop-button');
        sendBtn.title = '메시지 보내기';
    }
}

// 새 채팅 생성 이벤트
newChatBtn.addEventListener('click', () => {
    createNewChat();
    
    // 버튼 애니메이션
    newChatBtn.classList.add('clicked');
    setTimeout(() => {
        newChatBtn.classList.remove('clicked');
    }, 300);
});

// 채팅 삭제 이벤트
deleteChatBtn.addEventListener('click', () => {
    deleteCurrentChat();
    
    // 버튼 애니메이션
    deleteChatBtn.classList.add('clicked');
    setTimeout(() => {
        deleteChatBtn.classList.remove('clicked');
    }, 300);
});

// 모델 제거 이벤트
removeModelBtn.addEventListener('click', () => {
    removeSelectedModel();
    
    // 버튼 애니메이션
    removeModelBtn.classList.add('clicked');
    setTimeout(() => {
        removeModelBtn.classList.remove('clicked');
    }, 300);
});

// 모델 설치 모달 이벤트
installModelBtn.addEventListener('click', () => {
    installModelModal.style.display = 'flex';
    searchAvailableModels();
    
    // 모달 애니메이션
    setTimeout(() => {
        installModelModal.querySelector('.modal-content').classList.add('visible');
    }, 10);
    
    // 버튼 애니메이션
    installModelBtn.classList.add('clicked');
    setTimeout(() => {
        installModelBtn.classList.remove('clicked');
    }, 300);
});

closeModelModal.addEventListener('click', () => {
    // 모달 애니메이션
    installModelModal.querySelector('.modal-content').classList.remove('visible');
    
    setTimeout(() => {
        installModelModal.style.display = 'none';
    }, 300);
    
    // 버튼 애니메이션
    closeModelModal.classList.add('clicked');
    setTimeout(() => {
        closeModelModal.classList.remove('clicked');
    }, 300);
});

searchModelBtn.addEventListener('click', () => {
    searchAvailableModels();
    
    // 버튼 애니메이션
    searchModelBtn.classList.add('clicked');
    setTimeout(() => {
        searchModelBtn.classList.remove('clicked');
    }, 300);
});

// 설정 기본값으로 초기화 이벤트
resetSettingsBtn.addEventListener('click', () => {
    resetToModelDefaults();
    
    // 버튼 애니메이션
    resetSettingsBtn.classList.add('clicked');
    setTimeout(() => {
        resetSettingsBtn.classList.remove('clicked');
    }, 300);
});

// 설정 템플릿 저장 이벤트
saveTemplateBtn.addEventListener('click', () => {
    saveSettingsTemplate();
    
    // 버튼 애니메이션
    saveTemplateBtn.classList.add('clicked');
    setTimeout(() => {
        saveTemplateBtn.classList.remove('clicked');
    }, 300);
});

// 설정 템플릿 로드 이벤트
loadTemplateBtn.addEventListener('click', () => {
    loadSettingsTemplate();
    
    // 버튼 애니메이션
    loadTemplateBtn.classList.add('clicked');
    setTimeout(() => {
        loadTemplateBtn.classList.remove('clicked');
    }, 300);
});

// 슬라이더 값 표시 업데이트
temperature.addEventListener('input', () => {
    temperatureValue.textContent = temperature.value;
    temperatureValue.classList.add('highlight');
    setTimeout(() => {
        temperatureValue.classList.remove('highlight');
    }, 300);
});

topP.addEventListener('input', () => {
    topPValue.textContent = topP.value;
    topPValue.classList.add('highlight');
    setTimeout(() => {
        topPValue.classList.remove('highlight');
    }, 300);
});

repetitionPenalty.addEventListener('input', () => {
    repetitionPenaltyValue.textContent = repetitionPenalty.value;
    repetitionPenaltyValue.classList.add('highlight');
    setTimeout(() => {
        repetitionPenaltyValue.classList.remove('highlight');
    }, 300);
});

// 모델 선택 이벤트
modelSelect.addEventListener('change', loadSelectedModel);

// 디바이스 선택 이벤트
deviceGPU.addEventListener('change', () => {
    quantizationOptionsDiv.style.display = useQuantization.checked ? 'block' : 'none';
    updateSettings();
});

deviceCPU.addEventListener('change', () => {
    quantizationOptionsDiv.style.display = 'none';
    updateSettings();
});

useQuantization.addEventListener('change', () => {
    quantizationOptionsDiv.style.display = useQuantization.checked && deviceGPU.checked ? 'block' : 'none';
    updateSettings();
});

quantizationBits.addEventListener('change', updateSettings);

// showThinking 체크박스 이벤트
showThinking.addEventListener('change', () => {
    updateThinkButtonState();
    updateSettings();
});

// 새 채팅 생성 함수
function createNewChat() {
    fetch('/api/chat/new', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        currentChatId = data.chat_id;
        messages.innerHTML = '';
        
        // 환영 메시지 제거됨 (사용자 요청에 따라)
        
        loadChatHistory();
    })
    .catch(error => console.error('Error creating new chat:', error));
}

// 채팅 기록 로드 함수
function loadChatHistory() {
    fetch('/api/chat/history')
    .then(response => response.json())
    .then(data => {
        chatHistory.innerHTML = '';
        
        data.forEach((chat, index) => {
            const chatItem = document.createElement('div');
            chatItem.className = 'chat-item';
            if (chat.id === currentChatId) {
                chatItem.classList.add('active');
            }
            
            // 아이콘 추가
            const icon = document.createElement('i');
            icon.className = 'fas fa-comment';
            chatItem.appendChild(icon);
            
            // 텍스트 추가
            const text = document.createElement('span');
            text.textContent = chat.preview || `채팅 ${chat.id}`;
            chatItem.appendChild(text);
            
            chatItem.dataset.chatId = chat.id;
            
            chatItem.addEventListener('click', () => {
                loadChat(chat.id);
            });
            
            chatHistory.appendChild(chatItem);
            
            // 애니메이션 효과
            setTimeout(() => {
                chatItem.classList.add('visible');
            }, 50 * index);
        });
    })
    .catch(error => console.error('Error loading chat history:', error));
}

// 채팅 로드 함수
function loadChat(chatId) {
    fetch('/api/chat/load', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ chat_id: chatId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentChatId = chatId;
            messages.innerHTML = '';
            
            // 채팅 항목 활성화 상태 업데이트
            document.querySelectorAll('.chat-item').forEach(item => {
                item.classList.remove('active');
                if (item.dataset.chatId === chatId) {
                    item.classList.add('active');
                }
            });
            
            // 메시지 표시
            data.messages.forEach((msg, index) => {
                const messageElement = displayMessage(msg.role, msg.content, msg.thinking);
                
                // 애니메이션 효과
                setTimeout(() => {
                    messageElement.classList.add('visible');
                }, 100 * index);
            });
            
            // 스크롤을 맨 아래로
            scrollToBottom();
        }
    })
    .catch(error => console.error('Error loading chat:', error));
}

// 현재 채팅 삭제 함수
function deleteCurrentChat() {
    if (!currentChatId) return;
    
    // 확인 모달 생성
    const confirmModal = document.createElement('div');
    confirmModal.className = 'modal confirm-modal';
    confirmModal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2><i class="fas fa-trash"></i> 채팅 삭제</h2>
                <span class="close-modal">&times;</span>
            </div>
            <div class="modal-body">
                <p>현재 채팅을 삭제하시겠습니까?</p>
                <div class="confirm-buttons">
                    <button class="cancel-btn">취소</button>
                    <button class="confirm-btn">삭제</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(confirmModal);
    
    // 모달 애니메이션
    setTimeout(() => {
        confirmModal.querySelector('.modal-content').classList.add('visible');
    }, 10);
    
    // 이벤트 리스너 추가
    confirmModal.querySelector('.close-modal').addEventListener('click', () => {
        closeConfirmModal(confirmModal);
    });
    
    confirmModal.querySelector('.cancel-btn').addEventListener('click', () => {
        closeConfirmModal(confirmModal);
    });
    
    confirmModal.querySelector('.confirm-btn').addEventListener('click', () => {
        fetch('/api/chat/delete', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ chat_id: currentChatId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                createNewChat();
                
                // 성공 알림 표시
                showNotification('채팅이 삭제되었습니다.', 'success');
            }
        })
        .catch(error => console.error('Error deleting chat:', error));
        
        closeConfirmModal(confirmModal);
    });
}

// 확인 모달 닫기 함수
function closeConfirmModal(modal) {
    modal.querySelector('.modal-content').classList.remove('visible');
    
    setTimeout(() => {
        if (modal.parentNode) {
            modal.parentNode.removeChild(modal);
        }
    }, 300);
}

// 알림 표시 함수
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // 애니메이션 효과
    setTimeout(() => {
        notification.classList.add('visible');
    }, 10);
    
    // 자동 제거
    setTimeout(() => {
        notification.classList.remove('visible');
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// 메시지 표시 함수
function displayMessage(role, content, thinking = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    // 역할 표시
    const roleSpan = document.createElement('div');
    roleSpan.className = 'message-role';
    
    if (role === 'user') {
        roleSpan.innerHTML = '<i class="fas fa-user"></i> 사용자';
    } else {
        // AI 응답인 경우 현재 모델명 표시
        roleSpan.innerHTML = `<i class="fas fa-robot"></i> ${currentModelName}`;
    }
    
    messageDiv.appendChild(roleSpan);
    
    // 추론 과정이 있는 경우
    if (thinking && role === 'assistant') {
        // 토글 버튼 생성
        const thinkingToggle = document.createElement('button');
        thinkingToggle.className = 'thinking-toggle';
        thinkingToggle.innerHTML = '<i class="fas fa-brain"></i> Think';
        
        // 추론 과정 컨테이너
        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'thinking-content';
        thinkingDiv.style.display = 'none';
        thinkingDiv.innerHTML = marked.parse(thinking);
        
        // 코드 블록에 하이라이팅 적용
        thinkingDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
        
        // 추론 과정을 메시지에 추가
        messageDiv.appendChild(thinkingToggle);
        messageDiv.appendChild(thinkingDiv);
        
        // 토글 버튼 이벤트
        thinkingToggle.addEventListener('click', () => {
            const isHidden = thinkingDiv.style.display === 'none';
            thinkingDiv.style.display = isHidden ? 'block' : 'none';
            thinkingToggle.innerHTML = isHidden ? 
                '<i class="fas fa-brain"></i> Hide' : 
                '<i class="fas fa-brain"></i> Think';
            
            // 버튼 애니메이션
            thinkingToggle.classList.add('clicked');
            setTimeout(() => {
                thinkingToggle.classList.remove('clicked');
            }, 300);
            
            // 스크롤 조정
            if (isHidden) {
                setTimeout(scrollToBottom, 100);
            }
        });
    }
    
    // 내용 표시
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content markdown-content';
    contentDiv.innerHTML = marked.parse(content);
    
    // 코드 블록에 하이라이팅 적용
    contentDiv.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
    
    messageDiv.appendChild(contentDiv);
    messages.appendChild(messageDiv);
    
    // 스크롤을 맨 아래로
    scrollToBottom();
    
    return messageDiv;
}

// 메시지 전송 함수
function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    
    // 사용자 메시지 표시
    const userMessageElement = displayMessage('user', message);
    
    // 애니메이션 효과
    setTimeout(() => {
        userMessageElement.classList.add('visible');
    }, 10);
    
    // 입력 필드 초기화
    userInput.value = '';
    userInput.style.height = 'auto';
    
    // Thinking 표시
    thinkingIndicator.style.display = 'block';
    
    // 스크롤을 맨 아래로
    scrollToBottom();
    
    // 응답을 위한 DOM 요소 생성
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    
    // 역할 표시 (현재 모델명 사용)
    const roleSpan = document.createElement('div');
    roleSpan.className = 'message-role';
    roleSpan.innerHTML = `<i class="fas fa-robot"></i> ${currentModelName}`;
    messageDiv.appendChild(roleSpan);
    
    // 생각 컨테이너 미리 생성 (필요시 사용)
    let thinkingDiv = null;
    let thinkingToggle = null;
    
    // 내용 표시를 위한 컨테이너
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content markdown-content';
    
    // 메시지 영역에 추가
    messageDiv.appendChild(contentDiv);
    messages.appendChild(messageDiv);

    // 마크다운 렌더링을 위한 변수
    let accumulatedMarkdown = '';
    let accumulatedThinking = '';
    
    // 응답 중 상태로 변경
    isResponding = true;
    
    // 버튼 상태 업데이트
    updateSendButtonState(true);
    
    // 중단 컨트롤러 생성
    currentResponseController = new AbortController();
    const signal = currentResponseController.signal;
    
    // 서버에 메시지 전송 (스트리밍 모드)
    fetch('/api/chat/message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message }),
        signal: signal
    })
    .then(response => {
        // 스트리밍 처리
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        
        // Thinking 숨기기 (스트리밍이 시작되면)
        thinkingIndicator.style.display = 'none';
        
        // 응답 모드 변수
        let isThinkingMode = false;
        let thinkingContent = '';
        let responseContent = '';
        
        // 애니메이션 효과
        setTimeout(() => {
            messageDiv.classList.add('visible');
        }, 10);
        
        function read() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    console.log('스트림 종료');
                    // 응답 완료 시 상태 업데이트
                    isResponding = false;
                    updateSendButtonState(false);
                    currentResponseController = null;
                    return;
                }
                
                // 디코딩 후 라인 단위로 처리
                const text = decoder.decode(value);
                const lines = text.split('\n').filter(line => line.trim() !== '');
                
                for (const line of lines) {
                    try {
                        const data = JSON.parse(line);
                        
                        // 이벤트 타입에 따른 처리
                        if (data.type === 'token') {
                            // 일반 텍스트 토큰
                            responseContent += data.data;
                            accumulatedMarkdown = responseContent; // 누적이 아닌 현재 전체 응답으로 업데이트
                            
                            // 마크다운 렌더링
                            contentDiv.innerHTML = marked.parse(accumulatedMarkdown);
                            
                            // 코드 블록에 하이라이팅 적용
                            contentDiv.querySelectorAll('pre code').forEach((block) => {
                                hljs.highlightElement(block);
                            });
                            
                            // 스크롤을 맨 아래로
                            scrollToBottom();
                        } 
                        else if (data.type === 'event') {
                            // 특수 이벤트 처리
                            const event = data.data.event;
                            
                            if (event === 'thinking_start') {
                                console.log('Thinking 시작');
                                // thinking 표시기 활성화 또는 UI 업데이트
                                thinkingIndicator.style.display = 'block';
                            } 
                            else if (event === 'thinking_content_start') {
                                console.log('Thinking 내용 시작');
                                isThinkingMode = true;
                                thinkingContent = '';
                                accumulatedThinking = '';
                                
                                // thinking 토글 버튼과 컨테이너 생성
                                if (!thinkingToggle) {
                                    // 토글 버튼 생성
                                    thinkingToggle = document.createElement('button');
                                    thinkingToggle.className = 'thinking-toggle';
                                    thinkingToggle.innerHTML = '<i class="fas fa-brain"></i> Think';
                                    
                                    // 추론 과정 컨테이너
                                    thinkingDiv = document.createElement('div');
                                    thinkingDiv.className = 'thinking-content';
                                    thinkingDiv.style.display = 'none';
                                    
                                    // 추론 과정을 메시지 상단에 추가
                                    messageDiv.insertBefore(thinkingToggle, contentDiv);
                                    messageDiv.insertBefore(thinkingDiv, contentDiv);
                                    
                                    // 토글 버튼 이벤트
                                    thinkingToggle.addEventListener('click', () => {
                                        const isHidden = thinkingDiv.style.display === 'none';
                                        thinkingDiv.style.display = isHidden ? 'block' : 'none';
                                        thinkingToggle.innerHTML = isHidden ? 
                                            '<i class="fas fa-brain"></i> Hide' : 
                                            '<i class="fas fa-brain"></i> Think';
                                        
                                        // 버튼 애니메이션
                                        thinkingToggle.classList.add('clicked');
                                        setTimeout(() => {
                                            thinkingToggle.classList.remove('clicked');
                                        }, 300);
                                        
                                        // 스크롤 조정
                                        if (isHidden) {
                                            setTimeout(scrollToBottom, 100);
                                        }
                                    });
                                }
                            } 
                            else if (event === 'thinking_content') {
                                // thinking 내용 추가
                                if (isThinkingMode && thinkingDiv) {
                                    // [직중 답변] 텍스트 제거
                                    let content = data.data.content;
                                    if (thinkingContent === '' && content.startsWith('[직중 답변]')) {
                                        content = content.replace('[직중 답변]', '');
                                    }
                                    
                                    thinkingContent += content;
                                    accumulatedThinking = thinkingContent; // 누적이 아닌 현재 전체 추론 과정으로 업데이트
                                    
                                    // 마크다운 렌더링
                                    thinkingDiv.innerHTML = marked.parse(accumulatedThinking);
                                    
                                    // 코드 블록에 하이라이팅 적용
                                    thinkingDiv.querySelectorAll('pre code').forEach((block) => {
                                        hljs.highlightElement(block);
                                    });
                                }
                            } 
                            else if (event === 'thinking_content_end') {
                                console.log('Thinking 내용 종료');
                                isThinkingMode = false;
                                
                                // thinking 표시기 업데이트
                                thinkingIndicator.style.display = 'none';
                            } 
                            else if (event === 'response_start') {
                                console.log('응답 시작');
                                responseContent = '';
                            }
                        } 
                        else if (data.type === 'complete') {
                            console.log('응답 완료');
                            
                            // 응답 완료 시 상태 업데이트
                            isResponding = false;
                            updateSendButtonState(false);
                            currentResponseController = null;
                            
                            // 스크롤을 맨 아래로
                            scrollToBottom();
                        }
                    } catch (e) {
                        console.error('JSON 파싱 오류:', e, line);
                    }
                }
                
                // 계속 읽기
                return read();
            }).catch(error => {
                if (error.name === 'AbortError') {
                    console.log('사용자에 의해 응답이 중단되었습니다.');
                } else {
                    console.error('스트리밍 오류:', error);
                    
                    // 오류 알림 표시
                    showNotification('응답 처리 중 오류가 발생했습니다.', 'error');
                }
                
                // 응답 상태 업데이트
                isResponding = false;
                updateSendButtonState(false);
                currentResponseController = null;
            });
        }
        
        // 스트리밍 시작
        return read();
    })
    .catch(error => {
        if (error.name === 'AbortError') {
            console.log('사용자에 의해 요청이 중단되었습니다.');
        } else {
            console.error('요청 오류:', error);
            
            // 오류 알림 표시
            showNotification('메시지 전송 중 오류가 발생했습니다.', 'error');
        }
        
        // Thinking 숨기기
        thinkingIndicator.style.display = 'none';
        
        // 응답 상태 업데이트
        isResponding = false;
        updateSendButtonState(false);
        currentResponseController = null;
    });
}

// 스크롤을 맨 아래로 이동하는 함수
function scrollToBottom() {
    messages.scrollTop = messages.scrollHeight;
}

// 설치된 모델 로드 함수
function loadInstalledModels() {
    fetch('/api/models/installed')
    .then(response => response.json())
    .then(data => {
        modelSelect.innerHTML = '<option value="">모델을 선택하세요</option>';
        
        // 모델 목록 정렬 (이름 기준)
        const sortedModels = Object.entries(data).sort((a, b) => {
            return a[1].name.localeCompare(b[1].name);
        });
        
        sortedModels.forEach(([id, model]) => {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = `${model.name} (${model.author})`;
            modelSelect.appendChild(option);
        });
        
        // 이전에 선택된 모델이 있으면 선택
        if (currentModelId) {
            modelSelect.value = currentModelId;
        }
    })
    .catch(error => console.error('Error loading installed models:', error));
}

// 선택된 모델 로드 함수
function loadSelectedModel() {
    const modelId = modelSelect.value;
    
    if (!modelId) {
        return;
    }
    
    // 로딩 알림 표시
    showNotification('모델을 로드하는 중입니다...', 'info');
    
    fetch('/api/models/load', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model_id: modelId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentModelId = modelId;
            
            // 모델 이름 추출 및 저장
            const selectedOption = modelSelect.options[modelSelect.selectedIndex];
            currentModelName = selectedOption.textContent.split(' (')[0];
            
            // 설정 업데이트
            updateSettingsFromData(data.settings);
            
            // 성공 알림 표시
            showNotification(`${currentModelName} 모델이 로드되었습니다.`, 'success');
        } else {
            // GPU 메모리 부족 오류 처리
            if (data.gpu_memory_error) {
                showNotification('GPU 메모리 부족: CPU 모드로 전환하거나 더 작은 모델을 사용하세요.', 'warning');
                
                // CPU 모드로 자동 전환
                deviceCPU.checked = true;
                deviceGPU.checked = false;
                quantizationOptionsDiv.style.display = 'none';
                
                // 설정 업데이트 후 다시 시도
                updateSettings(() => {
                    setTimeout(() => {
                        loadSelectedModel();
                    }, 1000);
                });
            } else {
                // 일반 오류
                showNotification(`모델 로드 실패: ${data.error}`, 'error');
            }
        }
    })
    .catch(error => {
        console.error('Error loading model:', error);
        showNotification('모델 로드 중 오류가 발생했습니다.', 'error');
    });
}

// 선택된 모델 제거 함수
function removeSelectedModel() {
    const modelId = modelSelect.value;
    
    if (!modelId) {
        showNotification('제거할 모델을 선택하세요.', 'warning');
        return;
    }
    
    // 확인 모달 생성
    const confirmModal = document.createElement('div');
    confirmModal.className = 'modal confirm-modal';
    confirmModal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2><i class="fas fa-trash"></i> 모델 제거</h2>
                <span class="close-modal">&times;</span>
            </div>
            <div class="modal-body">
                <p>선택한 모델을 제거하시겠습니까?</p>
                <div class="confirm-buttons">
                    <button class="cancel-btn">취소</button>
                    <button class="confirm-btn">제거</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(confirmModal);
    
    // 모달 애니메이션
    setTimeout(() => {
        confirmModal.querySelector('.modal-content').classList.add('visible');
    }, 10);
    
    // 이벤트 리스너 추가
    confirmModal.querySelector('.close-modal').addEventListener('click', () => {
        closeConfirmModal(confirmModal);
    });
    
    confirmModal.querySelector('.cancel-btn').addEventListener('click', () => {
        closeConfirmModal(confirmModal);
    });
    
    confirmModal.querySelector('.confirm-btn').addEventListener('click', () => {
        // 로딩 알림 표시
        showNotification('모델을 제거하는 중입니다...', 'info');
        
        fetch('/api/models/remove', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_id: modelId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 현재 모델 초기화
                if (currentModelId === modelId) {
                    currentModelId = null;
                    currentModelName = "AI";
                }
                
                // 모델 목록 다시 로드
                loadInstalledModels();
                
                // 성공 알림 표시
                showNotification('모델이 제거되었습니다.', 'success');
            } else {
                // 오류 알림 표시
                showNotification('모델 제거 실패', 'error');
            }
        })
        .catch(error => {
            console.error('Error removing model:', error);
            showNotification('모델 제거 중 오류가 발생했습니다.', 'error');
        });
        
        closeConfirmModal(confirmModal);
    });
}

// 사용 가능한 모델 검색 함수
function searchAvailableModels() {
    const searchText = modelSearchInput.value.trim();
    const authorFilter = document.getElementById('modelAuthorInput').value.trim();
    const sortBy = document.getElementById('modelSortBy').value;
    
    // 로딩 표시
    availableModelList.innerHTML = '<div class="loading-message"><i class="fas fa-spinner fa-spin"></i> 모델을 검색하는 중입니다...</div>';
    
    fetch(`/api/models/available?filter=${encodeURIComponent(searchText)}&author=${encodeURIComponent(authorFilter)}&sort_by=${sortBy}`)
    .then(response => response.json())
    .then(data => {
        availableModelList.innerHTML = '';
        
        if (data.length === 0) {
            availableModelList.innerHTML = '<div class="no-models-message">검색 결과가 없습니다.</div>';
            return;
        }
        
        data.forEach(model => {
            const modelItem = document.createElement('div');
            modelItem.className = 'model-item';
            
            modelItem.innerHTML = `
                <div class="model-info">
                    <div class="model-name">${model.name}</div>
                    <div class="model-author">제작자: ${model.author}</div>
                    <div class="model-downloads">다운로드: ${model.downloads.toLocaleString()}</div>
                </div>
                <button class="install-model-btn" data-model-id="${model.id}">
                    <i class="fas fa-download"></i> 설치
                </button>
            `;
            
            // 설치 버튼 이벤트
            modelItem.querySelector('.install-model-btn').addEventListener('click', (e) => {
                const modelId = e.currentTarget.dataset.modelId;
                installModel(modelId, e.currentTarget);
            });
            
            availableModelList.appendChild(modelItem);
        });
    })
    .catch(error => {
        console.error('Error searching models:', error);
        availableModelList.innerHTML = '<div class="error-message">모델 검색 중 오류가 발생했습니다.</div>';
    });
}

// 모델 설치 함수
function installModel(modelId, button) {
    // 버튼 상태 업데이트
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 설치 중...';
    
    fetch('/api/models/install', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model_id: modelId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 버튼 상태 업데이트
            button.innerHTML = '<i class="fas fa-check"></i> 설치됨';
            button.classList.add('installed');
            
            // 모델 목록 다시 로드
            loadInstalledModels();
            
            // 성공 알림 표시
            showNotification('모델이 설치되었습니다.', 'success');
        } else {
            // 버튼 상태 복구
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-download"></i> 설치';
            
            // 오류 알림 표시
            showNotification('모델 설치 실패', 'error');
        }
    })
    .catch(error => {
        console.error('Error installing model:', error);
        
        // 버튼 상태 복구
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-download"></i> 설치';
        
        // 오류 알림 표시
        showNotification('모델 설치 중 오류가 발생했습니다.', 'error');
    });
}

// 설정 템플릿 로드 함수
function loadSettingsTemplates() {
    fetch('/api/settings/list')
    .then(response => response.json())
    .then(data => {
        templateSelect.innerHTML = '<option value="">템플릿 선택</option>';
        
        data.forEach(template => {
            const option = document.createElement('option');
            option.value = template;
            option.textContent = template;
            templateSelect.appendChild(option);
        });
    })
    .catch(error => console.error('Error loading settings templates:', error));
}

// 현재 설정 로드 함수
function loadCurrentSettings() {
    fetch('/api/settings/current')
    .then(response => response.json())
    .then(data => {
        updateSettingsFromData(data);
    })
    .catch(error => console.error('Error loading current settings:', error));
}

// 설정 데이터로 UI 업데이트 함수
function updateSettingsFromData(data) {
    // 입력 토큰 수
    maxInputTokens.value = data.max_input_tokens || 4096;
    
    // 출력 토큰 수
    maxOutputTokens.value = data.max_output_tokens || 2048;
    
    // 총 토큰 수
    maxTotalTokens.value = data.max_total_tokens || 8192;
    
    // Temperature
    temperature.value = data.temperature || 0.7;
    temperatureValue.textContent = temperature.value;
    
    // Top P
    topP.value = data.top_p || 0.9;
    topPValue.textContent = topP.value;
    
    // Top K
    topK.value = data.top_k || 50;
    
    // Repetition Penalty
    repetitionPenalty.value = data.repetition_penalty || 1.1;
    repetitionPenaltyValue.textContent = repetitionPenalty.value;
    
    // Do Sample
    doSample.checked = data.do_sample !== undefined ? data.do_sample : true;
    
    // Show Thinking
    showThinking.checked = data.show_thinking || false;
    updateThinkButtonState();
    
    // 디바이스 타입
    if (data.device_type === 'cpu') {
        deviceCPU.checked = true;
        deviceGPU.checked = false;
    } else {
        deviceGPU.checked = true;
        deviceCPU.checked = false;
    }
    
    // 양자화 사용
    useQuantization.checked = data.use_quantization !== undefined ? data.use_quantization : true;
    
    // 양자화 비트
    quantizationBits.value = data.quantization_bits || 8;
    
    // 양자화 옵션 표시 여부
    quantizationOptionsDiv.style.display = useQuantization.checked && deviceGPU.checked ? 'block' : 'none';
}

// 설정 업데이트 함수
function updateSettings(callback) {
    const settings = {
        max_input_tokens: parseInt(maxInputTokens.value),
        max_output_tokens: parseInt(maxOutputTokens.value),
        max_total_tokens: parseInt(maxTotalTokens.value),
        temperature: parseFloat(temperature.value),
        top_p: parseFloat(topP.value),
        top_k: parseInt(topK.value),
        repetition_penalty: parseFloat(repetitionPenalty.value),
        do_sample: doSample.checked,
        show_thinking: showThinking.checked,
        device_type: deviceGPU.checked ? 'gpu' : 'cpu',
        use_quantization: useQuantization.checked,
        quantization_bits: parseInt(quantizationBits.value)
    };
    
    fetch('/api/settings/update', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ settings })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('설정이 업데이트되었습니다.');
            
            // 콜백 함수가 있으면 실행
            if (callback && typeof callback === 'function') {
                callback();
            }
        }
    })
    .catch(error => console.error('Error updating settings:', error));
}

// 설정 템플릿 저장 함수
function saveSettingsTemplate() {
    const name = templateName.value.trim();
    
    if (!name) {
        showNotification('템플릿 이름을 입력하세요.', 'warning');
        return;
    }
    
    const settings = {
        max_input_tokens: parseInt(maxInputTokens.value),
        max_output_tokens: parseInt(maxOutputTokens.value),
        max_total_tokens: parseInt(maxTotalTokens.value),
        temperature: parseFloat(temperature.value),
        top_p: parseFloat(topP.value),
        top_k: parseInt(topK.value),
        repetition_penalty: parseFloat(repetitionPenalty.value),
        do_sample: doSample.checked,
        show_thinking: showThinking.checked,
        device_type: deviceGPU.checked ? 'gpu' : 'cpu',
        use_quantization: useQuantization.checked,
        quantization_bits: parseInt(quantizationBits.value)
    };
    
    fetch('/api/settings/save', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ name, settings })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 템플릿 이름 초기화
            templateName.value = '';
            
            // 템플릿 목록 다시 로드
            loadSettingsTemplates();
            
            // 성공 알림 표시
            showNotification('템플릿이 저장되었습니다.', 'success');
        } else {
            // 오류 알림 표시
            showNotification('템플릿 저장 실패', 'error');
        }
    })
    .catch(error => {
        console.error('Error saving template:', error);
        showNotification('템플릿 저장 중 오류가 발생했습니다.', 'error');
    });
}

// 설정 템플릿 로드 함수
function loadSettingsTemplate() {
    const name = templateSelect.value;
    
    if (!name) {
        showNotification('로드할 템플릿을 선택하세요.', 'warning');
        return;
    }
    
    fetch('/api/settings/load', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ name })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 설정 업데이트
            updateSettingsFromData(data.settings);
            
            // 성공 알림 표시
            showNotification('템플릿이 로드되었습니다.', 'success');
        } else {
            // 오류 알림 표시
            showNotification('템플릿 로드 실패', 'error');
        }
    })
    .catch(error => {
        console.error('Error loading template:', error);
        showNotification('템플릿 로드 중 오류가 발생했습니다.', 'error');
    });
}

// 모델 기본값으로 설정 초기화 함수
function resetToModelDefaults() {
    if (!currentModelId) {
        showNotification('먼저 모델을 선택하세요.', 'warning');
        return;
    }
    
    fetch(`/api/models/settings?model_id=${currentModelId}`)
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 설정 업데이트
            updateSettingsFromData(data.settings);
            
            // 설정 서버에 저장
            updateSettings();
            
            // 성공 알림 표시
            showNotification('설정이 기본값으로 초기화되었습니다.', 'success');
        } else {
            // 오류 알림 표시
            showNotification('설정 초기화 실패', 'error');
        }
    })
    .catch(error => {
        console.error('Error resetting settings:', error);
        showNotification('설정 초기화 중 오류가 발생했습니다.', 'error');
    });
}
