
# 提供测试页面
@app.get("/test-agent", response_class=HTMLResponse)
async def test_agent_page():
    """返回 WebSocket 测试页面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Knowledge Graph Agent Test</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                border-bottom: 3px solid #007bff;
                padding-bottom: 10px;
            }
            .input-group {
                margin: 20px 0;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #555;
            }
            textarea, input[type="text"] {
                width: 100%;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                box-sizing: border-box;
            }
            textarea {
                min-height: 100px;
                resize: vertical;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin-right: 10px;
            }
            button:hover {
                background-color: #0056b3;
            }
            button:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }
            .approval-btn {
                background-color: #28a745;
            }
            .approval-btn:hover {
                background-color: #218838;
            }
            .reject-btn {
                background-color: #dc3545;
            }
            .reject-btn:hover {
                background-color: #c82333;
            }
            #output {
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 15px;
                margin-top: 20px;
                min-height: 300px;
                max-height: 600px;
                overflow-y: auto;
                font-family: 'Courier New', monospace;
                font-size: 13px;
            }
            .message {
                margin: 8px 0;
                padding: 8px;
                border-radius: 4px;
            }
            .agent-msg {
                background-color: #e3f2fd;
                border-left: 4px solid #2196f3;
            }
            .tool-msg {
                background-color: #fff3e0;
                border-left: 4px solid #ff9800;
            }
            .approval-req {
                background-color: #fff9c4;
                border-left: 4px solid #fbc02d;
                padding: 15px;
                margin: 10px 0;
            }
            .system-msg {
                background-color: #e8f5e9;
                border-left: 4px solid #4caf50;
            }
            .error-msg {
                background-color: #ffebee;
                border-left: 4px solid #f44336;
            }
            .debug-msg {
                background-color: #f3e5f5;
                border-left: 4px solid #9c27b0;
                font-size: 11px;
            }
            .status {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
                margin-left: 10px;
            }
            .status-connected {
                background-color: #4caf50;
                color: white;
            }
            .status-disconnected {
                background-color: #f44336;
                color: white;
            }
            .approval-buttons {
                display: none;
                margin-top: 10px;
            }
            .checkbox-group {
                margin: 15px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Knowledge Graph Agent WebSocket 测试</h1>

            <div class="input-group">
                <label for="userInput">用户输入：</label>
                <textarea id="userInput" placeholder="例如：I'd like a bill of materials graph (BOM graph)...">I'd like a bill of materials graph (BOM graph) which includes all levels from suppliers to finished product, which can support root-cause analysis.</textarea>
            </div>

            <div class="checkbox-group">
                <label>
                    <input type="checkbox" id="debugMode"> 启用 Debug 模式
                </label>
            </div>

            <div class="input-group">
                <button id="startBtn" onclick="startWorkflow()">▶️ 启动工作流</button>
                <button id="connectBtn" onclick="connect()">🔌 连接 WebSocket</button>
                <button id="disconnectBtn" onclick="disconnect()" disabled>🔌 断开连接</button>
                <button onclick="clearOutput()">🗑️ 清空输出</button>
                <span id="status" class="status status-disconnected">未连接</span>
            </div>

            <div id="approvalButtons" class="approval-buttons">
                <h3>⏸️ 需要人工审批</h3>
                <div id="approvalInfo"></div>
                <button class="approval-btn" onclick="sendApproval(true)">✅ 批准</button>
                <button class="reject-btn" onclick="sendApproval(false)">❌ 拒绝</button>
            </div>

            <div id="output"></div>
        </div>

        <script>
            let ws = null;

            function connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/agent/knowledge-graph`;

                ws = new WebSocket(wsUrl);

                ws.onopen = () => {
                    addMessage('system-msg', '✅ WebSocket 连接成功');
                    updateStatus(true);
                };

                ws.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    handleMessage(message);
                };

                ws.onerror = (error) => {
                    addMessage('error-msg', `❌ WebSocket 错误: ${error}`);
                };

                ws.onclose = () => {
                    addMessage('system-msg', '🔌 WebSocket 连接已断开');
                    updateStatus(false);
                };
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }

            function updateStatus(connected) {
                const status = document.getElementById('status');
                const startBtn = document.getElementById('startBtn');
                const connectBtn = document.getElementById('connectBtn');
                const disconnectBtn = document.getElementById('disconnectBtn');

                if (connected) {
                    status.textContent = '已连接';
                    status.className = 'status status-connected';
                    startBtn.disabled = false;
                    connectBtn.disabled = true;
                    disconnectBtn.disabled = false;
                } else {
                    status.textContent = '未连接';
                    status.className = 'status status-disconnected';
                    startBtn.disabled = true;
                    connectBtn.disabled = false;
                    disconnectBtn.disabled = true;
                }
            }

            function startWorkflow() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    alert('请先连接 WebSocket');
                    return;
                }

                const userInput = document.getElementById('userInput').value;
                const debug = document.getElementById('debugMode').checked;

                if (!userInput.trim()) {
                    alert('请输入用户问题');
                    return;
                }

                ws.send(JSON.stringify({
                    type: 'start',
                    data: {
                        user_input: userInput,
                        debug: debug
                    }
                }));

                addMessage('system-msg', '🚀 工作流已启动...');
                hideApprovalButtons();
            }

            function sendApproval(approved) {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    alert('WebSocket 未连接');
                    return;
                }

                ws.send(JSON.stringify({
                    type: 'approval_response',
                    data: {
                        approved: approved
                    }
                }));

                addMessage('system-msg', approved ? '✅ 已发送批准响应' : '❌ 已发送拒绝响应');
                hideApprovalButtons();
            }

            function handleMessage(message) {
                const type = message.type;
                const data = message.data;

                switch(type) {
                    case 'workflow_start':
                        addMessage('system-msg', `🚀 ${data.message} (Thread ID: ${data.thread_id})`);
                        break;

                    case 'agent_message':
                        addMessage('agent-msg', `🤖 Agent: ${data.content}`);
                        break;

                    case 'tool_message':
                        addMessage('tool-msg', `🔧 Tool: ${data.content}`);
                        break;

                    case 'approval_request':
                        showApprovalRequest(data);
                        break;

                    case 'approval_result':
                        addMessage('system-msg', `${data.approved ? '✅' : '❌'} ${data.message}`);
                        break;

                    case 'workflow_complete':
                        addMessage('system-msg', `✅ ${data.message} (Agent: ${data.agent_responses_count}, Tool: ${data.tool_responses_count}, 迭代: ${data.iterations})`);
                        break;

                    case 'debug':
                        addMessage('debug-msg', `[DEBUG] ${JSON.stringify(data, null, 2)}`);
                        break;

                    case 'warning':
                        addMessage('error-msg', `⚠️ ${data.message}`);
                        break;

                    case 'error':
                        addMessage('error-msg', `❌ 错误: ${data.message}`);
                        break;

                    default:
                        addMessage('debug-msg', `未知消息类型: ${type}`);
                }
            }

            function showApprovalRequest(data) {
                const approvalDiv = document.getElementById('approvalButtons');
                const infoDiv = document.getElementById('approvalInfo');

                let infoHtml = `<p><strong>${data.question || '需要审批'}</strong></p>`;

                if (data.tool_calls) {
                    infoHtml += '<p><strong>工具调用:</strong></p><ul>';
                    data.tool_calls.forEach((tc, i) => {
                        infoHtml += `<li><strong>${tc.tool_name}</strong><br>`;
                        infoHtml += `参数: ${JSON.stringify(tc.arguments, null, 2)}</li>`;
                    });
                    infoHtml += '</ul>';
                }

                if (data.tool_results) {
                    infoHtml += '<p><strong>工具结果:</strong></p><ul>';
                    data.tool_results.forEach((tr, i) => {
                        infoHtml += `<li><strong>${tr.tool_name}</strong><br>`;
                        infoHtml += `结果: ${tr.result}</li>`;
                    });
                    infoHtml += '</ul>';
                }

                infoDiv.innerHTML = infoHtml;
                approvalDiv.style.display = 'block';

                addMessage('approval-req', '⏸️ 工作流需要人工审批，请在上方选择批准或拒绝');
            }

            function hideApprovalButtons() {
                document.getElementById('approvalButtons').style.display = 'none';
            }

            function addMessage(className, content) {
                const output = document.getElementById('output');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${className}`;
                messageDiv.textContent = content;
                output.appendChild(messageDiv);
                output.scrollTop = output.scrollHeight;
            }

            function clearOutput() {
                document.getElementById('output').innerHTML = '';
                hideApprovalButtons();
            }

            // 页面加载时自动连接
            window.onload = () => {
                connect();
            };
        </script>
    </body>
    </html>
    """
