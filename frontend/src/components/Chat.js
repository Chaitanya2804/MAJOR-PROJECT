// frontend/src/components/Chat.js
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function Chat({ user, setUser, darkMode }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [model, setModel] = useState('medicinal');
  const [history, setHistory] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const fileInputRef = useRef();
  const navigate = useNavigate();

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('http://localhost:5000/api/chat/history', {
        headers: { 'x-auth-token': token }
      });
      setHistory(response.data);
    } catch (err) {
      console.error('Error fetching history:', err);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    navigate('/');
  };

  const handleFileSelect = (e) => {
    setSelectedFiles(Array.from(e.target.files));
  };

  const sendMessage = async () => {
    if (!input.trim() && selectedFiles.length === 0) return;

    const newMessage = { 
      text: input, 
      sender: 'user', 
      files: selectedFiles.map(f => f.name),
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, newMessage]);
    setInput('');
    
    try {
      const token = localStorage.getItem('token');
      const formData = new FormData();
      formData.append('message', input);
      formData.append('model', model);
      
      selectedFiles.forEach(file => {
        formData.append('files', file);
      });

      const response = await axios.post(
        'http://localhost:5000/api/chat/message',
        formData,
        {
          headers: {
            'x-auth-token': token,
            'Content-Type': 'multipart/form-data'
          }
        }
      );

      const aiMessage = {
        text: response.data.response,
        sender: 'ai',
        model: model,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
      setSelectedFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = '';
      
      fetchHistory(); // Refresh history
    } catch (err) {
      console.error('Error sending message:', err);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-sidebar">
        <div className="sidebar-header">
          <h3>🏥 VITAS-AI</h3>
          <p>Welcome, {user.name}</p>
        </div>
        ``
        <div className="chat-history">
          <h4>Chat History</h4>
          <div className="history-list">
            {history.map((chat, index) => (
              <div key={index} className="history-item">
                <p>{chat.message.substring(0, 30)}...</p>
                <span className={`model-tag ${chat.model}`}>
                  {chat.model}
                </span>
                <small>{new Date(chat.timestamp).toLocaleDateString()}</small>
              </div>
            ))}
          </div>
        </div>
        
        <div className="sidebar-footer">
          <button onClick={() => navigate('/')} className="btn-secondary">
            🏠 Home
          </button>
          <button onClick={handleLogout} className="btn-secondary">
            Logout
          </button>
        </div>
      </div>

      <div className="chat-main">
        <div className="chat-header">
          <h3>Healthcare Assistant</h3>
          <div className="model-switch">
            <label>
              <input
                type="checkbox"
                checked={model === 'ayurvedic'}
                onChange={(e) => setModel(e.target.checked ? 'ayurvedic' : 'medicinal')}
              />
              <span className="slider"></span>
            </label>
            <span className="model-label">
              {model === 'medicinal' ? '🏥 Medical' : '🌿 Ayurvedic'}
            </span>
          </div>
        </div>

        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <div className="message-content">
                <p>{msg.text}</p>
                {msg.files && msg.files.length > 0 && (
                  <div className="files">
                    {msg.files.map((file, i) => (
                      <span key={i} className="file-tag">📎 {file}</span>
                    ))}
                  </div>
                )}
                {msg.model && (
                  <span className={`model-tag ${msg.model}`}>{msg.model}</span>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="chat-input">
          <div className="input-container">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about your health concerns..."
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            />
            
            <div className="file-controls">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.jpg,.jpeg,.png"
                multiple
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              
              <button 
                onClick={() => fileInputRef.current?.click()}
                className="file-btn"
                title="Upload Files"
              >
                📎
              </button>
              
              <button 
                onClick={sendMessage}
                className="send-btn"
                title="Send Message"
              >
                ➤
              </button>
            </div>
          </div>
          
          {selectedFiles.length > 0 && (
            <div className="selected-files">
              {selectedFiles.map((file, index) => (
                <span key={index} className="selected-file">
                  {file.name}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Chat;