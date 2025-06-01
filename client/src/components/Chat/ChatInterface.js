import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import GraphVisualization from './GraphVisualization';
import { useAuth } from '../../contexts/AuthContext';

function ChatInterface({ user }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [showCoT, setShowCoT] = useState(false);
  const [showGoT, setShowGoT] = useState(false);
  const [currentQuestionId, setCurrentQuestionId] = useState(null);
  const [chatId, setChatId] = useState(null);
  const [cotData, setCotData] = useState([]);
  const [gotData, setGotData] = useState(null);
  const messagesEndRef = useRef(null);
  const { checkGuestLimit, guestSessions } = useAuth(); // Destructure guestSessions

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!user) {
      checkGuestLimit().then((canProceed) => {
        if (!canProceed) {
          setMessages((prev) => [
            ...prev,
            { type: 'bot', content: 'Guest limit reached. Please log in to continue.' },
          ]);
        }
      });
    }
  }, [user, checkGuestLimit]);

  const handleSend = async () => {
    if (!input.trim()) return;

    setMessages([...messages, { type: 'user', content: input }]);
    
    try {
      const res = await axios.post('/api/chat', { message: input, chatId });
      const { response, questionId, chatId: newChatId, cot, got } = res.data;
      
      setChatId(newChatId);
      setMessages((prev) => [...prev, { type: 'bot', content: response, questionId }]);
      setCurrentQuestionId(questionId);
      setCotData(cot || []);
      setGotData(got || null);
      setInput('');
    } catch (err) {
      setMessages((prev) => [...prev, { type: 'bot', content: 'Error processing your request.' }]);
    }
  };

  const handleAnswer = async (answer) => {
    setMessages((prev) => [...prev, { type: 'user', content: answer }]);
    
    try {
      const res = await axios.post('/api/chat/answer', { questionId: currentQuestionId, answer, chatId });
      const { response, questionId, chatId: newChatId, cot, got } = res.data;
      
      setChatId(newChatId);
      setMessages((prev) => [...prev, { type: 'bot', content: response, questionId }]);
      setCurrentQuestionId(questionId);
      setCotData(cot || []);
      setGotData(got || null);
    } catch (err) {
      setMessages((prev) => [...prev, { type: 'bot', content: 'Error processing your answer.' }]);
    }
  };

  const handleDownloadReport = async () => {
    try {
      const res = await axios.get(`/api/report/${chatId}`, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `medical_report_${chatId}.pdf`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error('Error downloading report:', err);
      setMessages((prev) => [...prev, { type: 'bot', content: 'Error downloading report.' }]);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] bg-white dark:bg-gray-900">
      <div className="flex justify-between p-4 bg-gray-100 dark:bg-gray-800">
        <div className="flex space-x-2">
          <button
            onClick={() => setShowCoT(!showCoT)}
            className="p-2 bg-primary text-white rounded hover:bg-blue-600"
          >
            {showCoT ? 'Hide CoT' : 'Show CoT'}
          </button>
          <button
            onClick={() => setShowGoT(!showGoT)}
            className="p-2 bg-secondary text-white rounded hover:bg-green-600"
          >
            {showGoT ? 'Hide GoT' : 'Show GoT'}
          </button>
          {chatId && (
            <button
              onClick={handleDownloadReport}
              className="p-2 bg-purple-500 text-white rounded hover:bg-purple-600"
            >
              Download Report
            </button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`chat-message-${msg.type}`}
          >
            <strong>{msg.type === 'user' ? 'You' : 'Bot'}:</strong>
            <ReactMarkdown>{msg.content}</ReactMarkdown>
            {msg.questionId && (
              <div className="mt-2 flex space-x-2">
                <button
                  onClick={() => handleAnswer('Yes')}
                  className="p-1 bg-secondary text-white rounded hover:bg-green-600"
                >
                  Yes
                </button>
                <button
                  onClick={() => handleAnswer('No')}
                  className="p-1 bg-danger text-white rounded hover:bg-red-600"
                >
                  No
                </button>
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      {showCoT && cotData.length > 0 && (
        <div className="p-4 cot-block">
          <h3 className="text-lg font-bold text-primary mb-2">Chain of Thought</h3>
          {cotData.map((step, index) => (
            <div key={index} className="mb-2">
              <ReactMarkdown>{`- **${step.step_type}**: ${step.content}`}</ReactMarkdown>
            </div>
          ))}
        </div>
      )}
      {showGoT && gotData && (
        <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded">
          <h3 className="text-lg font-bold text-primary mb-2">Graph of Thought</h3>
          <GraphVisualization graphData={gotData} />
        </div>
      )}
      <div className="p-4 bg-gray-100 dark:bg-gray-800">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          className="w-full p-2 border rounded dark:bg-gray-700 dark:text-white"
          placeholder="Describe your symptoms..."
          disabled={guestSessions >= 3 && !user} // Use guestSessions from useAuth
        />
      </div>
      <div className="disclaimer">
        <strong>Disclaimer:</strong> This is a research prototype and not a real medical diagnostic tool. Always consult a healthcare professional.
      </div>
    </div>
  );
}

export default ChatInterface;