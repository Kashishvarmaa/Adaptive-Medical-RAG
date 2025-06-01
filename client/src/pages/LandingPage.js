import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import Login from '../components/Auth/Login';
import Signup from '../components/Auth/Signup';
import { HeartIcon, ArrowRightIcon, SparklesIcon } from '@heroicons/react/24/solid';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import { motion } from 'framer-motion';

function LandingPage() {
  const [activeTab, setActiveTab] = useState('login');
  const [showAuth, setShowAuth] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);
  const { checkGuestLimit, currentUser } = useAuth();
  const navigate = useNavigate();

  // Scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Check guest limit on mount
  useEffect(() => {
    if (!currentUser) {
      const sessionId = localStorage.getItem('sessionId');
      const headers = sessionId ? { 'X-Session-ID': sessionId } : {};
      checkGuestLimit(headers).then((canProceed) => {
        if (!canProceed) {
          setShowAuth(true);
          setMessages((prev) => [
            ...prev,
            { type: 'bot', content: 'Guest limit reached. Please log in to continue.' },
          ]);
        }
      });
    }
  }, [checkGuestLimit, currentUser]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const sessionId = localStorage.getItem('sessionId');
    const headers = sessionId ? { 'X-Session-ID': sessionId } : {};

    const canProceed = await checkGuestLimit(headers);
    if (!canProceed) {
      setShowAuth(true);
      setMessages((prev) => [
        ...prev,
        { type: 'bot', content: 'Guest limit reached. Please log in to continue.' },
      ]);
      return;
    }

    setMessages([...messages, { type: 'user', content: input }]);

    try {
      const res = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/chat`,
        { message: input, chatId: null },
        { headers }
      );
      const { content } = res.data;
      const newSessionId = res.headers['x-session-id'];
      if (newSessionId) {
        localStorage.setItem('sessionId', newSessionId);
      }
      setMessages((prev) => [...prev, { type: 'bot', content }]);
      setInput('');
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { type: 'bot', content: 'Error processing your request. Please try again.' },
      ]);
    }
  };

  const handleLoginClick = () => {
    setShowAuth(true);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0f0c29] via-[#302b63] to-[#24243e] flex flex-col overflow-hidden relative">
      {/* Galaxy Stars Background */}
      <div className="absolute inset-0 overflow-hidden">
        {[...Array(50)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-white rounded-full"
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
              scale: Math.random() * 2,
              opacity: Math.random(),
            }}
            animate={{
              opacity: [0, 1, 0],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration: Math.random() * 3 + 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>

      {/* Header */}
      <motion.header
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ type: 'spring', stiffness: 100 }}
        className="p-4 bg-black/20 backdrop-blur-md shadow-lg border-b border-white/10"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              setShowAuth(false);
              setActiveTab('login');
            }}
            className="flex items-center space-x-2 cursor-pointer"
          >
            <SparklesIcon className="h-8 w-8 text-purple-400 animate-pulse" />
            <h1 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
              Medical RAG Chatbot
            </h1>
          </motion.div>
          {!currentUser && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleLoginClick}
              className="px-6 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-full hover:from-purple-600 hover:to-pink-600 transition shadow-lg"
            >
              Login
            </motion.button>
          )}
        </div>
      </motion.header>

      {/* Main Section */}
      <main className="flex-1 flex items-center justify-center p-4 relative">
        {!showAuth ? (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center max-w-3xl w-full"
          >
            <motion.div
              className="mb-8"
              animate={{ scale: [1, 1.02, 1] }}
              transition={{ repeat: Infinity, duration: 2 }}
            >
              <h2 className="text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400">
                Medical RAG Chatbot
              </h2>
              <p className="text-xl text-gray-300 mt-4">
                Your AI-powered medical assistant for research and insights. Try it now!
              </p>
            </motion.div>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="bg-black/30 backdrop-blur-xl rounded-2xl shadow-2xl p-6 border border-white/10"
            >
              <div className="h-100 overflow-y-auto mb-4 p-4 bg-gradient-to-b from-purple-900/50 to-pink-900/50 rounded-lg">
                {messages.length === 0 && (
                  <div className="text-gray-300 text-center">
                    Start chatting to explore medical insights!
                  </div>
                )}
                {messages.map((msg, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: msg.type === 'user' ? 20 : -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`chat-message-${msg.type} mb-2 flex items-start space-x-2`}
                  >
                    <div className={`p-1 rounded-full ${msg.type === 'user' ? 'bg-purple-500' : 'bg-pink-500'}`}>
                      {msg.type === 'user' ? (
                        <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M10 10a4 4 0 100-8 4 4 0 000 8zm-5 7a5 5 0 0110 0H5z" />
                        </svg>
                      ) : (
                        <SparklesIcon className="w-4 h-4 text-white" />
                      )}
                    </div>
                    <ReactMarkdown className="flex-1 text-gray-200">{msg.content}</ReactMarkdown>
                  </motion.div>
                ))}
                <div ref={messagesEndRef} />
              </div>
              <div className="flex items-center">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="Ask a medical question..."
                  className="flex-1 p-3 border-0 rounded-l-lg bg-black/50 text-gray-200 placeholder-gray-400 focus:ring-2 focus:ring-purple-500 shadow-inner"
                  disabled={messages.filter((msg) => msg.type === 'user').length >= 3 && !currentUser}
                />
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={handleSend}
                  className="p-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-r-lg hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 shadow-md"
                  disabled={messages.filter((msg) => msg.type === 'user').length >= 3 && !currentUser}
                >
                  <ArrowRightIcon className="h-5 w-5" />
                </motion.button>
              </div>
              {messages.filter((msg) => msg.type === 'user').length > 0 && !currentUser && (
                <p className="text-sm text-gray-400 mt-2">
                  {3 - messages.filter((msg) => msg.type === 'user').length} guest attempts remaining.
                </p>
              )}
            </motion.div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="bg-black/30 backdrop-blur-xl rounded-2xl shadow-2xl p-8 w-full max-w-md border border-white/10"
          >
            <div className="flex border-b border-white/10 mb-6">
              <button
                type="button"
                className={`flex-1 py-2 text-center font-medium text-lg ${
                  activeTab === 'login'
                    ? 'text-purple-400 border-b-2 border-purple-400'
                    : 'text-gray-400 hover:text-purple-400'
                }`}
                onClick={() => setActiveTab('login')}
              >
                Login
              </button>
              <button
                type="button"
                className={`flex-1 py-2 text-center font-medium text-lg ${
                  activeTab === 'signup'
                    ? 'text-purple-400 border-b-2 border-purple-400'
                    : 'text-gray-400 hover:text-purple-400'
                }`}
                onClick={() => setActiveTab('signup')}
              >
                Signup
              </button>
            </div>
            {activeTab === 'login' ? <Login /> : <Signup />}
          </motion.div>
        )}
      </main>

      {/* Footer */}
      <motion.footer
        initial={{ y: 50 }}
        animate={{ y: 0 }}
        transition={{ delay: 0.8 }}
        className="bg-black/20 backdrop-blur-md p-4 flex justify-between items-center border-t border-white/10"
      >
        <div className="text-sm text-gray-400">
          <strong>Disclaimer:</strong> This is a research prototype, not a medical diagnostic tool. Always consult a healthcare professional.
        </div>
      </motion.footer>
    </div>
  );
}

export default LandingPage;