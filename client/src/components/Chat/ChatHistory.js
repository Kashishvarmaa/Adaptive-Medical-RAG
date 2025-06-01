import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { UserCircleIcon, DocumentTextIcon } from '@heroicons/react/24/solid';

function ChatHistory({ sidebarOpen, setSidebarOpen }) {
  const [chats, setChats] = useState([]);
  const [selectedChat, setSelectedChat] = useState(null);

  useEffect(() => {
    const fetchChats = async () => {
      try {
        const res = await axios.get('/api/chat/history');
        setChats(res.data);
      } catch (err) {
        console.error('Error fetching chat history:', err);
      }
    };
    fetchChats();
  }, []);

  const handleSelectChat = (chatId) => {
    setSelectedChat(chatId);
    // Optionally fetch and display chat details
  };

  return (
    <div
      className={`fixed inset-y-0 left-0 w-64 bg-white dark:bg-gray-800 p-4 transform ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      } transition-transform duration-300 ease-in-out`}
    >
      <h3 className="text-lg font-bold text-primary mb-4">Chat History</h3>
      {chats.length === 0 ? (
        <p className="text-gray-500 dark:text-gray-400">No chats yet.</p>
      ) : (
        <ul>
          {chats.map((chat) => (
            <li
              key={chat._id}
              onClick={() => handleSelectChat(chat._id)}
              className={`p-2 mb-2 rounded cursor-pointer ${
                selectedChat === chat._id ? 'bg-primary text-white' : 'hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <DocumentTextIcon className="h-5 w-5 inline mr-2" />
              Chat {new Date(chat.createdAt).toLocaleDateString()}
            </li>
          ))}
        </ul>
      )}
      <div className="mt-4">
        <h4 className="text-md font-bold text-primary mb-2">Future Features</h4>
        <ul className="text-gray-500 dark:text-gray-400">
          <li>Saved Searches</li>
          <li>Favorite Resources</li>
          <li>User Analytics</li>
        </ul>
      </div>
      <div className="mt-4">
        <UserCircleIcon className="h-6 w-6 inline mr-2" />
        <span>Profile & Settings</span>
      </div>
    </div>
  );
}

export default ChatHistory;