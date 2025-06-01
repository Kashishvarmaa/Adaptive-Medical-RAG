import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import ChatInterface from '../components/Chat/ChatInterface';
import ChatHistory from '../components/Chat/ChatHistory';

function HomePage() {
  const { currentUser, logout, checkGuestLimit } = useAuth();
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    const checkLimit = async () => {
      const canProceed = await checkGuestLimit();
      if (!canProceed) {
        alert('Guest limit reached. Please log in to continue.');
        navigate('/');
      }
    };
    checkLimit();
  }, [checkGuestLimit, navigate]);

  return (
    <div className="flex min-h-screen">
      <ChatHistory sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
      <div className="flex-1 p-4">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-primary">Medical Consultation</h2>
          <div className="flex space-x-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 bg-primary text-white rounded"
            >
              {sidebarOpen ? 'Hide Sidebar' : 'Show Sidebar'}
            </button>
            {currentUser && (
              <button onClick={logout} className="p-2 bg-danger text-white rounded">
                Logout
              </button>
            )}
          </div>
        </div>
        <ChatInterface user={currentUser} />
      </div>
    </div>
  );
}

export default HomePage;