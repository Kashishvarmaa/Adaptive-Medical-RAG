import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import Login from '../components/Auth/Login';
import Signup from '../components/Auth/Signup';

function LandingPage() {
  const { currentUser } = useAuth();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 dark:bg-gray-900">
      <h1 className="text-4xl font-bold text-primary mb-4">Medical RAG Chatbot</h1>
      <p className="text-lg text-gray-600 dark:text-gray-300 mb-8">
        Get personalized medical insights with our AI-powered chatbot. Try up to 3 chats as a guest!
      </p>
      {currentUser ? (
        <Link to="/home" className="p-3 bg-primary text-white rounded-lg">
          Go to Dashboard
        </Link>
      ) : (
        <div className="w-full max-w-md space-y-4">
          <Login />
          <Signup />
        </div>
      )}
    </div>
  );
}

export default LandingPage;