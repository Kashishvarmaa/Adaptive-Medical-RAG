import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { auth, RecaptchaVerifier } from '../../firebase';
import { motion } from 'framer-motion';
import { EnvelopeIcon, LockClosedIcon, PhoneIcon } from '@heroicons/react/24/outline';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [phone, setPhone] = useState('');
  const [otp, setOtp] = useState('');
  const [confirmationResult, setConfirmationResult] = useState(null);
  const [error, setError] = useState('');
  const [showOtp, setShowOtp] = useState(false);
  const { login, googleLogin, phoneLogin } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    window.recaptchaVerifier = new RecaptchaVerifier('recaptcha-container', {
      size: 'invisible',
      callback: () => {},
    }, auth);
    return () => {
      if (window.recaptchaVerifier) {
        window.recaptchaVerifier.clear();
      }
    };
  }, []);

  const handleEmailLogin = async (e) => {
    e.preventDefault();
    try {
      await login(email, password);
      navigate('/home');
    } catch (err) {
      setError(err.message);
    }
  };

  const handleGoogleLogin = async () => {
    try {
      await googleLogin();
      navigate('/home');
    } catch (err) {
      setError(err.message);
    }
  };

  const handlePhoneLogin = async (e) => {
    e.preventDefault();
    try {
      const appVerifier = window.recaptchaVerifier;
      const result = await phoneLogin(phone, appVerifier);
      setConfirmationResult(result);
      setShowOtp(true);
    } catch (err) {
      setError(err.message);
    }
  };

  const verifyOtp = async (e) => {
    e.preventDefault();
    try {
      await confirmationResult.confirm(otp);
      navigate('/home');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="space-y-6">
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-3 bg-red-500/20 border border-red-500/30 text-red-400 rounded-lg backdrop-blur-sm"
        >
          {error}
        </motion.div>
      )}

      <form onSubmit={handleEmailLogin} className="space-y-4">
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-300">
            Email
          </label>
          <div className="relative">
            <EnvelopeIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your email"
              className="w-full pl-10 p-3 bg-black/50 border border-white/10 rounded-lg text-gray-200 placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
              required
            />
          </div>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-300">
            Password
          </label>
          <div className="relative">
            <LockClosedIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              className="w-full pl-10 p-3 bg-black/50 border border-white/10 rounded-lg text-gray-200 placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
              required
            />
          </div>
        </div>

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          type="submit"
          className="w-full p-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 transition shadow-lg"
        >
          Login
        </motion.button>
      </form>

      <div className="relative">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-white/10"></div>
        </div>
        <div className="relative flex justify-center text-sm">
          <span className="px-2 bg-black/30 text-gray-400">Or continue with</span>
        </div>
      </div>

      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={handleGoogleLogin}
        className="w-full p-3 bg-white/10 hover:bg-white/20 text-white rounded-lg transition flex items-center justify-center space-x-2 border border-white/10"
      >
        <svg className="w-5 h-5" viewBox="0 0 24 24">
          <path
            fill="currentColor"
            d="M12.24 10.285V14.4h6.806c-.275 1.765-2.056 5.174-6.806 5.174-4.095 0-7.439-3.389-7.439-7.574s3.345-7.574 7.439-7.574c2.33 0 3.891.989 4.785 1.849l3.254-3.138C18.189 1.186 15.479 0 12.24 0c-6.635 0-12 5.365-12 12s5.365 12 12 12c6.926 0 11.52-4.869 11.52-11.726 0-.788-.085-1.39-.189-1.989H12.24z"
          />
        </svg>
        <span>Sign in with Google</span>
      </motion.button>

      <form onSubmit={showOtp ? verifyOtp : handlePhoneLogin} className="space-y-4">
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-300">
            Phone Number
          </label>
          <div className="relative">
            <PhoneIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="tel"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="+1234567890"
              className="w-full pl-10 p-3 bg-black/50 border border-white/10 rounded-lg text-gray-200 placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
              disabled={showOtp}
            />
          </div>
        </div>

        {showOtp && (
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-300">
              OTP
            </label>
            <div className="relative">
              <LockClosedIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                value={otp}
                onChange={(e) => setOtp(e.target.value)}
                placeholder="Enter OTP"
                className="w-full pl-10 p-3 bg-black/50 border border-white/10 rounded-lg text-gray-200 placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent transition"
              />
            </div>
          </div>
        )}

        <div id="recaptcha-container" className="hidden"></div>
        
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          type="submit"
          className="w-full p-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 transition shadow-lg"
        >
          {showOtp ? 'Verify OTP' : 'Send OTP'}
        </motion.button>
      </form>
    </div>
  );
}

export default Login;
