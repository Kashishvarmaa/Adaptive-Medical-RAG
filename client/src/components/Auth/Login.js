import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { PhoneIcon } from '@heroicons/react/24/solid';
import Recaptcha from 'react-google-recaptcha';

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
    <div className="max-w-md mx-auto p-4 bg-white dark:bg-gray-800 rounded-lg shadow-md">
      <h2 className="text-2xl font-bold text-primary mb-4">Login</h2>
      {error && <p className="text-danger mb-4">{error}</p>}
      <form onSubmit={handleEmailLogin} className="mb-4">
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          className="w-full p-2 mb-2 border rounded dark:bg-gray-700 dark:text-white"
          required
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          className="w-full p-2 mb-2 border rounded dark:bg-gray-700 dark:text-white"
          required
        />
        <button type="submit" className="w-full p-2 bg-primary text-white rounded">
          Login
        </button>
      </form>
      <button
        onClick={handleGoogleLogin}
        className="w-full p-2 bg-red-500 text-white rounded mb-2"
      >
        Sign in with Google
      </button>
      <form onSubmit={showOtp ? verifyOtp : handlePhoneLogin}>
        <input
          type="tel"
          value={phone}
          onChange={(e) => setPhone(e.target.value)}
          placeholder="Phone Number"
          className="w-full p-2 mb-2 border rounded dark:bg-gray-700 dark:text-white"
          disabled={showOtp}
        />
        {showOtp && (
          <input
            type="text"
            value={otp}
            onChange={(e) => setOtp(e.target.value)}
            placeholder="Enter OTP"
            className="w-full p-2 mb-2 border rounded dark:bg-gray-700 dark:text-white"
          />
        )}
        <Recaptcha
          sitekey={process.env.REACT_APP_RECAPTCHA_SITE_KEY}
          render="explicit"
          onLoad={() => {
            window.recaptchaVerifier = new firebase.auth.RecaptchaVerifier('recaptcha-container', {
              size: 'invisible',
            });
          }}
        />
        <div id="recaptcha-container"></div>
        <button type="submit" className="w-full p-2 bg-secondary text-white rounded">
          {showOtp ? 'Verify OTP' : 'Send OTP'}
        </button>
      </form>
    </div>
  );
}

export default Login;