import React, { createContext, useContext, useState, useEffect } from 'react';
import { auth, signInWithPopup, googleProvider, signInWithEmailAndPassword, createUserWithEmailAndPassword, signInWithPhoneNumber } from '../firebase';
import axios from 'axios';

const AuthContext = createContext();

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);
  const [guestSessions, setGuestSessions] = useState(0);
  const [loading, setLoading] = useState(true);

  const signup = async (email, password) => {
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const token = await userCredential.user.getIdToken();
      await axios.post(`${process.env.REACT_APP_API_URL}/api/auth/signup`, { token });
      setCurrentUser(userCredential.user);
    } catch (err) {
      console.error('Signup error:', err);
      throw err; // Re-throw to handle in the component
    }
  };

  const login = async (email, password) => {
    try {
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      const token = await userCredential.user.getIdToken();
      await axios.post(`${process.env.REACT_APP_API_URL}/api/auth/login`, { token });
      setCurrentUser(userCredential.user);
    } catch (err) {
      console.error('Login error:', err);
      throw err; // Re-throw to handle in the component
    }
  };

  const googleLogin = async () => {
    try {
      const userCredential = await signInWithPopup(auth, googleProvider);
      const token = await userCredential.user.getIdToken();
      await axios.post(`${process.env.REACT_APP_API_URL}/api/auth/google`, { token });
      setCurrentUser(userCredential.user);
    } catch (err) {
      console.error('Google login error:', err);
      throw err; // Re-throw to handle in the component
    }
  };

  const phoneLogin = async (phone, appVerifier) => {
    const confirmationResult = await signInWithPhoneNumber(auth, phone, appVerifier);
    return confirmationResult;
  };

  const logout = async () => {
    await auth.signOut();
    setCurrentUser(null);
  };

  const checkGuestLimit = async (headers = {}) => {
    if (!currentUser) {
      try {
        const res = await axios.get(`${process.env.REACT_APP_API_URL}/api/auth/guest-limit`, { headers });
        setGuestSessions(3 - res.data.remaining);
        const newSessionId = res.headers['x-session-id'];
        if (newSessionId) {
          localStorage.setItem('sessionId', newSessionId);
        }
        return res.data.remaining > 0;
      } catch (err) {
        console.error('Guest limit check failed:', err);
        if (guestSessions < 3) {
          setGuestSessions(guestSessions + 1);
          return true;
        }
        return false;
      }
    }
    return true;
  };

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(async (user) => {
      setCurrentUser(user);
      if (user) {
        const token = await user.getIdToken();
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      } else {
        delete axios.defaults.headers.common['Authorization'];
      }
      setLoading(false);
    });
    return unsubscribe;
  }, []);

  const value = {
    currentUser,
    guestSessions,
    signup,
    login,
    googleLogin,
    phoneLogin,
    logout,
    checkGuestLimit,
  };

  return <AuthContext.Provider value={value}>{!loading && children}</AuthContext.Provider>;
}