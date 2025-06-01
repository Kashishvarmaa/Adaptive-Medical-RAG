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
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    const token = await userCredential.user.getIdToken();
    await axios.post('/api/auth/signup', { token });
    setCurrentUser(userCredential.user);
  };

  const login = async (email, password) => {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    const token = await userCredential.user.getIdToken();
    await axios.post('/api/auth/login', { token });
    setCurrentUser(userCredential.user);
  };

  const googleLogin = async () => {
    const userCredential = await signInWithPopup(auth, googleProvider);
    const token = await userCredential.user.getIdToken();
    await axios.post('/api/auth/google', { token });
    setCurrentUser(userCredential.user);
  };

  const phoneLogin = async (phone, appVerifier) => {
    const confirmationResult = await signInWithPhoneNumber(auth, phone, appVerifier);
    return confirmationResult;
  };

  const logout = async () => {
    await auth.signOut();
    setCurrentUser(null);
  };

  const checkGuestLimit = async () => {
    if (!currentUser) {
      try {
        const res = await axios.get('/api/auth/guest-limit');
        setGuestSessions(3 - res.data.remaining);
        return res.data.remaining > 0;
      } catch (err) {
        // Mock for local testing
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