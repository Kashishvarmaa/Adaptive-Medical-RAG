const express = require('express');
const router = express.Router();
const GuestSession = require('../models/GuestSession');
const guestLimiter = require('../middleware/rateLimit');

router.get('/guest-limit', guestLimiter, async (req, res) => {
  try {
    let sessionId = req.headers['x-session-id'];
    if (!sessionId) {
      sessionId = require('crypto').randomBytes(16).toString('hex');
      res.set('X-Session-ID', sessionId);
    }

    let session = await GuestSession.findOne({ sessionId });
    if (!session) {
      session = new GuestSession({ sessionId });
      await session.save();
    }

    // Check if an hour has passed since the last reset
    const now = new Date();
    const lastReset = new Date(session.lastReset);
    const hoursSinceLastReset = (now - lastReset) / (1000 * 60 * 60); // Convert milliseconds to hours

    if (hoursSinceLastReset >= 1) {
      // Reset message count and update lastReset
      session.messageCount = 0;
      session.lastReset = now;
      await session.save();
    }

    session.messageCount = session.messageCount || 0;
    const remaining = Math.max(0, 3 - session.messageCount);
    res.json({ remaining });
  } catch (err) {
    console.error('Guest limit error:', err);
    res.status(500).json({ message: 'Server error' });
  }
});

router.post('/signup', (req, res) => {
  res.json({ message: 'Signup successful' });
});

router.post('/login', (req, res) => {
  res.json({ message: 'Login successful' });
});

router.post('/google', (req, res) => {
  res.json({ message: 'Google login successful' });
});

module.exports = router;