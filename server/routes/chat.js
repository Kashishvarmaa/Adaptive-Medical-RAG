const express = require('express');
const router = express.Router();
const Chat = require('../models/Chat');
const GuestSession = require('../models/GuestSession');
const verifyToken = require('../middleware/auth');
const guestLimiter = require('../middleware/rateLimit');

router.post('/', async (req, res) => {
  const { message, chatId } = req.body;
  const token = req.headers.authorization?.split('Bearer ')[1];
  let sessionId = req.headers['x-session-id'];

  try {
    // Guest user handling
    if (!token) {
      if (!sessionId) {
        sessionId = require('crypto').randomBytes(16).toString('hex');
        res.set('X-Session-ID', sessionId);
      }

      let session = await GuestSession.findOne({ sessionId });
      if (!session) {
        session = new GuestSession({ sessionId });
        await session.save();
      }

      if (session.messageCount >= 3) {
        return res.status(403).json({ message: 'Guest limit reached' });
      }

      session.messageCount += 1;
      await session.save();

      // Mock bot response for guests
      const botResponse = `Echo: ${message}`; // Replace with real AI response later
      res.json({ content: botResponse });
      return;
    }

    // Authenticated user handling
    const decoded = require('jsonwebtoken').verify(token, process.env.JWT_SECRET);
    const userId = decoded.uid;

    let chat = await Chat.findOne({ chatId, userId });
    if (!chat) {
      chat = new Chat({ userId, chatId: chatId || require('crypto').randomBytes(16).toString('hex'), messages: [] });
    }

    chat.messages.push({ type: 'user', content: message });
    const botResponse = `Echo: ${message}`; // Replace with real AI response later
    chat.messages.push({ type: 'bot', content: botResponse });
    await chat.save();

    res.json({ content: botResponse });
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router;