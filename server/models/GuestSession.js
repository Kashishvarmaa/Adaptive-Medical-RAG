const mongoose = require('mongoose');

const guestSessionSchema = new mongoose.Schema({
  sessionId: { type: String, required: true, unique: true },
  messageCount: { type: Number, default: 0 },
  createdAt: { type: Date, default: Date.now, expires: '24h' }, // Expire after 24 hours
});

module.exports = mongoose.model('GuestSession', guestSessionSchema);