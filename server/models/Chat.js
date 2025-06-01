const mongoose = require('mongoose');

const messageSchema = new mongoose.Schema({
  type: { type: String, enum: ['user', 'bot'], required: true },
  content: { type: String, required: true },
  timestamp: { type: Date, default: Date.now },
});

const chatSchema = new mongoose.Schema({
  userId: { type: String, required: true }, // Firebase UID
  messages: [messageSchema],
  chatId: { type: String, required: true, unique: true },
});

module.exports = mongoose.model('Chat', chatSchema);