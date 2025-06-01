const rateLimit = require('express-rate-limit');

const guestLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000, // 24 hours
  max: 3, // Limit to 3 requests per session
  keyGenerator: (req) => req.sessionId || req.ip, // Use sessionId or IP as the key
  message: { remaining: 0 },
});

module.exports = guestLimiter;