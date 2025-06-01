const jwt = require('jsonwebtoken');

const verifyToken = async (req, res, next) => {
  const token = req.headers.authorization?.split('Bearer ')[1];
  if (!token) {
    return res.status(401).json({ message: 'No token provided' });
  }

  try {
    // In a production app, verify the Firebase token with Firebase Admin SDK
    // For this example, we'll use a simple JWT verification with a shared secret
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    console.error('Token verification error:', err);
    res.status(401).json({ message: 'Invalid token' });
  }
};

module.exports = verifyToken;