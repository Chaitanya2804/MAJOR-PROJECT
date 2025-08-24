// backend/routes/chat.js
const express = require('express');
const multer = require('multer');
const path = require('path');
const User = require('../models/User');
const auth = require('../middleware/auth');

const router = express.Router();

// File upload configuration
const storage = multer.diskStorage({
  destination: './uploads/',
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  fileFilter: (req, file, cb) => {
    const filetypes = /jpeg|jpg|png|pdf/;
    const extname = filetypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = filetypes.test(file.mimetype);
    
    if (mimetype && extname) return cb(null, true);
    else cb('Error: Images and PDFs only!');
  }
});

// Send message
router.post('/message', auth, upload.array('files'), async (req, res) => {
  try {
    const { message, model = 'medicinal' } = req.body;
    const userId = req.user.id;
    
    // Mock AI response based on model
    const responses = {
      medicinal: `Medical Analysis: For ${message}, I recommend consulting with a healthcare professional. Common treatments may include proper medication and lifestyle changes.`,
      ayurvedic: `Ayurvedic Analysis: For ${message}, traditional remedies suggest herbal treatments like turmeric, neem, or ashwagandha based on your dosha constitution.`
    };
    
    const response = responses[model] || responses.medicinal;
    
    const files = req.files ? req.files.map(file => ({
      filename: file.originalname,
      path: file.path
    })) : [];
    
    const user = await User.findById(userId);
    user.chatHistory.push({
      message,
      response,
      model,
      files
    });
    
    await user.save();
    
    res.json({ response, files });
  } catch (err) {
    res.status(500).json({ msg: 'Server error' });
  }
});

// Get chat history
router.get('/history', auth, async (req, res) => {
  try {
    const user = await User.findById(req.user.id);
    res.json(user.chatHistory);
  } catch (err) {
    res.status(500).json({ msg: 'Server error' });
  }
});

module.exports = router;