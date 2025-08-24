// backend/models/User.js
const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  chatHistory: [{
    message: String,
    response: String,
    model: { type: String, enum: ['medicinal', 'ayurvedic'], default: 'medicinal' },
    timestamp: { type: Date, default: Date.now },
    files: [{ filename: String, path: String }]
  }]
}, { timestamps: true });

module.exports = mongoose.model('User', UserSchema);