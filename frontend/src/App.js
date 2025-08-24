// frontend/src/App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Home from './components/Home';
import Chat from './components/Chat';
import Auth from './components/Auth';
import './App.css';

function App() {
  const [user, setUser] = useState(null);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    if (token && userData) {
      setUser(JSON.parse(userData));
    }
    
    const savedTheme = localStorage.getItem('darkMode');
    if (savedTheme) {
      setDarkMode(JSON.parse(savedTheme));
    }
  }, []);

  useEffect(() => {
    document.body.className = darkMode ? 'dark' : 'light';
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  return (
    <div className={`App ${darkMode ? 'dark' : 'light'}`}>
      <Router>
        <Routes>
          <Route path="/" element={
            <Home 
              user={user} 
              setUser={setUser} 
              darkMode={darkMode} 
              setDarkMode={setDarkMode} 
            />
          } />
          <Route path="/auth" element={
            user ? <Navigate to="/" /> : <Auth setUser={setUser} darkMode={darkMode} />
          } />
          <Route path="/chat" element={
            user ? <Chat user={user} setUser={setUser} darkMode={darkMode} /> : <Navigate to="/auth" />
          } />
        </Routes>
      </Router>
    </div>
  );
}

export default App;
