// frontend/src/components/Home.js
import React from 'react';
import { useNavigate } from 'react-router-dom';

function Home({ user, setUser, darkMode, setDarkMode }) {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    navigate('/');
  };

  return (
    <div className="home">
      <nav className="navbar">
        <div className="nav-brand">
          <h2>🏥 VITAS-AI</h2>
        </div>
        <div className="nav-links">
          <a href="#features">Features</a>
          <a href="#about">About</a>
          <button 
            className="theme-toggle"
            onClick={() => setDarkMode(!darkMode)}
          >
            {darkMode ? '☀️' : '🌙'}
          </button>
          {user ? (
            <>
              <span>Welcome, {user.name}</span>
              <button onClick={() => navigate('/chat')} className="btn-primary">
                💬 Chat Now
              </button>
              <button onClick={handleLogout} className="btn-secondary">
                Logout
              </button>
            </>
          ) : (
            <button onClick={() => navigate('/auth')} className="btn-primary">
              Login / Sign Up
            </button>
          )}
        </div>
      </nav>

      <section className="hero">
        <div className="hero-content">
          <h1>Welcome to VITAS-AI</h1>
          <p>Your AI-Powered Healthcare Assistant</p>
          <p>Get instant medical and ayurvedic solutions for your health concerns</p>
          {user ? (
            <button onClick={() => navigate('/chat')} className="btn-hero">
              Start Chatting 💬
            </button>
          ) : (
            <button onClick={() => navigate('/auth')} className="btn-hero">
              Get Started
            </button>
          )}
        </div>
      </section>

      <section id="features" className="features">
        <h2>Features</h2>
        <div className="feature-grid">
          <div className="feature-card">
            <h3>🏥 Medical Solutions</h3>
            <p>Get evidence-based medical advice and treatment recommendations</p>
          </div>
          <div className="feature-card">
            <h3>🌿 Ayurvedic Remedies</h3>
            <p>Discover traditional ayurvedic treatments and natural healing methods</p>
          </div>
          <div className="feature-card">
            <h3>📁 Multi-Input Support</h3>
            <p>Upload text, images, and PDF documents for comprehensive analysis</p>
          </div>
          <div className="feature-card">
            <h3>💾 Chat History</h3>
            <p>Keep track of all your consultations and recommendations</p>
          </div>
        </div>
      </section>

      <section id="about" className="about">
        <h2>About VITAS-AI</h2>
        <p>
          VITAS-AI combines modern medical knowledge with traditional ayurvedic wisdom 
          to provide comprehensive healthcare guidance. Our AI assistant helps you 
          understand symptoms, explore treatment options, and make informed health decisions.
        </p>
      </section>

      <footer className="footer">
        <p>&copy; 2024 VITAS-AI. Your health, our priority.</p>
      </footer>
    </div>
  );
}

export default Home;